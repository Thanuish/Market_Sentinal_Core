"""Hybrid retrieval for the RAG evaluator.

Ephemeral by design (see DECISIONS.md): each query scrapes a handful of
fresh articles, chunks and ranks them in memory, and none of it is
persisted -- financial news decays in relevance within hours, so there's
no value in a growing index and real cost (staleness, disk bloat) in
keeping one. A persistent, metadata-filtered store (e.g. for 10-K filings,
which are too expensive to re-parse and re-embed on every ticker check) is
a legitimate future phase, but a different problem with a different
design -- not built here.

Hybrid means two separate things layered together, on purpose:
  - Hybrid CHUNKING: split scraped HTML by its structure first (headings,
    paragraphs), and only fall back to a recursive character-based split
    for any single structural chunk that's still too large to embed
    sensibly. This keeps a chunk's meaning intact instead of cutting a
    sentence in half at an arbitrary character count.
  - Hybrid RETRIEVAL: rank chunks by BOTH dense embedding similarity
    (catches semantically related wording) AND sparse BM25 keyword
    scoring (catches exact terms, like a ticker symbol, that a paraphrase
    might drift away from), merged with Reciprocal Rank Fusion rather than
    a raw weighted sum of two differently-scaled numbers.
"""

import math
import re
from collections import Counter
from typing import Callable, Dict, List, Optional

import numpy as np
from bs4 import BeautifulSoup
from pydantic import BaseModel

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
BLOCK_TAGS = {"p", "li", "blockquote"}


class Chunk(BaseModel):
    text: str
    source_url: str
    heading_context: Optional[str] = None


def chunk_html(html: str, source_url: str, max_chunk_chars: int = 800) -> List[Chunk]:
    """Structural split by HTML tags first, recursive character split as fallback.

    Each chunk is tagged with the nearest preceding heading, so a paragraph
    under "Q3 Earnings" doesn't lose that context once it's separated from
    its section. Raises ValueError on empty input rather than silently
    returning an empty chunk list, since a caller quietly getting zero
    chunks from a real page is a bug worth surfacing, not swallowing.
    """
    if not html or not html.strip():
        raise ValueError("html must be non-empty.")

    soup = BeautifulSoup(html, "lxml")
    current_heading = None
    chunks: List[Chunk] = []

    for tag in soup.find_all(list(HEADING_TAGS | BLOCK_TAGS)):
        text = tag.get_text(separator=" ", strip=True)
        if not text:
            continue
        if tag.name in HEADING_TAGS:
            current_heading = text
            continue
        for piece in _recursive_split(text, max_chunk_chars):
            chunks.append(Chunk(text=piece, source_url=source_url, heading_context=current_heading))

    return chunks


def _recursive_split(text: str, max_chars: int) -> List[str]:
    """Falls back to smaller boundaries only when a structural chunk is too big.

    Tries paragraph breaks, then sentence breaks, then a hard character
    cut, in that order -- each one only invoked if the previous boundary
    still leaves a piece over max_chars.
    """
    if len(text) <= max_chars:
        return [text]

    for separator in ["\n\n", ". "]:
        parts = [p.strip() for p in text.split(separator) if p.strip()]
        if len(parts) > 1:
            result: List[str] = []
            for part in parts:
                result.extend(_recursive_split(part, max_chars))
            return _merge_small_pieces(result, max_chars)

    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _merge_small_pieces(pieces: List[str], max_chars: int) -> List[str]:
    """Recombines undersized fragments so we don't end up with a chunk that's one clause long."""
    merged: List[str] = []
    buffer = ""
    for piece in pieces:
        candidate = f"{buffer} {piece}".strip() if buffer else piece
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            if buffer:
                merged.append(buffer)
            buffer = piece
    if buffer:
        merged.append(buffer)
    return merged


_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def bm25_rank(chunks: List[Chunk], query: str, k1: float = 1.5, b: float = 0.75) -> List[int]:
    """Okapi BM25 over the in-memory chunk set. Returns chunk indices, best match first.

    Hand-rolled rather than a dependency -- the corpus here is a handful of
    freshly-scraped chunks per query, not a large static index, so a
    from-scratch implementation is both simpler to audit and plenty fast.
    """
    if not chunks:
        return []

    query_terms = _tokenize(query)
    doc_tokens = [_tokenize(c.text) for c in chunks]
    doc_lengths = [len(toks) for toks in doc_tokens]
    avg_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    n_docs = len(chunks)

    df: Counter = Counter()
    for toks in doc_tokens:
        df.update(set(toks))

    scores = [0.0] * n_docs
    for term in query_terms:
        n_qi = df.get(term, 0)
        if n_qi == 0:
            continue
        idf = math.log((n_docs - n_qi + 0.5) / (n_qi + 0.5) + 1.0)
        for i, toks in enumerate(doc_tokens):
            freq = toks.count(term)
            if freq == 0:
                continue
            denom = freq + k1 * (1 - b + b * doc_lengths[i] / avg_len) if avg_len else freq
            scores[i] += idf * (freq * (k1 + 1)) / denom

    return sorted(range(n_docs), key=lambda i: scores[i], reverse=True)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def embedding_rank(chunk_embeddings: List[List[float]], query_embedding: List[float]) -> List[int]:
    """Ranks chunk indices by cosine similarity to the query embedding, best-first."""
    if not chunk_embeddings:
        return []
    sims = [cosine_similarity(emb, query_embedding) for emb in chunk_embeddings]
    return sorted(range(len(chunk_embeddings)), key=lambda i: sims[i], reverse=True)


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = 60) -> List[int]:
    """Merges multiple ranked-index lists into one via Reciprocal Rank Fusion.

    RRF over raw score blending on purpose: BM25 scores and cosine
    similarities live on incompatible scales, so averaging them directly
    would let whichever metric happens to produce bigger numbers dominate.
    RRF only uses each item's RANK POSITION in each list, which sidesteps
    that entirely -- the standard technique real hybrid-search systems use
    to combine sparse and dense retrieval.
    """
    fused_scores: Dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused_scores.keys(), key=lambda idx: fused_scores[idx], reverse=True)


def hybrid_rank_chunks(
    chunks: List[Chunk],
    query: str,
    embed_fn: Callable[[str], List[float]],
    top_k: int = 5,
) -> List[Chunk]:
    """The full pipeline: BM25 + embedding similarity, fused, top_k returned.

    embed_fn is injected rather than hardcoded to a specific embedding
    model or client -- keeps this function testable with a fake,
    deterministic embedding function, and swappable for whichever local
    Ollama embedding model is actually configured at call time.
    """
    if not chunks:
        return []
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}.")

    bm25_ranking = bm25_rank(chunks, query)
    chunk_embeddings = [embed_fn(c.text) for c in chunks]
    query_embedding = embed_fn(query)
    embedding_ranking = embedding_rank(chunk_embeddings, query_embedding)

    fused = reciprocal_rank_fusion([bm25_ranking, embedding_ranking])
    return [chunks[i] for i in fused[:top_k]]
