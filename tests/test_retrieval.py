import pytest

from src.tools.retrieval import (
    Chunk,
    bm25_rank,
    chunk_html,
    cosine_similarity,
    embedding_rank,
    hybrid_rank_chunks,
    reciprocal_rank_fusion,
)


class TestChunkHtml:
    def test_splits_by_heading_and_paragraph_structure(self):
        html = """
        <html><body>
        <h2>Q3 Earnings</h2>
        <p>Revenue grew 12 percent year over year.</p>
        <h2>Guidance</h2>
        <p>The company raised full year guidance.</p>
        </body></html>
        """
        chunks = chunk_html(html, source_url="http://example.com/a")
        assert len(chunks) == 2
        assert chunks[0].heading_context == "Q3 Earnings"
        assert "Revenue grew" in chunks[0].text
        assert chunks[1].heading_context == "Guidance"
        assert "raised full year guidance" in chunks[1].text

    def test_paragraph_with_no_heading_gets_none_context(self):
        html = "<html><body><p>Just a paragraph, no heading above it.</p></body></html>"
        chunks = chunk_html(html, source_url="http://example.com/a")
        assert chunks[0].heading_context is None

    def test_recursive_fallback_splits_oversized_paragraph(self):
        long_sentence_block = ". ".join([f"Sentence number {i} about the market" for i in range(40)])
        html = f"<html><body><p>{long_sentence_block}.</p></body></html>"
        chunks = chunk_html(html, source_url="http://example.com/a", max_chunk_chars=200)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text) <= 220  # small slack for merge boundary, never wildly over

    def test_hard_cut_when_no_natural_boundary_exists(self):
        no_boundaries = "x" * 500  # single unbroken token, no periods or paragraph breaks
        html = f"<html><body><p>{no_boundaries}</p></body></html>"
        chunks = chunk_html(html, source_url="http://example.com/a", max_chunk_chars=100)
        assert len(chunks) == 5
        for c in chunks:
            assert len(c.text) <= 100

    def test_all_chunks_carry_the_source_url(self):
        html = "<html><body><p>one</p><p>two</p></body></html>"
        chunks = chunk_html(html, source_url="http://example.com/xyz")
        assert all(c.source_url == "http://example.com/xyz" for c in chunks)

    @pytest.mark.parametrize("bad_html", ["", "   ", None])
    def test_empty_or_none_html_raises(self, bad_html):
        with pytest.raises(ValueError, match="non-empty"):
            chunk_html(bad_html, source_url="http://example.com/a")


class TestBm25Rank:
    def _chunks(self):
        return [
            Chunk(text="The Federal Reserve raised interest rates today.", source_url="u1"),
            Chunk(text="A recipe for chocolate chip cookies and baking tips.", source_url="u2"),
            Chunk(text="Interest rates and Federal Reserve policy explained.", source_url="u3"),
        ]

    def test_relevant_chunks_outrank_irrelevant_ones(self):
        ranking = bm25_rank(self._chunks(), query="Federal Reserve interest rates")
        assert ranking[0] in (0, 2)
        assert ranking[-1] == 1  # the cookie recipe should rank last

    def test_empty_chunks_returns_empty_list(self):
        assert bm25_rank([], query="anything") == []

    def test_query_with_no_matching_terms_still_returns_all_indices(self):
        ranking = bm25_rank(self._chunks(), query="zzz nonexistent term")
        assert set(ranking) == {0, 1, 2}


class TestCosineSimilarity:
    def test_identical_vectors_are_one(self):
        assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_are_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_are_negative_one(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero_not_nan_or_crash(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestEmbeddingRank:
    def test_closest_vector_ranked_first(self):
        query = [1.0, 0.0]
        embeddings = [[0.0, 1.0], [0.99, 0.01], [-1.0, 0.0]]
        ranking = embedding_rank(embeddings, query)
        assert ranking[0] == 1
        assert ranking[-1] == 2

    def test_empty_embeddings_returns_empty_list(self):
        assert embedding_rank([], [1.0, 0.0]) == []


class TestReciprocalRankFusion:
    def test_single_ranking_is_unchanged(self):
        assert reciprocal_rank_fusion([[2, 0, 1]]) == [2, 0, 1]

    def test_agreement_between_rankings_wins(self):
        # index 0 is #1 in both lists -- should fuse to #1.
        bm25 = [0, 1, 2]
        embedding = [0, 2, 1]
        fused = reciprocal_rank_fusion([bm25, embedding])
        assert fused[0] == 0

    def test_consistently_decent_beats_a_spike_then_crash(self):
        # index 0: #1 in bm25, but DEAD LAST (of 5) in embedding -- a spike.
        # index 1: #2 in both lists -- consistently decent, never great.
        # RRF's harmonic weighting means being #1 anywhere still counts for
        # a lot, so this doesn't claim index 1 tops the fused ranking
        # outright -- only that it isn't dragged down the way a genuine
        # crash-to-last-place would drag index 0, which is the actual
        # point of fusing on rank position rather than raw score.
        bm25 = [0, 1, 2, 3, 4]
        embedding = [2, 1, 3, 4, 0]
        fused = reciprocal_rank_fusion([bm25, embedding])
        assert fused.index(1) < fused.index(0)


class TestHybridRankChunks:
    def _fake_embed(self, text: str):
        # Deterministic fake: embedding dimension 2, encodes presence of
        # "rate" vs "recipe" as axes, so ranking is fully predictable
        # without any real model or network call.
        lowered = text.lower()
        return [1.0 if "rate" in lowered else 0.0, 1.0 if "recipe" in lowered else 0.0]

    def test_returns_top_k_relevant_chunks(self):
        chunks = [
            Chunk(text="Interest rate hikes continue this quarter.", source_url="u1"),
            Chunk(text="A recipe for banana bread.", source_url="u2"),
            Chunk(text="The central bank raised the rate again.", source_url="u3"),
        ]
        result = hybrid_rank_chunks(chunks, query="interest rate", embed_fn=self._fake_embed, top_k=2)
        assert len(result) == 2
        result_texts = {c.text for c in result}
        assert "A recipe for banana bread." not in result_texts

    def test_empty_chunks_returns_empty_list_without_calling_embed_fn(self):
        calls = []

        def tracking_embed(text):
            calls.append(text)
            return [0.0]

        result = hybrid_rank_chunks([], query="anything", embed_fn=tracking_embed, top_k=3)
        assert result == []
        assert calls == []

    def test_top_k_less_than_one_raises(self):
        chunks = [Chunk(text="something", source_url="u1")]
        with pytest.raises(ValueError, match="top_k"):
            hybrid_rank_chunks(chunks, query="q", embed_fn=self._fake_embed, top_k=0)
