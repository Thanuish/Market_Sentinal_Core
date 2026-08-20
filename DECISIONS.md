# DECISIONS.md

This file exists for one reason: every non-obvious choice in this codebase should be defensible, not just functional. If a decision below isn't written down, treat it as unintentional and flag it.

Each entry follows the same shape: **what we chose**, **what we didn't choose and why not**, **what it costs us**. The "cost" section is deliberate — a decision with no downside listed hasn't been thought through honestly.

---

## 1. The LLM never computes a number that matters

**Decision:** Every number that determines money at risk — technical indicators (SMA, Bollinger Bands, volatility), the Monte Carlo win probability / expected return / variance, and the Kelly position size — is computed by deterministic Python (`watchdog.py`, `monte_carlo.py`, `sizing.py`). The LLM (Llama 3.1 via Ollama) only ever sees these numbers as *input text* and reasons over them in natural language.

**Alternative considered:** Let the LLM read price history and estimate volatility/position size directly, or have it call a "calculator" tool and trust its arithmetic.

**Why not:** LLMs are not reliable calculators, and more importantly, a hallucinated or subtly-wrong number here is invisible in the output — it looks exactly like a correct one. A wrong Bollinger Band value doesn't throw an error, it just quietly sizes a position wrong. Keeping money-math in code means the failure mode for a bug is a crash or a wrong-but-inspectable number, not silent LLM confabulation.

**Cost:** The LLM's job shrinks to "explain and contextualize," which is a less impressive-sounding use of an LLM than "the AI decides the trade." That's the point, not a bug — but it means this isn't a system that showcases LLM reasoning; it showcases LLM *restraint*.

---

## 2. Guardrails live in graph topology, not prompts

**Decision:** Safety-critical control flow (reject on unsafe input, stop before execution without approval, sanitize before output) is implemented as `add_conditional_edges` with explicit `path_map` in `graph.py` — actual branches in the state machine — not as instructions in a system prompt asking the model to "please refuse unsafe requests."

**Alternative considered:** A single LLM call with a long system prompt covering injection resistance, toxicity, PII handling, and "don't execute trades without approval."

**Why not:** A prompt is a suggestion the model can be talked out of; a graph edge is not. If `security_check` returns `REJECTED`, there is no path in the graph from that state to `executor_stub` — not "the model was told not to go there," but "the edge doesn't exist." The interrupt before execution is the same principle: it's not the LLM choosing to pause, it's `human_approval` being a real blocking node with no way around it.

**Cost:** Every new safety requirement means a new node or edge, not a prompt tweak — slower to iterate on than "just reword the system prompt," and it means the graph shape itself has to be understood to reason about safety, not just the prompts.

**Explicit gap (not yet fixed):** `GuardrailGateway.verify_input()` in `guardrails.py` currently has no try/except around the injection-classifier and toxicity-model calls. If either model call errors or times out, the node crashes rather than resolving to a safe rejection. This is a fail-open gap and directly contradicts principle 2's own spirit — it's flagged, understood, and not yet fixed.

---

## 3. Human approval is a real LangGraph `interrupt()`, not a flag check

**Decision:** `human_approval` calls `interrupt(...)`, which suspends the graph run entirely. Nothing after that node executes until the graph is resumed with `Command(resume=...)` on the same `thread_id`. State is persisted by the checkpointer (`MemorySaver` in dev), so the pause survives even a process restart — it's not held on a stack frame in memory.

**Alternative considered:** A boolean `approved` field on the state that a node checks before proceeding, set by some external polling loop or webhook.

**Why not:** A flag check is advisory — anything that constructs the state dict wrong, or a bug that sets the flag prematurely, bypasses it. `interrupt()` is structural: the graph literally cannot advance past that point without an explicit resume call carrying a decision. It also means the pause is a first-class part of the execution model, not a workaround bolted onto it.

**Cost:** Requires a checkpointer and thread-based execution model, which adds infrastructure (even if it's just `MemorySaver` for now) that a simpler flag-check wouldn't need. In production this becomes a durable checkpointer (Postgres/SQLite-backed), which is a real operational dependency, not a toy one.

---

## 4. Cost per successful outcome, not raw accuracy

**Decision:** Every graph decision — not just executed trades — is logged to a SQLite ledger (`src/database/ledger.py`). "Success" is judged, per trade, by comparing the realized 1-day return against *that same trade's own* Monte Carlo-predicted return, computed at a 1-day horizon specifically for the check (`expected_return_1d`), not the 5-day estimate used for sizing and not an arbitrary fixed bar like "beat 2%." The headline metric (`compute_cost_per_successful_outcome` in `metrics.py`) is tokens-per-success and seconds-per-success.

**Alternatives considered and explicitly rejected:**
- *Raw win-rate / accuracy*: says nothing about what it cost to get there. A system that's right 80% of the time but burns 10x the compute per call isn't obviously better than one that's right 65% of the time cheaply — "accuracy" alone can't answer "is this worth running."
- *Fixed success threshold (e.g., "beat 2% return")*: arbitrary, and doesn't adapt to how confident the system itself was. A trade the model predicted would return 0.3% shouldn't need to hit 2% to count as a win; a trade it predicted would return 5% probably should.
- *20-day evaluation horizon* (the original plan): initially chosen, then deliberately shortened to 1 day. A 20-day horizon means weeks of latency before the metric has any signal at all during development and iteration — for a system meant to be evaluated and defended now, a 1-day horizon gives a metric that actually produces feedback while still building.

**The anti-gaming detail:** `cost_per_success_tokens`/`cost_per_success_seconds` return `None`, not `0`, when `total_successes == 0` — deliberately. A strategy that always recommends HOLD would have zero cost and (trivially) zero failures, and a naive ratio would show `0` cost-per-success, which reads as "free and perfect." Returning `None` makes "no successes yet" visibly different from "amazing performance," so an always-HOLD strategy can't accidentally look optimal.

**Cost:** This metric is more honest but less flattering than accuracy — it can't be gamed by cherry-picking easy calls, but it also means a genuinely good system with high compute cost per call will show a worse number than a cheap-but-mediocre one, which requires explaining rather than just reporting.

---

## 5. Everything runs locally via Ollama — no paid API

**Decision:** `ChatOllama(model="llama3.1")` for reasoning, `OllamaEmbeddings(model="nomic-embed-text")` for retrieval embeddings. No OpenAI/Anthropic/any hosted API calls anywhere in the pipeline.

**Why:** This is a constraint, not a preference — it forces every design decision above to hold up without the crutch of "just use a bigger/cheaper hosted model." It also means principle 4 (cost per successful outcome) is measured in *tokens and latency*, real local resource cost, rather than dollars billed to a vendor — a more honest cost signal for a system meant to run unattended and repeatedly.

**Cost:** Llama 3.1 8B locally is meaningfully weaker than frontier hosted models, especially at instruction-following under tool-calling loops. Some of the defensive design elsewhere in this project (structured payloads over NLU, deterministic math never trusted to the LLM, fail-safe fallbacks around every LLM-adjacent call) exists *because* the model is not fully trustworthy — that's a feature of the constraint, not a coincidence.

---

## 6. If it's not tested, it doesn't exist

**Decision:** Every module that isn't a thin orchestration wrapper has direct unit tests — `watchdog.py`, `sizing.py`, `monte_carlo.py`, `retrieval.py`, `ledger.py`, `metrics.py`, `guardrails.py` — using fake/stub packages where a live dependency (Ollama, network) would make tests non-deterministic or slow, rather than skipping coverage for those paths.

**Why:** A design principle that isn't verified is a claim, not a fact. "Guardrails live in topology" is only true if a test actually asserts the rejected-input path never reaches the executor node; "cost-per-success resists gaming" is only true if a test actually asserts the always-HOLD case returns `None`.

**Cost:** Stub packages (fake `langchain_ollama`, fake `scrapling`, etc.) mean the test suite verifies *logic*, not integration with the real Ollama server or real scraping — a green test suite doesn't guarantee the live system works end-to-end, only that each piece behaves correctly in isolation. Live smoke tests (like `scripts/try_watchdog.py`) exist separately to cover that gap, but they're manual, not part of CI.

---

## Supporting technical decisions (not full principles, but asked-about in review)

**Fractional Kelly, not full Kelly.** `KELLY_FRACTION = 0.5` and `MAX_POSITION_PCT = 0.05` are module-level constants in `sizing.py`, not function parameters. Full Kelly is provably volatility-maximizing in the long run but produces position sizes most practitioners consider too aggressive given real-world estimation error in win probability and expected return — half-Kelly is a standard, well-understood haircut for exactly that estimation uncertainty. The hard 5% cap exists as a second, independent backstop in case the Kelly fraction itself is ever misconfigured or the inputs are unusually extreme — defense in depth, not redundancy for its own sake.

**MCP tool exposure is orthogonal to LLM tool-calling.** The Monte Carlo estimator and Kelly sizer are exposed as MCP tools (via `mcp.server.fastmcp.FastMCP`) so they're inspectable/callable through the MCP protocol, but they are never `.bind_tools()`-attached to the LLM. Only `fetch_financial_news` is LLM-callable. This is principle 1 enforced structurally: exposing a tool via MCP is about interoperability and inspection, not about handing the LLM the ability to decide when risk math runs.

**Structured payload input, not natural-language understanding, for the trade pipeline.** The graph expects `{"ticker": "GLD", "asset_type": "commodity"}`-shaped input, not a chat message the LLM has to parse intent from. This was a deliberate choice to avoid paying an LLM call before every deterministic step just to extract a ticker symbol — agents don't need small talk, they need data. A conversational front-end could sit in front of this and translate "what's gold doing" into a structured payload, but that translation layer is explicitly out of scope for the trading graph itself.

**Hybrid chunking: structural HTML split, recursive fallback.** `chunk_html()` first splits by semantic HTML structure (headings/paragraphs via BeautifulSoup) so chunks respect the article's own organization, then recursively character-splits any chunk that's still oversized. Structure-first because heading context is a real retrieval signal (a paragraph under "Risks" section header is different from one under "Q3 Earnings"); recursive fallback exists because real-world HTML is inconsistent and pure structural splitting alone produces occasional oversized chunks.

**Hybrid retrieval: BM25 + dense embeddings, fused with Reciprocal Rank Fusion.** `hybrid_rank_chunks()` computes a hand-rolled Okapi BM25 ranking and a dense cosine-similarity ranking (via `nomic-embed-text`), then fuses the two *rank orders* with RRF rather than blending the two *raw scores*. BM25 scores and cosine similarities live on incompatible scales (unbounded term-frequency scores vs. `[-1, 1]` cosine) — averaging them directly would let whichever score happens to have the larger numeric range dominate for reasons that have nothing to do with relevance. RRF sidesteps this by only ever looking at rank position, not score magnitude.

**Ephemeral retrieval scope for news; persistent scope deliberately deferred.** Chunks and rankings are rebuilt per-query and never persisted, because financial news relevance decays within hours — an index built this morning is actively misleading by afternoon. Persistent vector storage (e.g., for SEC 10-K filings, which are stable for months) is a real future phase, explicitly deferred rather than an oversight — the two use cases have genuinely different half-lives and shouldn't share one retrieval strategy by default.

**Fail-safe, not fail-open, around the retrieval ranking call.** `fetch_financial_news` wraps `hybrid_rank_chunks()` in a try/except that falls back to unranked chunks (scrape order) if ranking fails for any reason — embedding model not pulled, Ollama unreachable, etc. — rather than letting the exception propagate and crash the evaluator node. This mirrors the same "fail-safe, not fail-open" principle that governs guardrail design (see principle 2's open gap above, which is the same principle *not yet* applied consistently everywhere).

---

## Known gaps (stated plainly, not hidden)

- `security_check` has no fail-safe wrapper around its classifier calls — a model error currently crashes the node instead of defaulting to a safe rejection. Understood, not yet fixed.
- `executor_stub` is still a stub — it prints the trade instead of calling `ledger.record_trade()`. The ledger and metrics modules are fully built and tested in isolation but not yet wired into the live graph execution path. This is the single largest remaining integration gap.
- `position_sizing` currently only calls Monte Carlo at the 5-day horizon used for sizing; a second call at `horizon_days=1` to populate `expected_return_1d` for the outcome-check metric is not yet wired in for real (non-test) trades.
- No live broker/paper-trading API is connected. The Kelly sizing and Monte Carlo risk logic are fully implemented and tested, but there is no real account to observe filled trades against yet — that's future work, not a hidden limitation.
