# Market Sentinel

A local, multi-agent paper-trading research system built with LangGraph, Ollama, and MCP — designed around one governing question: **what does it cost to reach a successful outcome, not how often does the model sound right?**

This project exists to defend a specific set of engineering claims, not just to produce trade recommendations. It was rebuilt from the ground up around six non-negotiable design principles (below), each chosen because it's the kind of thing that either holds up under a hard follow-up question in an interview, or doesn't.
---

## Design principles (non-negotiable)

1. **The LLM never computes numbers that matter.** Position sizing and risk math are pure, unit-tested Python. An LLM can narrate a decision; it cannot decide how many dollars are at risk.
2. **Guardrails live in graph topology, not prompts.** Safety and control are enforced by which nodes an edge can and cannot reach — not by asking a model nicely to behave.
3. **A human approves before anything executes.** Every trade pauses at a real LangGraph `interrupt()`, backed by a checkpointer, until a human decision resumes it.
4. **Measure cost per successful outcome, not raw accuracy.** A system that's cheap and mediocre can beat one that's expensive and slightly better — this project is built to make that tradeoff visible, not to assume accuracy alone is the goal.
5. **Everything runs locally via Ollama.** No paid API, no external LLM dependency, no per-call billing.
6. **If it's not tested, it doesn't exist.** Every deterministic function has adversarial-input tests; every structural guarantee (like the approval gate) is proven against the compiled graph, not just described in a comment.

---

## Architecture

The system is a single LangGraph `StateGraph` — not a swarm of independent agents. Almost every node is a deterministic Python function with zero model calls; exactly one node (`rag_evaluator`) behaves like a classic ReAct agent, deciding whether to call a tool and synthesizing a result. Every other node exists specifically to constrain what that one node's unpredictability is allowed to touch.

```
Ticker / structured input
        │
        ▼
  security ──(unsafe)──▶ END [REJECTED]
        │ (safe)
        ▼
  watchdog ──(error)──▶ END [ERROR]
        │ (analyzed)
        ▼
  rag_evaluator
        │
        ▼
  xss_sanitizer
        │
        ▼
  position_sizing ──(error)──▶ END [Sizing Error]
        │ (sized)
        ▼
  human_approval  (interrupt() + checkpoint — execution PAUSES here)
        │
   ┌────┴─────┐
"approved"   anything else
   │            │
   ▼            ▼
executor_stub  log_and_stop
   │            │
   ▼            ▼
  END          END
```

The full color-coded diagram, with the MCP call map and the LLM-callable vs. graph-only-callable distinction, lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (Mermaid source, viewable on GitHub) and [`docs/architecture.html`](docs/architecture.html) (self-contained, opens directly in a browser). Both are kept in sync with the actual graph as it's built, not written once and left stale.

### Node-by-node

**`security` (`security_check`)** — Runs every query through `GuardrailGateway`: a prompt-injection classifier (`protectai/deberta-v3-base-prompt-injection-v2`), a toxicity model (`Detoxify`), and regex-based PII masking (email, phone, IBAN). Unsafe input never reaches any other node. This is a pure safety filter — it does not interpret intent or extract a ticker from free text (see **Structured input, not natural language**, below).

**`watchdog` (`technical_analysis`)** — Pure Python technical analysis, zero LLM involvement. Computes 50-day and 200-day simple moving averages (a golden cross is `sma_50 > sma_200`), 20-day Bollinger Bands (±2 standard deviations), and 30-day annualized volatility. Supports three asset classes through two data sources: `yfinance` for stocks and forex (forex tickers reformatted to the `EURUSD=X` convention), and Binance's public REST klines endpoint for crypto. Returns a typed `TechnicalSignals` object — including the full price history, which the Monte Carlo simulation downstream needs for bootstrapping.

**`rag_evaluator` (`fundamental_rag`)** — The one agentic node. An Ollama `llama3.1` model is bound to a single tool, `fetch_financial_news`, and decides whether to call it. If it does, the tool scrapes live pages via Scrapling (`StealthyFetcher`) and returns text for the model to reason over. **Honest limitation:** `recommended_action` is currently hardcoded (`"BUY" if golden_cross else "HOLD"`) and `confidence_score` is a flat `0.85` — the LLM's narrative reasoning does not yet influence the actual trade decision. This is a known, deliberate-to-flag gap, not an oversight discovered by someone else first.

**`xss_sanitizer` (`sanitize_output`)** — HTML-escapes the LLM's reasoning text before it goes anywhere else, since it may echo scraped web content.

**`position_sizing`** — Calls two pure functions directly, as plain Python — never through the LLM's tool-calling loop: `estimate_risk_monte_carlo` (historical bootstrap simulation of the ticker's own daily returns, seeded for reproducibility) produces a win probability, expected return, and variance; `compute_position_size` applies fractional Kelly (`KELLY_FRACTION = 0.5`, half-Kelly) to that estimate, then clips the result to a hard cap (`MAX_POSITION_PCT = 0.05`) as a second, independent guardrail. Both constants are module-level — not function parameters — so no caller, human or LLM, can override risk policy without editing and redeploying source.

**`human_approval`** — Calls LangGraph's `interrupt()`, which genuinely suspends the graph. Nothing after this node runs until a human resumes it with `Command(resume=...)` on the same `thread_id`. State is persisted by a `MemorySaver` checkpointer, so the pause survives independently of any single process's memory.

**`executor_stub` / `log_and_stop`** — On `resume="approved"`, `executor_stub` records the paper trade (currently prints; wiring to the SQLite ledger — `src/database/ledger.py`, which exists and is fully tested — is the next integration step, see **Status**, below). Anything other than the exact string `"approved"` routes to `log_and_stop` instead — proven by adversarial routing tests, not just described.

### Structured input, not natural language

There is deliberately no natural-language-understanding step that parses a free-text question ("what's the price of gold?") into a ticker. `security_check` screens for safety; it does not extract intent. The system's real interface is a structured payload — `ticker` and `asset_type`, supplied directly by the caller — not a conversational prompt. This was a considered choice, not an unfinished feature: adding an LLM parsing step ahead of every deterministic node would mean every query pays for a model call before any real work starts, which directly fights Principle 4 (cost per successful outcome). See `DECISIONS.md` for the full reasoning.

### MCP tools vs. LLM-callable tools

These are two independent properties, not the same thing:

- `fetch_financial_news` is a plain LangChain tool, bound to the evaluator's LLM via `.bind_tools()` — the model decides whether and what to fetch. Safe because it's read-only.
- `estimate_risk_monte_carlo` and `compute_position_size` are both served over MCP by `quant_tools` (`src/mcp/server.py`), but **neither is bound to any LLM anywhere in the codebase.** `position_sizing` calls the underlying pure functions directly. MCP exposure and "who's allowed to call this" are separate design axes — the money-affecting math is MCP-wrapped for external protocol access, but the graph itself never routes through that protocol layer, and the LLM never sees these tools at all.

### Hybrid retrieval

`fetch_financial_news` used to scrape raw page text and truncate it to 1,500 characters per source — grounded, but not retrieval in the technical sense. `src/tools/retrieval.py` now implements real hybrid retrieval, and `fetch_financial_news` calls it directly: structural HTML chunking (split by heading/paragraph boundaries, with a recursive character-based fallback for oversized chunks), then Reciprocal Rank Fusion combining BM25 keyword scoring with embedding cosine similarity (via a local `OllamaEmbeddings("nomic-embed-text")` call). It's deliberately ephemeral — rebuilt fresh per query, nothing persisted to disk, since financial news decays in relevance within hours and a growing index would add staleness risk for no benefit. (A persistent, metadata-filtered store is a reasonable future phase for something like 10-K filings, which are too expensive to re-parse on every check — not built here.) The ranking step is wrapped in its own fail-safe fallback: if the embedding call fails for any reason (Ollama down, model not pulled), the tool falls back to unranked chunks in scrape order rather than crashing the evaluator node. **Status: built, tested (23 tests), and wired into `fetch_financial_news`.** Requires `ollama pull nomic-embed-text` to actually run the dense-retrieval side; falls back gracefully if that model isn't available.

### The ledger and cost-per-successful-outcome metric

`src/database/ledger.py` logs **every** graph decision to SQLite — not just executed trades, including ones a human rejected or that errored in sizing — because `src/database/metrics.py`'s `compute_cost_per_successful_outcome` needs the total cost of every query divided by how many resulted in a genuinely successful trade. A trade is judged 1 trading day after execution, against its **own** Monte Carlo-predicted return for that horizon (not an arbitrary fixed bar, and not the 5-day estimate used for sizing — a separate estimate, computed for the horizon it's actually judged at). This design specifically resists being gamed: a strategy that always recommends HOLD would be cheap, but would also never produce a successful outcome to divide by, so the ratio stays undefined rather than looking artificially good. **Status: both modules built and unit-tested (21 tests, in-memory SQLite, zero filesystem dependency); not yet wired into `executor_stub`, and no automated job yet calls `record_outcome` to close the loop later.** This is the single largest remaining integration gap in the project.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangGraph (`StateGraph`, `interrupt`, `MemorySaver`) | Explicit topology is what makes the guardrails provable, not just described |
| LLM | Ollama, `llama3.1` (local) | Principle 5 — no paid API, no per-call billing |
| Market data | `yfinance` (stocks, forex), Binance public REST (crypto) | Free, no API key required |
| Web scraping | Scrapling (`StealthyFetcher`) | Stealth fetch for live news grounding |
| Chunking / retrieval | `beautifulsoup4`, `lxml`, hand-rolled BM25, `numpy` cosine similarity | No new heavy dependencies — everything needed was already in the stack |
| Security | `detoxify`, `transformers` (prompt-injection classifier), regex PII masking | Local, no external moderation API |
| Tool protocol | MCP (`mcp.server.fastmcp.FastMCP`) | Standardized tool exposure, decoupled from who's allowed to call it |
| Persistence | SQLite (ledger), in-memory `MemorySaver` (graph checkpoints) | Zero-ops, embedded, no server to run |
| Testing | `pytest`, `unittest.mock` | Every external boundary (yfinance, Binance, Ollama, Scrapling) is mocked; nothing network-dependent runs in the test suite |

---

## Project structure

```
market_sentinel/
├── src/
│   ├── agent/
│   │   ├── state.py         # MarketSentinelState — the single state contract
│   │   ├── nodes.py         # All node functions
│   │   └── graph.py         # StateGraph wiring, routing functions, compiled market_agent
│   ├── tools/
│   │   ├── watchdog.py      # Deterministic technical analysis, 3 asset classes
│   │   ├── monte_carlo.py   # Historical bootstrap risk simulation
│   │   ├── sizing.py        # Fractional Kelly + hard cap
│   │   └── retrieval.py     # Hybrid chunking + hybrid retrieval (RRF)
│   ├── database/
│   │   ├── ledger.py        # SQLite trade ledger
│   │   └── metrics.py       # cost_per_successful_outcome
│   ├── mcp/
│   │   └── server.py        # quant_tools MCP server
│   └── security/
│       └── guardrails.py    # GuardrailGateway — injection, toxicity, PII
├── tests/                   # One test file per module above, mirrored 1:1
├── docs/
│   ├── ARCHITECTURE.md      # Auto-maintained Mermaid diagram + call map
│   └── architecture.html    # Browser-viewable version
├── DECISIONS.md             # Why, not just what — one entry per real design call
├── requirements.txt
└── pytest.ini
```

---

## Setup

Requires Python 3.11+, a local [Ollama](https://ollama.com) installation, and the `llama3.1` model pulled locally.

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd market_sentinel

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the local model
ollama pull llama3.1

# 5. (For hybrid retrieval, once wired in) pull a local embedding model
ollama pull nomic-embed-text
```

## Running the tests

```bash
pytest tests/ -v
```

86 tests across 7 files, mirroring `src/` 1:1. Every external dependency — `yfinance`, the Binance API, Ollama, Scrapling, `OllamaEmbeddings` — is mocked at the test boundary. The suite requires no network access and no running Ollama server to pass.

## Current status

| Piece | Status |
|---|---|
| Security guardrails (injection / toxicity / PII) | Built, tested |
| Deterministic technical analysis (3 asset classes) | Built, tested |
| LLM fundamental evaluation + live news tool-calling | Built, tested |
| Output sanitization | Built, tested |
| Monte Carlo risk estimation | Built, tested |
| Fractional Kelly position sizing (MCP-wrapped, graph-only-callable) | Built, tested |
| Human approval gate (`interrupt` + checkpoint) | Built, tested — proven structurally, at the routing level, and end-to-end |
| Full graph wiring, `security` → `executor_stub`/`log_and_stop` | Built, tested (structural edge-topology tests included) |
| Hybrid chunking + retrieval | Built, tested, wired into `fetch_financial_news` |
| SQLite trade ledger | Built, unit-tested (16 tests) — not yet wired into `executor_stub` |
| Cost-per-successful-outcome metric | Built, unit-tested (5 tests) — not yet fed by live data |
| Automated outcome-checking job | Not yet built |
| `recommended_action` / `confidence_score` driven by LLM reasoning | Not yet built — currently hardcoded off the golden-cross signal |
| Broker/live execution | Explicitly out of scope — this is a paper-trading research system |

## Known limitations (stated plainly, not discovered by someone else first)

- The LLM's fundamental reasoning is currently narrative only — it does not influence `recommended_action` or `confidence_score`, both of which are deterministic today.
- There is no natural-language entrypoint; callers must supply a ticker and asset type directly. This is a deliberate design choice (see `DECISIONS.md`), not a missing feature.
- The 1-day outcome-evaluation horizon trades signal cleanliness for fast feedback — expect noisier success/failure labels than a longer window would produce.
- The SQLite ledger and cost-per-successful-outcome metric exist as tested modules but are not yet integrated into the live graph path — `executor_stub` still prints instead of writing a real row, and no scheduled job yet calls `record_outcome` to close the loop on a trade's later performance.


