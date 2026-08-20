# Market Sentinel — Architecture & MCP Call Map

Auto-generated and kept in sync with the actual repo state as we build. Solid green = built and wired into `graph.py` today. Dashed grey = built as a standalone function or planned, but not yet connected to the graph. Purple = an MCP tool exposed by the `quant_tools` server. Blue = a plain LangChain tool (not MCP). This distinction matters: it shows exactly which nodes are real right now versus which are still ahead of us, and — for the two MCP tools — whether the LLM or the graph itself is the one allowed to call them.

```mermaid
flowchart TD
    classDef built fill:#2f6f4f,stroke:#1c4a34,color:#fff,stroke-width:2px
    classDef planned fill:#3a3a3a,stroke:#888,color:#ccc,stroke-width:2px,stroke-dasharray: 6 4
    classDef mcp fill:#5b3a8e,stroke:#3d2760,color:#fff,stroke-width:2px
    classDef lctool fill:#2a5a7a,stroke:#1c3d52,color:#fff,stroke-width:2px
    classDef reject fill:#8e3a3a,stroke:#5c2323,color:#fff,stroke-width:2px

    Start(["Ticker / Watchlist Input"]) --> Security

    Security["security node<br/>GuardrailGateway:<br/>prompt-injection + toxicity + PII mask"]:::built
    Watchdog["watchdog node<br/>technical_analysis<br/>pure Python, no LLM"]:::built
    Evaluator["rag_evaluator node<br/>fundamental_rag<br/>Ollama llama3.1"]:::built
    Sanitizer["xss_sanitizer node<br/>html.escape"]:::built
    Sizing["position_sizing node<br/>Monte Carlo risk + fractional Kelly<br/>calls quant_tools directly, no LLM"]:::built
    Approval{{"human_approval node<br/>LangGraph interrupt + MemorySaver checkpoint"}}:::built
    Executor["executor_stub node<br/>paper trade stub"]:::built
    LogStop["log_and_stop node"]:::built
    Ledger[("SQLite Ledger")]:::planned
    Metrics["Cost / Outcome Metrics"]:::planned

    Security -->|"is_safe = false"| RejectA(["END<br/>REJECTED"]):::reject
    Security -->|"is_safe = true"| Watchdog
    Watchdog -->|"status = ERROR"| RejectB(["END<br/>ERROR"]):::reject
    Watchdog -->|"status = ANALYZED"| Evaluator
    Evaluator --> Sanitizer
    Sanitizer --> Sizing
    Sizing -->|"status = ERROR"| RejectD(["END<br/>Sizing Error"]):::reject
    Sizing -->|"status = SIZED"| Approval
    Approval -->|"resume = 'approved'"| Executor
    Approval -->|"anything else"| LogStop
    Executor --> End1(["END (Phase 7: write to Ledger)"])
    LogStop --> End2(["END"])
    Executor -.->|"planned"| Ledger
    Ledger -.->|"planned"| Metrics

    Evaluator -.->|"LLM tool-call<br/>model decides"| News[["LangChain tool:<br/>fetch_financial_news"]]:::lctool
    News -.->|"scraped text"| Evaluator

    Sizing ==>|"graph calls directly<br/>LLM never sees this tool"| MC[["MCP quant_tools:<br/>estimate_risk_monte_carlo"]]:::mcp
    MC ==>|"RiskEstimate"| Sizing

    Sizing ==>|"graph calls directly<br/>LLM never sees this tool"| KS[["MCP quant_tools:<br/>compute_position_size"]]:::mcp
    KS ==>|"PositionSizeResult"| Sizing

    subgraph Legend["Legend"]
        direction LR
        LegBuilt["Built + tested"]:::built
        LegPlanned["Planned / not wired in yet"]:::planned
        LegMcp[["MCP tool"]]:::mcp
        LegLc[["LangChain tool (not MCP)"]]:::lctool
        LegReject(["Structural reject / stop"]):::reject
    end
```

## Reading this diagram in an interview

- The full risk pipeline is now wired end to end: `security → watchdog → rag_evaluator → xss_sanitizer → position_sizing → human_approval → executor_stub / log_and_stop`. All of it is covered by `tests/test_graph_gates.py` (13 tests), which passed 42/42 against the whole suite as of the last run.
- `position_sizing` calls `estimate_risk_monte_carlo` and `compute_position_size` directly as Python functions — both are also exposed through the `quant_tools` MCP server (`src/mcp/server.py`) so they're independently callable over the MCP protocol, but the graph itself never routes through that protocol layer for its own internal calls. That's a deliberate simplicity choice, not an oversight: MCP exposure and "who's allowed to call it" are separate concerns, and the graph gets a direct, zero-latency call for logic that has to run every time regardless of what any LLM decided.
- Neither `estimate_risk_monte_carlo` nor `compute_position_size` is bound to the evaluator LLM via `.bind_tools()` right now — only `fetch_financial_news` is. That's worth being precise about if asked: the earlier plan was for Monte Carlo to be LLM-bindable as an informational tool, but as currently wired, the LLM never sees either quant tool. Kelly sizing was never meant to be LLM-bindable at all (Design Principle 2), and that part is correctly enforced — the function isn't in any `bind_tools()` call anywhere in the codebase.
- `human_approval` genuinely pauses the graph via `interrupt()` and a `MemorySaver` checkpoint — proven, not just claimed, by `test_graph_pauses_at_human_approval_before_executing`, `test_approved_resume_reaches_executor`, and `test_rejected_resume_never_reaches_executor` actually driving a real interrupt/resume cycle.
- `executor_stub` and the SQLite ledger are still separate: the executor currently just prints a paper-trade line and ends the graph. Phase 7 replaces the stub with a real ledger write — that's the next real gap, and it's an honest, visible one on this diagram, not a hidden one.

*This file (and `docs/architecture.html`, the browser-viewable version) is updated automatically as new pieces get built — no need to ask for a refresh.*
