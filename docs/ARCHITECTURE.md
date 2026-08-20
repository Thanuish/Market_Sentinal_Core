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
    Sizing["position_sizing node<br/>NOT YET WIRED INTO GRAPH"]:::planned
    Approval{{"Human Approval Gate<br/>LangGraph interrupt + checkpointer"}}:::planned
    Executor["executor node<br/>paper trade only"]:::planned
    Ledger[("SQLite Ledger")]:::planned
    Metrics["Cost / Outcome Metrics"]:::planned

    Security -->|"is_safe = false"| RejectA(["LOG + END<br/>REJECTED"]):::reject
    Security -->|"is_safe = true"| Watchdog
    Watchdog -->|"status = ERROR"| RejectB(["LOG + END<br/>ERROR"]):::reject
    Watchdog -->|"status = ANALYZED"| Evaluator
    Evaluator --> Sanitizer
    Sanitizer --> End0(["END (current graph stops here)"])

    Sanitizer -.->|"not yet connected"| Sizing
    Sizing -.->|"not yet connected"| Approval
    Approval -->|"approved"| Executor
    Approval -->|"rejected / edited"| RejectC(["LOG + STOP"]):::reject
    Executor --> Ledger
    Ledger --> Metrics
    Metrics --> End1(["END (planned)"])

    Evaluator -.->|"LLM tool-call<br/>model decides"| MC[["MCP quant_tools:<br/>estimate_risk_monte_carlo"]]:::mcp
    MC -.->|"RiskEstimate"| Evaluator

    Evaluator -.->|"LLM tool-call<br/>model decides"| News[["LangChain tool:<br/>fetch_financial_news"]]:::lctool
    News -.->|"scraped text"| Evaluator

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

- The graph as actually wired in `graph.py` today stops at `xss_sanitizer → END`. Everything from `position_sizing` onward is designed but not connected — that's an honest, visible gap, not a hidden one.
- `estimate_risk_monte_carlo` is bound to the evaluator's LLM via `.bind_tools()` — the model decides whether and how to call it. That's safe because it's read-only and informational.
- `compute_position_size` is never bound to the LLM. Only the graph's own deterministic code is meant to call it — this is the structural guardrail from Design Principle 2 in practice, not just a claim.
- `fetch_financial_news` is a plain LangChain tool, not MCP — worth being precise about the difference if asked, since only `quant_tools` is the actual MCP server this project exposes.

*This file (and `docs/architecture.html`, the browser-viewable version) is updated automatically as new pieces get built — no need to ask for a refresh.*
