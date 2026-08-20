import html
from typing import Dict, Any
from src.agent.state import MarketSentinelState
from src.security.guardrails import GuardrailGateway
from src.tools.watchdog import run_watchdog
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from scrapling.fetchers import StealthyFetcher
from langgraph.types import interrupt
from src.tools.monte_carlo import estimate_risk_monte_carlo
from src.tools.sizing import compute_position_size, PositionSizeResult
from src.tools.retrieval import chunk_html, hybrid_rank_chunks
gatekeeper = GuardrailGateway()

_embeddings = OllamaEmbeddings(model="nomic-embed-text")


def _embed_text(text: str) -> list[float]:
    return _embeddings.embed_query(text)


@tool
def fetch_financial_news(urls: list[str], query: str = "") -> str:
    """Fetches real-time text from multiple financial news URLs, then chunks
    and ranks it with hybrid retrieval so only the most relevant material
    reaches the model -- not just the first 1500 characters of raw markup.

    Ephemeral by design -- chunks and rankings exist only for this single
    call and are never persisted, since financial news relevance decays
    within hours (see DECISIONS.md). `query` lets the calling model steer
    what "relevant" means for this specific evaluation; if omitted, a
    generic market-moving-news query is used instead.
    """
    print(f"\n    [Scrapling Tool] Executing production fetch for {len(urls)} URLs...")
    all_chunks = []

    for url in urls:
        try:
            page = StealthyFetcher.fetch(url)
            html_body = page.html_content
            chunks = chunk_html(html_body, source_url=url) if html_body else []
            all_chunks.extend(chunks)
            print(f"    [Scrapling Tool] Extracted {len(chunks)} chunks from {url}")
        except Exception as e:
            print(f"    [Scrapling Tool] Failed to fetch {url}: {str(e)}")

    if not all_chunks:
        return "No readable article text found from any source."

    search_query = query or "financial news, earnings, and market-moving events"

    # Fail-safe, not fail-open: if hybrid ranking breaks for any reason (the
    # embedding model isn't pulled, Ollama isn't reachable, etc.), fall back
    # to the raw chunks in scrape order rather than crashing the whole
    # evaluator node over a ranking failure.
    try:
        top_chunks = hybrid_rank_chunks(all_chunks, query=search_query, embed_fn=_embed_text, top_k=8)
    except Exception as e:
        print(f"    [Retrieval] Hybrid ranking failed, falling back to unranked chunks: {str(e)}")
        top_chunks = all_chunks[:8]

    formatted = []
    for c in top_chunks:
        heading = f" ({c.heading_context})" if c.heading_context else ""
        formatted.append(f"--- SOURCE: {c.source_url}{heading} ---\n{c.text}\n")

    return "\n".join(formatted)


def security_check(state: MarketSentinelState) -> Dict[str, Any]:
    """Node 1: Input Guardrails & PII Masking"""
    report = gatekeeper.verify_input(state.user_query)
    if not report.is_safe:
        return {'status': 'REJECTED', 'rejection_reason': report.reason}
    return {'status': 'SAFE', 'anonymized_query': report.anonymized_prompt}

def technical_analysis(state: MarketSentinelState) -> Dict[str, Any]:
    """Node 2: Deterministic Watchdog Math"""
    if not state.ticker:
        return {'status': 'ERROR', 'rejection_reason': 'No ticker provided.'}
    try:
        signals = run_watchdog(state.ticker, state.asset_type)
        signals_dict = signals.model_dump() if hasattr(signals, "model_dump") else signals.dict()

        return {'technical_signals': signals, 'status': 'ANALYZED'}
    except Exception as e:
        return {'status': 'ERROR', 'rejection_reason': f'Watchdog Error: {str(e)}'}

def fundamental_rag(state: MarketSentinelState) -> Dict[str, Any]:
    """Node 3: LangChain RAG Evaluator (Stochastic Context Overlay)"""
    sig = state.technical_signals

    # 1. Initialize the Local Ollama LLM
    llm = ChatOllama(model="llama3.1", temperature=0)
    llm_with_tools = llm.bind_tools([fetch_financial_news])

    # 2. Advanced Quantitative System Prompt
    quant_philosophy = (
        "You are an elite quantitative financial RAG agent. "
        "The deterministic modeling approach assumes all input variables (like prices and moving averages) "
        "are known with certainty, relying on fixed relationships like Golden Crosses and Bollinger bands. "
        "However, you understand this ignores complex dynamics, randomness, and uncertainty in real markets. "
        "Your job is to act as the stochastic, probabilistic overlay. "
        "Use the fetch_financial_news tool to retrieve live market reality, "
        "and evaluate the strict deterministic signals against this uncertainty to provide a balanced assessment."
    )

    # 3. Pass the full rich telemetry from the unified Watchdog
    messages = [
        SystemMessage(content=quant_philosophy),
        HumanMessage(
            content=f"Deterministic Watchdog Report for {sig.ticker}:\n"
            f"- Current Price: {sig.current_price}\n"
            f"- 50-day SMA: {sig.sma_50} | 200-day SMA: {sig.sma_200}\n"
            f"- Macro Trend (Golden Cross): {sig.moving_average_cross}\n"
            f"- Bollinger Volatility State: {sig.volatility_state}\n"
            f"- 30-Day Annualized Volatility: {sig.volatility_index}\n\n"
            f"Fetch live news for {sig.ticker} using the fetch_financial_news tool, "
            f"then provide a probabilistic evaluation balancing this strict math against market uncertainty."
        )
    ]

    # 4. First LLM Invoke (Requesting Tools)
    response = llm_with_tools.invoke(messages)

    # 5. The Agentic Loop: Execute Live Tool & Feed it Back
    if response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            if tool_call["name"] == "fetch_financial_news":
                tool_output = fetch_financial_news.invoke(tool_call["args"])
                messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))

        final_response = llm_with_tools.invoke(messages)
        reasoning = final_response.content
    else:
        reasoning = response.content

    return {
        "evaluation_reasoning": reasoning,
        "recommended_action": "BUY" if sig.moving_average_cross else "HOLD",
        "confidence_score": 0.85,
        "status": "EVALUATED"
    }

def sanitize_output(state: MarketSentinelState) -> Dict[str, Any]:
    """Node 4: Output Guardrail (XSS Defense)"""
    if state.evaluation_reasoning:
        clean_reasoning = html.escape(state.evaluation_reasoning)
        return {'evaluation_reasoning': clean_reasoning, 'status': 'COMPLETED'}
    return {'status': 'COMPLETED'}
def position_sizing(state: MarketSentinelState) -> Dict[str, Any]:
    """Node: deterministic risk estimation + Kelly sizing.

    Calls both quant_tools functions directly -- never through the LLM's
    tool-calling loop -- so a real position size exists regardless of what the
    evaluator's model did or didn't decide to call during its own reasoning.
    """
    sig = state.technical_signals

    if state.recommended_action != "BUY":
        return {
            "position_size": PositionSizeResult(
                position_pct=0.0, position_dollars=0.0, full_kelly_pct=0.0, capped=False
            ),
            "status": "SIZED",
        }

    try:
        risk = estimate_risk_monte_carlo(sig.price_history)
        position = compute_position_size(
            win_probability=risk.win_probability,
            expected_return=risk.expected_return,
            variance=risk.variance,
            bankroll=state.bankroll,
        )
        return {"risk_estimate": risk, "position_size": position, "status": "SIZED"}
    except ValueError as e:
        return {"status": "ERROR", "rejection_reason": f"Sizing Error: {str(e)}"}


def human_approval(state: MarketSentinelState) -> Dict[str, Any]:
    """Node: pause and wait for a real human decision.

    interrupt() suspends execution here -- nothing after this node runs until
    the graph is resumed with Command(resume=...) on the same thread_id. The
    checkpointer persists state across that pause, including across a process
    restart, because it's written to storage, not held in memory on the stack.
    """
    decision = interrupt(
        {
            "ticker": state.ticker,
            "recommended_action": state.recommended_action,
            "evaluation_reasoning": state.evaluation_reasoning,
            "position_size": state.position_size.model_dump() if state.position_size else None,
        }
    )
    return {"approval_decision": decision}


def executor_stub(state: MarketSentinelState) -> Dict[str, Any]:
    """Stub -- Phase 7 replaces this with a real paper-trade write to the SQLite ledger."""
    pct = state.position_size.position_pct if state.position_size else 0.0
    dollars = state.position_size.position_dollars if state.position_size else 0.0
    print(
        f"[EXECUTOR-STUB] Paper trade recorded: {state.ticker} {state.recommended_action} "
        f"{pct:.2%} of bankroll (${dollars:,.2f})"
    )
    return {"status": "EXECUTED"}


def log_and_stop(state: MarketSentinelState) -> Dict[str, Any]:
    print(f"[LOG] Trade not executed. approval_decision={state.approval_decision!r}")
    return {"status": "STOPPED"}
