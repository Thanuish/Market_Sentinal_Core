import html
from typing import Dict, Any
from src.agent.state import MarketSentinelState
from src.security.guardrails import GuardrailGateway
from src.tools.watchdog import run_watchdog
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from scrapling.fetchers import StealthyFetcher

gatekeeper = GuardrailGateway()

@tool
def fetch_financial_news(urls: list[str]) -> str:
    """Fetches real-time text from multiple financial news URLs concurrently."""
    print(f"\n    [Scrapling Tool] Executing production fetch for {len(urls)} URLs...")
    scraped_results = []

    for url in urls:
        try:
            page = StealthyFetcher.fetch(url)
            paragraphs = page.css("p::text").getall()
            clean_text = " ".join(paragraphs)[:1500] if paragraphs else "No readable article text found."
            scraped_results.append(f"--- SOURCE: {url} ---\n{clean_text}\n")
            print(f"    [Scrapling Tool] Successfully extracted data from {url}")
        except Exception as e:
            print(f"    [Scrapling Tool] Failed to fetch {url}: {str(e)}")
            scraped_results.append(f"--- SOURCE: {url} ---\nError fetching data: {str(e)}\n")

    return "\n".join(scraped_results)


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