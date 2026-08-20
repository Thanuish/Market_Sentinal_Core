from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import StateGraph, END

from src.agent.state import MarketSentinelState
from src.agent.nodes import (
    security_check,
    technical_analysis,
    fundamental_rag,
    sanitize_output,
    position_sizing,
    human_approval,
    executor_stub,
    log_and_stop,
)


def route_after_security(state: MarketSentinelState) -> str:
    if state.status == "REJECTED":
        return END
    return "watchdog"

def route_after_analysis(state: MarketSentinelState) -> str:
    if state.status == "ERROR":
        return END
    return "rag_evaluator"

def route_after_sizing(state: MarketSentinelState) -> str:
    if state.status == "ERROR":
        return END
    return "human_approval"

def route_after_approval(state: MarketSentinelState) -> str:
    if state.approval_decision == "approved":
        return "executor"
    return "log_and_stop"

workflow = StateGraph(MarketSentinelState)

workflow.add_node("security", security_check)
workflow.add_node("watchdog", technical_analysis)
workflow.add_node("rag_evaluator", fundamental_rag)
workflow.add_node("xss_sanitizer", sanitize_output)
workflow.add_node("position_sizing", position_sizing)
workflow.add_node("human_approval", human_approval)
workflow.add_node("executor", executor_stub)
workflow.add_node("log_and_stop", log_and_stop)

workflow.set_entry_point("security")

workflow.add_conditional_edges(
    "security", route_after_security, {"watchdog": "watchdog", END: END}
)
workflow.add_conditional_edges(
    "watchdog", route_after_analysis, {"rag_evaluator": "rag_evaluator", END: END}
)

workflow.add_edge("rag_evaluator", "xss_sanitizer")
workflow.add_edge("xss_sanitizer", "position_sizing")
workflow.add_conditional_edges(
    "position_sizing", route_after_sizing, {"human_approval": "human_approval", END: END}
)
workflow.add_conditional_edges(
    "human_approval", route_after_approval, {"executor": "executor", "log_and_stop": "log_and_stop"}
)
workflow.add_edge("executor", END)
workflow.add_edge("log_and_stop", END)

market_agent = workflow.compile(
    checkpointer=MemorySaver(
        serde=JsonPlusSerializer(
            allowed_msgpack_modules=[
                ("src.tools.watchdog", "TechnicalSignals"),
                ("src.tools.monte_carlo", "RiskEstimate"),
                ("src.tools.sizing", "PositionSizeResult"),
            ]
        )
    )
)
