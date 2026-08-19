from langgraph.graph import StateGraph, END
from src.agent.state import MarketSentinelState
from src.agent.nodes import (
    security_check, 
    technical_analysis, 
    fundamental_rag, 
    sanitize_output
)

def route_after_security(state: MarketSentinelState) -> str:
    if state.status == "REJECTED":
        return END
    return "watchdog"
    
def route_after_analysis(state: MarketSentinelState) -> str:
    if state.status == "ERROR":
        return END
    return "rag_evaluator"

# 1. Initialize Graph with Pydantic State Schema
workflow = StateGraph(MarketSentinelState)

# 2. Add Nodes
workflow.add_node("security", security_check)
workflow.add_node("watchdog", technical_analysis)
workflow.add_node("rag_evaluator", fundamental_rag)
workflow.add_node("xss_sanitizer", sanitize_output)

# 3. Define Control Flow Edges
workflow.set_entry_point("security")

workflow.add_conditional_edges("security", route_after_security)
workflow.add_conditional_edges("watchdog", route_after_analysis)

workflow.add_edge("rag_evaluator", "xss_sanitizer")
workflow.add_edge("xss_sanitizer", END)

# 4. Compile the Machine
market_agent = workflow.compile()
