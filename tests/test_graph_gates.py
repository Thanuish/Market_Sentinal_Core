"""Phase 5: prove the structural gate, don't just claim it.

Three independent angles, so one kind of mistake can't slip through all of them:
  1. Structural: inspect the COMPILED graph's real edge topology. If someone
     rewires position_sizing straight to executor, or deletes the human_approval
     node, this fails -- regardless of whether anyone remembers to update a test
     for it, because it's reading the graph's actual wiring, not a copy of it.
  2. Routing logic: the route_after_approval() function in isolation. If someone
     changes `== "approved"` to a truthy check (a classic bug: "rejected" is a
     non-empty string too), this fails immediately.
  3. End-to-end: the real position_sizing/human_approval/executor_stub/log_and_stop
     node implementations, run through an actual interrupt + checkpoint + resume
     cycle. If someone removes the interrupt() call itself, this fails.

(1) imports the real, fully-wired src.agent.graph -- this only needs the same
import chain your own "graph compiled OK" check already exercised (no Ollama
call, no live network fetch). (2) and (3) import the real node/routing functions
directly and drive them with hand-built state, so they run in well under a
second with no external services required at all.
"""

from typing import Any, Dict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from src.agent.graph import market_agent, route_after_approval
from src.agent.nodes import executor_stub, human_approval, log_and_stop, position_sizing
from src.agent.state import MarketSentinelState
from src.tools.watchdog import TechnicalSignals


# ---------------------------------------------------------------------------
# 1. Structural proof: executor is reachable ONLY from human_approval.
# ---------------------------------------------------------------------------

def test_executor_has_no_incoming_edge_except_from_human_approval():
    edges = market_agent.get_graph().edges
    incoming_to_executor = {e.source for e in edges if e.target == "executor"}
    assert incoming_to_executor == {"human_approval"}, (
        f"executor should only be reachable from human_approval, "
        f"but found incoming edges from: {incoming_to_executor}"
    )


def test_human_approval_can_reach_both_executor_and_log_and_stop():
    # Confirms the gate actually branches -- not a pass-through that always
    # reaches executor regardless of the decision.
    edges = market_agent.get_graph().edges
    outgoing = {e.target for e in edges if e.source == "human_approval"}
    assert outgoing == {"executor", "log_and_stop"}


# ---------------------------------------------------------------------------
# 2. Routing logic in isolation -- catches a truthy-check bug immediately.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "decision,expected_route",
    [
        ("approved", "executor"),
        ("rejected", "log_and_stop"),
        ("", "log_and_stop"),
        (None, "log_and_stop"),
        ("Approved", "log_and_stop"),      # case must match exactly
        ("approved ", "log_and_stop"),     # no silent whitespace tolerance
        ("true", "log_and_stop"),          # not a truthy check in disguise
        ("yes", "log_and_stop"),
    ],
)
def test_route_after_approval_only_executes_on_exact_string_approved(decision, expected_route):
    state = MarketSentinelState(user_query="q", ticker="TEST", approval_decision=decision)
    assert route_after_approval(state) == expected_route


# ---------------------------------------------------------------------------
# 3. End-to-end: real nodes, real interrupt, real checkpointer, real resume.
# ---------------------------------------------------------------------------

def _fake_watchdog_node(state: MarketSentinelState) -> Dict[str, Any]:
    prices = [100.0 + i * 0.5 for i in range(60)]  # clean uptrend, deterministic
    sig = TechnicalSignals(
        ticker=state.ticker, current_price=prices[-1], sma_50=105.0, sma_200=98.0,
        moving_average_cross=True, upper_band=115.0, lower_band=95.0,
        volatility_state="NORMAL", volatility_index=0.18, price_history=prices,
    )
    return {"technical_signals": sig, "status": "ANALYZED"}


def _fake_evaluator_node(state: MarketSentinelState) -> Dict[str, Any]:
    return {"evaluation_reasoning": "test", "recommended_action": "BUY",
            "confidence_score": 0.85, "status": "EVALUATED"}


def _route_after_sizing(state: MarketSentinelState) -> str:
    return END if state.status == "ERROR" else "human_approval"


def _build_test_graph():
    workflow = StateGraph(MarketSentinelState)
    workflow.add_node("watchdog", _fake_watchdog_node)
    workflow.add_node("evaluator", _fake_evaluator_node)
    workflow.add_node("position_sizing", position_sizing)
    workflow.add_node("human_approval", human_approval)
    workflow.add_node("executor", executor_stub)
    workflow.add_node("log_and_stop", log_and_stop)
    workflow.set_entry_point("watchdog")
    workflow.add_edge("watchdog", "evaluator")
    workflow.add_edge("evaluator", "position_sizing")
    workflow.add_conditional_edges(
        "position_sizing", _route_after_sizing, {"human_approval": "human_approval", END: END}
    )
    workflow.add_conditional_edges(
        "human_approval", route_after_approval, {"executor": "executor", "log_and_stop": "log_and_stop"}
    )
    workflow.add_edge("executor", END)
    workflow.add_edge("log_and_stop", END)
    return workflow.compile(
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


def test_graph_pauses_at_human_approval_before_executing():
    graph = _build_test_graph()
    config = {"configurable": {"thread_id": "pause-test"}}
    result = graph.invoke(MarketSentinelState(user_query="q", ticker="TEST"), config=config)
    assert "__interrupt__" in result, "graph should have paused for human approval, not run to completion"
    assert result["status"] != "EXECUTED", "nothing should be executed before a human has decided"


def test_approved_resume_reaches_executor():
    graph = _build_test_graph()
    config = {"configurable": {"thread_id": "approved-test"}}
    graph.invoke(MarketSentinelState(user_query="q", ticker="TEST"), config=config)
    final = graph.invoke(Command(resume="approved"), config=config)
    assert final["status"] == "EXECUTED"


def test_rejected_resume_never_reaches_executor():
    graph = _build_test_graph()
    config = {"configurable": {"thread_id": "rejected-test"}}
    graph.invoke(MarketSentinelState(user_query="q", ticker="TEST"), config=config)
    final = graph.invoke(Command(resume="rejected"), config=config)
    assert final["status"] == "STOPPED"
    assert final["status"] != "EXECUTED"