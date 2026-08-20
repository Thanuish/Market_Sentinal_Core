from typing import Optional

from pydantic import BaseModel

from src.tools.monte_carlo import RiskEstimate
from src.tools.sizing import PositionSizeResult
from src.tools.watchdog import TechnicalSignals


class MarketSentinelState(BaseModel):
    """Central state contract flowing through the LangGraph workflow.

    technical_signals is typed against the SAME TechnicalSignals class that
    run_watchdog() actually returns (imported from src.tools.watchdog, not
    redefined here) -- two separate classes with the same name used to exist,
    one here and one in watchdog.py, and LangGraph's Pydantic-schema validation
    raised a hard ValidationError the moment the watchdog node's output reached
    the next node, because it isn't duck-typed: a differently-defined class
    with the same field names is still a different class to Pydantic.
    """

    user_query: str
    anonymized_query: Optional[str] = None
    asset_type: str = "stock"
    ticker: Optional[str] = None
    status: str = "PENDING"
    rejection_reason: Optional[str] = None
    technical_signals: Optional[TechnicalSignals] = None
    evaluation_reasoning: Optional[str] = None
    confidence_score: Optional[float] = None
    recommended_action: Optional[str] = None

    # Position sizing
    bankroll: float = 100_000.0  # fixed paper-trading bankroll for now; Phase 7's ledger will replace this with a real running balance
    risk_estimate: Optional[RiskEstimate] = None
    position_size: Optional[PositionSizeResult] = None

    # Human approval
    approval_decision: Optional[str] = None  # "approved" | "rejected" | anything else routes to log_and_stop
    final_allocation_pct: Optional[float] = None