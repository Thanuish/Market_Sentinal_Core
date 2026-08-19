from typing import Optional
from pydantic import BaseModel, Field


class TechnicalSignals(BaseModel):
    """Deterministic mathematical output from Watchdog router."""

    ticker: str
    current_price: float
    moving_average_cross: bool
    volatility_index: float


class MarketSentinelState(BaseModel):
    """Central state contract flowing through the LangGraph workflow."""

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
    final_allocation_pct: Optional[float] = None
