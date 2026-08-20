"""Phase 8: cost per successful outcome, not raw accuracy.

Deliberately divides total cost across EVERY query the graph processed --
including ones a human rejected, ones that errored out in sizing, and ones
still pending an outcome -- by the count of trades later judged SUCCESS.
This punishes both an expensive system and an inaccurate one, and can't be
gamed by a strategy that just recommends HOLD constantly: that would be
cheap, but it would also never produce a SUCCESS to divide by, so the ratio
stays undefined rather than looking artificially good.
"""

import sqlite3
from typing import NamedTuple, Optional


class CostPerOutcomeReport(NamedTuple):
    total_queries: int
    total_successes: int
    total_failures: int
    pending_outcomes: int
    total_llm_tokens: int
    total_latency_seconds: float
    cost_per_success_tokens: Optional[float]
    cost_per_success_seconds: Optional[float]


def compute_cost_per_successful_outcome(conn: sqlite3.Connection) -> CostPerOutcomeReport:
    """Aggregates the whole ledger into the one number the CTO asked about.

    Returns None for the two cost-per-success fields (rather than raising or
    dividing by zero) when there are no SUCCESS rows yet -- an undefined
    ratio is the honest answer early on, not a misleading 0 or an infinity.
    """
    total_queries = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    total_successes = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE outcome_label = 'SUCCESS'"
    ).fetchone()[0]
    total_failures = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE outcome_label = 'FAILURE'"
    ).fetchone()[0]
    pending_outcomes = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE status = 'EXECUTED' AND outcome_label IS NULL"
    ).fetchone()[0]
    total_tokens = conn.execute(
        "SELECT COALESCE(SUM(llm_tokens_used), 0) FROM trades"
    ).fetchone()[0]
    total_seconds = conn.execute(
        "SELECT COALESCE(SUM(latency_seconds), 0) FROM trades"
    ).fetchone()[0]

    cost_per_success_tokens = (total_tokens / total_successes) if total_successes > 0 else None
    cost_per_success_seconds = (total_seconds / total_successes) if total_successes > 0 else None

    return CostPerOutcomeReport(
        total_queries=total_queries,
        total_successes=total_successes,
        total_failures=total_failures,
        pending_outcomes=pending_outcomes,
        total_llm_tokens=total_tokens,
        total_latency_seconds=total_seconds,
        cost_per_success_tokens=cost_per_success_tokens,
        cost_per_success_seconds=cost_per_success_seconds,
    )
