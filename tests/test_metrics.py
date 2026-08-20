import sqlite3

import pytest

from src.database.ledger import init_db, record_trade
from src.database.metrics import compute_cost_per_successful_outcome
from src.tools.sizing import PositionSizeResult


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    init_db(c)
    yield c
    c.close()


def _position():
    return PositionSizeResult(position_pct=0.02, position_dollars=2000.0, full_kelly_pct=0.04, capped=False)


def _insert(conn, status, tokens, seconds, outcome_label=None):
    trade_id = record_trade(
        conn, thread_id="t", ticker="AAPL", asset_type="stock",
        recommended_action="BUY", confidence_score=0.85, entry_price=100.0,
        position=_position(), risk_5d=None, expected_return_1d=0.02,
        approval_decision="approved" if status == "EXECUTED" else "rejected",
        status=status, llm_tokens_used=tokens, latency_seconds=seconds,
        created_at="2026-01-01T00:00:00",
    )
    if outcome_label:
        conn.execute("UPDATE trades SET outcome_label = ? WHERE id = ?", (outcome_label, trade_id))
        conn.commit()
    return trade_id


class TestComputeCostPerSuccessfulOutcome:
    def test_empty_ledger_returns_zeros_and_none_costs(self, conn):
        report = compute_cost_per_successful_outcome(conn)
        assert report.total_queries == 0
        assert report.total_successes == 0
        assert report.cost_per_success_tokens is None
        assert report.cost_per_success_seconds is None

    def test_no_successes_yet_returns_none_not_zero_or_crash(self, conn):
        _insert(conn, "EXECUTED", tokens=500, seconds=3.0)  # pending, no outcome yet
        _insert(conn, "STOPPED", tokens=200, seconds=1.0)
        report = compute_cost_per_successful_outcome(conn)
        assert report.total_successes == 0
        assert report.cost_per_success_tokens is None
        assert report.cost_per_success_seconds is None
        assert report.total_llm_tokens == 700  # cost still counted even with no successes

    def test_cost_divides_total_cost_across_all_queries_by_successes_only(self, conn):
        # Every query's cost counts, but only SUCCESS rows count in the denominator.
        _insert(conn, "EXECUTED", tokens=400, seconds=2.0, outcome_label="SUCCESS")
        _insert(conn, "EXECUTED", tokens=600, seconds=3.0, outcome_label="SUCCESS")
        _insert(conn, "EXECUTED", tokens=500, seconds=2.5, outcome_label="FAILURE")
        _insert(conn, "STOPPED", tokens=100, seconds=0.5)  # rejected by human, still cost something
        _insert(conn, "SIZED", tokens=0, seconds=0.2)      # never reached approval at all

        report = compute_cost_per_successful_outcome(conn)
        assert report.total_queries == 5
        assert report.total_successes == 2
        assert report.total_failures == 1
        assert report.total_llm_tokens == 400 + 600 + 500 + 100 + 0
        assert report.cost_per_success_tokens == pytest.approx((400 + 600 + 500 + 100 + 0) / 2)
        assert report.cost_per_success_seconds == pytest.approx((2.0 + 3.0 + 2.5 + 0.5 + 0.2) / 2)

    def test_pending_outcomes_are_counted_separately_from_successes_and_failures(self, conn):
        _insert(conn, "EXECUTED", tokens=300, seconds=1.0)  # no outcome_label -> pending
        _insert(conn, "EXECUTED", tokens=300, seconds=1.0, outcome_label="SUCCESS")
        report = compute_cost_per_successful_outcome(conn)
        assert report.pending_outcomes == 1
        assert report.total_successes == 1

    def test_a_strategy_that_always_recommends_hold_never_looks_artificially_good(self, conn):
        # No trades ever executed -> no successes possible -> ratio stays
        # undefined, not a misleadingly cheap-looking 0.
        for _ in range(10):
            _insert(conn, "SIZED", tokens=50, seconds=0.3)
        report = compute_cost_per_successful_outcome(conn)
        assert report.total_successes == 0
        assert report.cost_per_success_tokens is None
