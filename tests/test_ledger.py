import sqlite3

import pytest

from src.database.ledger import get_pending_outcomes, init_db, record_outcome, record_trade
from src.tools.monte_carlo import RiskEstimate
from src.tools.sizing import PositionSizeResult


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    init_db(c)
    yield c
    c.close()


def _position(pct=0.025, dollars=2500.0, full_kelly=0.05, capped=False):
    return PositionSizeResult(
        position_pct=pct, position_dollars=dollars, full_kelly_pct=full_kelly, capped=capped
    )


def _risk(win_prob=0.6, exp_return=0.01, variance=0.02):
    return RiskEstimate(
        win_probability=win_prob, expected_return=exp_return, variance=variance,
        num_simulations=10_000, horizon_days=5,
    )


class TestInitDb:
    def test_is_idempotent(self, conn):
        init_db(conn)  # fixture already called it once; calling again must not raise
        init_db(conn)


class TestRecordTrade:
    def test_returns_incrementing_row_ids(self, conn):
        id1 = record_trade(
            conn, thread_id="t1", ticker="AAPL", asset_type="stock",
            recommended_action="BUY", confidence_score=0.85, entry_price=150.0,
            position=_position(), risk_5d=_risk(), expected_return_1d=0.03,
            approval_decision="approved", status="EXECUTED",
            llm_tokens_used=400, latency_seconds=2.1, created_at="2026-01-01T00:00:00",
        )
        id2 = record_trade(
            conn, thread_id="t2", ticker="MSFT", asset_type="stock",
            recommended_action="BUY", confidence_score=0.85, entry_price=300.0,
            position=_position(), risk_5d=_risk(), expected_return_1d=0.02,
            approval_decision="approved", status="EXECUTED",
            llm_tokens_used=380, latency_seconds=1.9, created_at="2026-01-01T00:05:00",
        )
        assert id1 == 1
        assert id2 == 2

    def test_stored_row_matches_inputs(self, conn):
        trade_id = record_trade(
            conn, thread_id="t1", ticker="AAPL", asset_type="stock",
            recommended_action="BUY", confidence_score=0.85, entry_price=150.0,
            position=_position(pct=0.04, dollars=4000.0, full_kelly=0.09, capped=True),
            risk_5d=_risk(win_prob=0.62, exp_return=0.015, variance=0.03),
            expected_return_1d=0.05, approval_decision="approved", status="EXECUTED",
            llm_tokens_used=512, latency_seconds=3.4, created_at="2026-01-01T00:00:00",
        )
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        assert row["ticker"] == "AAPL"
        assert row["position_pct"] == pytest.approx(0.04)
        assert row["full_kelly_pct"] == pytest.approx(0.09)
        assert row["capped"] == 1
        assert row["win_probability_5d"] == pytest.approx(0.62)
        assert row["expected_return_1d"] == pytest.approx(0.05)
        assert row["status"] == "EXECUTED"

    def test_allows_null_risk_for_non_buy_trades(self, conn):
        # position_sizing returns risk_5d=None style zero position when the
        # evaluator didn't recommend BUY -- the ledger must accept that.
        trade_id = record_trade(
            conn, thread_id="t3", ticker="TSLA", asset_type="stock",
            recommended_action="HOLD", confidence_score=0.85, entry_price=220.0,
            position=_position(pct=0.0, dollars=0.0, full_kelly=0.0, capped=False),
            risk_5d=None, expected_return_1d=None,
            approval_decision=None, status="SIZED",
            llm_tokens_used=None, latency_seconds=None, created_at="2026-01-01T00:00:00",
        )
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        assert row["win_probability_5d"] is None
        assert row["expected_return_1d"] is None
        assert row["llm_tokens_used"] is None

    @pytest.mark.parametrize("bad_price", [0.0, -10.0])
    def test_non_positive_entry_price_raises(self, conn, bad_price):
        with pytest.raises(ValueError, match="entry_price"):
            record_trade(
                conn, thread_id="t1", ticker="AAPL", asset_type="stock",
                recommended_action="BUY", confidence_score=0.85, entry_price=bad_price,
                position=_position(), risk_5d=_risk(), expected_return_1d=0.03,
                approval_decision="approved", status="EXECUTED",
                llm_tokens_used=400, latency_seconds=2.1, created_at="2026-01-01T00:00:00",
            )

    def test_negative_llm_tokens_raises(self, conn):
        with pytest.raises(ValueError, match="llm_tokens_used"):
            record_trade(
                conn, thread_id="t1", ticker="AAPL", asset_type="stock",
                recommended_action="BUY", confidence_score=0.85, entry_price=150.0,
                position=_position(), risk_5d=_risk(), expected_return_1d=0.03,
                approval_decision="approved", status="EXECUTED",
                llm_tokens_used=-5, latency_seconds=2.1, created_at="2026-01-01T00:00:00",
            )

    def test_negative_latency_raises(self, conn):
        with pytest.raises(ValueError, match="latency_seconds"):
            record_trade(
                conn, thread_id="t1", ticker="AAPL", asset_type="stock",
                recommended_action="BUY", confidence_score=0.85, entry_price=150.0,
                position=_position(), risk_5d=_risk(), expected_return_1d=0.03,
                approval_decision="approved", status="EXECUTED",
                llm_tokens_used=400, latency_seconds=-1.0, created_at="2026-01-01T00:00:00",
            )


class TestGetPendingOutcomes:
    def _insert(self, conn, status, created_at, outcome_label=None):
        trade_id = record_trade(
            conn, thread_id="t", ticker="AAPL", asset_type="stock",
            recommended_action="BUY", confidence_score=0.85, entry_price=150.0,
            position=_position(), risk_5d=_risk(), expected_return_1d=0.03,
            approval_decision="approved", status=status,
            llm_tokens_used=400, latency_seconds=2.1, created_at=created_at,
        )
        if outcome_label:
            conn.execute(
                "UPDATE trades SET outcome_label = ? WHERE id = ?", (outcome_label, trade_id)
            )
            conn.commit()
        return trade_id

    def test_only_returns_executed_unjudged_rows_old_enough_to_check(self, conn):
        old_executed = self._insert(conn, "EXECUTED", "2026-01-01T00:00:00")
        self._insert(conn, "EXECUTED", "2026-02-15T00:00:00")  # too recent
        self._insert(conn, "STOPPED", "2026-01-01T00:00:00")  # never executed
        self._insert(conn, "EXECUTED", "2026-01-01T00:00:00", outcome_label="SUCCESS")  # already judged

        pending = get_pending_outcomes(conn, as_of="2026-01-29T00:00:00")
        pending_ids = {row["id"] for row in pending}
        assert pending_ids == {old_executed}


class TestRecordOutcome:
    def _executed_trade(self, conn, entry_price=100.0, expected_return_1d=0.02):
        return record_trade(
            conn, thread_id="t", ticker="AAPL", asset_type="stock",
            recommended_action="BUY", confidence_score=0.85, entry_price=entry_price,
            position=_position(), risk_5d=_risk(), expected_return_1d=expected_return_1d,
            approval_decision="approved", status="EXECUTED",
            llm_tokens_used=400, latency_seconds=2.1, created_at="2026-01-01T00:00:00",
        )

    def test_beats_prediction_is_success(self, conn):
        trade_id = self._executed_trade(conn, entry_price=100.0, expected_return_1d=0.02)
        label = record_outcome(conn, trade_id, exit_price=110.0, checked_at="2026-01-29T00:00:00")
        assert label == "SUCCESS"  # realized 10% > predicted 2%

    def test_below_prediction_is_failure(self, conn):
        trade_id = self._executed_trade(conn, entry_price=100.0, expected_return_1d=0.05)
        label = record_outcome(conn, trade_id, exit_price=101.0, checked_at="2026-01-29T00:00:00")
        assert label == "FAILURE"  # realized 1% < predicted 5%

    def test_positive_but_below_prediction_is_still_failure(self, conn):
        # Proves this isn't just "any positive return" -- it's judged against
        # what THIS trade specifically predicted, per the design decision.
        trade_id = self._executed_trade(conn, entry_price=100.0, expected_return_1d=0.08)
        label = record_outcome(conn, trade_id, exit_price=103.0, checked_at="2026-01-29T00:00:00")
        assert label == "FAILURE"

    def test_writes_realized_return_and_timestamps(self, conn):
        trade_id = self._executed_trade(conn, entry_price=200.0, expected_return_1d=0.01)
        record_outcome(conn, trade_id, exit_price=220.0, checked_at="2026-01-29T00:00:00")
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        assert row["realized_return"] == pytest.approx(0.10)
        assert row["exit_price"] == pytest.approx(220.0)
        assert row["outcome_checked_at"] == "2026-01-29T00:00:00"

    def test_unknown_trade_id_raises(self, conn):
        with pytest.raises(ValueError, match="No trade"):
            record_outcome(conn, 9999, exit_price=100.0, checked_at="2026-01-29T00:00:00")

    @pytest.mark.parametrize("bad_price", [0.0, -5.0])
    def test_non_positive_exit_price_raises(self, conn, bad_price):
        trade_id = self._executed_trade(conn)
        with pytest.raises(ValueError, match="exit_price"):
            record_outcome(conn, trade_id, exit_price=bad_price, checked_at="2026-01-29T00:00:00")
