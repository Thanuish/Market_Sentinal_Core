"""Phase 7: the paper-trading ledger.

Every decision the graph reaches gets ONE row here -- not just executed
trades. This is deliberate: cost_per_successful_outcome (src/database/metrics.py)
needs the cost of every query that went through the pipeline, including the
ones a human rejected or that errored out in sizing, divided by how many of
the executed trades actually turned out to be successful. A ledger that only
logged executed trades couldn't compute that ratio honestly.

Two horizons get recorded for the same trade on purpose, and they answer
different questions:
  - risk_5d is the Monte Carlo estimate position_sizing actually used to size
    the trade (src/tools/sizing.py) -- it answers "how much should we risk."
  - expected_return_1d is a SEPARATE Monte Carlo estimate, run at the
    horizon the trade is actually judged at (see DECISIONS.md for why
    1 trading day -- fast feedback, at the cost of a noisier signal than
    a longer window would give). It answers "what did the model predict
    for the window we're about to check it against." Reusing the 5-day
    sizing estimate to judge a 1-day outcome would be comparing two
    different questions.
"""

import sqlite3
from typing import List, Optional

from src.tools.monte_carlo import RiskEstimate
from src.tools.sizing import PositionSizeResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    confidence_score REAL,
    entry_price REAL NOT NULL,

    position_pct REAL NOT NULL,
    position_dollars REAL NOT NULL,
    full_kelly_pct REAL NOT NULL,
    capped INTEGER NOT NULL,

    win_probability_5d REAL,
    expected_return_5d REAL,
    variance_5d REAL,
    expected_return_1d REAL,

    approval_decision TEXT,
    status TEXT NOT NULL,

    llm_tokens_used INTEGER,
    latency_seconds REAL,
    created_at TEXT NOT NULL,

    outcome_checked_at TEXT,
    exit_price REAL,
    realized_return REAL,
    outcome_label TEXT
);
"""


def init_db(conn: sqlite3.Connection) -> None:
    """Creates the trades table if it doesn't exist yet. Safe to call every startup."""
    conn.execute(SCHEMA)
    conn.commit()


def record_trade(
    conn: sqlite3.Connection,
    *,
    thread_id: str,
    ticker: str,
    asset_type: str,
    recommended_action: str,
    confidence_score: Optional[float],
    entry_price: float,
    position: PositionSizeResult,
    risk_5d: Optional[RiskEstimate],
    expected_return_1d: Optional[float],
    approval_decision: Optional[str],
    status: str,
    llm_tokens_used: Optional[int],
    latency_seconds: Optional[float],
    created_at: str,
) -> int:
    """Writes one row for one graph decision -- executed, rejected, or errored.

    Returns the new row's id, so callers that need to reference this trade
    later (e.g. to record its outcome once the holding horizon has passed)
    don't have to re-query for it.
    """
    if entry_price <= 0:
        raise ValueError(f"entry_price must be positive, got {entry_price}.")
    if llm_tokens_used is not None and llm_tokens_used < 0:
        raise ValueError(f"llm_tokens_used cannot be negative, got {llm_tokens_used}.")
    if latency_seconds is not None and latency_seconds < 0:
        raise ValueError(f"latency_seconds cannot be negative, got {latency_seconds}.")

    cursor = conn.execute(
        """
        INSERT INTO trades (
            thread_id, ticker, asset_type, recommended_action, confidence_score,
            entry_price, position_pct, position_dollars, full_kelly_pct, capped,
            win_probability_5d, expected_return_5d, variance_5d, expected_return_1d,
            approval_decision, status, llm_tokens_used, latency_seconds, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            thread_id, ticker, asset_type, recommended_action, confidence_score,
            entry_price, position.position_pct, position.position_dollars,
            position.full_kelly_pct, int(position.capped),
            risk_5d.win_probability if risk_5d else None,
            risk_5d.expected_return if risk_5d else None,
            risk_5d.variance if risk_5d else None,
            expected_return_1d,
            approval_decision, status, llm_tokens_used, latency_seconds, created_at,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_pending_outcomes(conn: sqlite3.Connection, as_of: str) -> List[sqlite3.Row]:
    """Rows that executed, are old enough to judge, and haven't been judged yet.

    as_of is an ISO timestamp string, caller-supplied rather than computed
    here with datetime.now() -- keeps this function deterministic and
    testable against any fixed "current time" a test wants to simulate.
    """
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT * FROM trades
        WHERE status = 'EXECUTED'
          AND outcome_label IS NULL
          AND created_at <= ?
        """,
        (as_of,),
    )
    return cursor.fetchall()


def record_outcome(
    conn: sqlite3.Connection,
    trade_id: int,
    exit_price: float,
    checked_at: str,
) -> str:
    """Judges one executed trade against ITS OWN 1-day Monte Carlo prediction.

    A trade is a SUCCESS if its realized return beat what the model
    predicted for this exact horizon at trade time (expected_return_1d) --
    not an arbitrary fixed bar, and not the 5-day sizing estimate, which
    was answering a different question. Returns the outcome label so the
    caller can log or report it without a second query.
    """
    if exit_price <= 0:
        raise ValueError(f"exit_price must be positive, got {exit_price}.")

    row = conn.execute(
        "SELECT entry_price, expected_return_1d FROM trades WHERE id = ?", (trade_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"No trade with id {trade_id}.")
    entry_price, expected_return_1d = row

    realized_return = (exit_price - entry_price) / entry_price
    bar = expected_return_1d if expected_return_1d is not None else 0.0
    outcome_label = "SUCCESS" if realized_return > bar else "FAILURE"

    conn.execute(
        """
        UPDATE trades
        SET exit_price = ?, realized_return = ?, outcome_label = ?, outcome_checked_at = ?
        WHERE id = ?
        """,
        (exit_price, realized_return, outcome_label, checked_at, trade_id),
    )
    conn.commit()
    return outcome_label
