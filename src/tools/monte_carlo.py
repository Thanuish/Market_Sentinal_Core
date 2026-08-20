import numpy as np
from pydantic import BaseModel, Field


class RiskEstimate(BaseModel):
    """Output of a historical-bootstrap Monte Carlo simulation. Purely descriptive —
    nothing in this model authorizes a trade or sizes a position."""

    win_probability: float = Field(..., description="Fraction of simulated paths with a positive return over the horizon.")
    expected_return: float = Field(..., description="Mean simulated return over the horizon.")
    variance: float = Field(..., description="Variance of simulated returns over the horizon.")
    num_simulations: int
    horizon_days: int


def estimate_risk_monte_carlo(
    prices: list[float],
    horizon_days: int = 5,
    num_simulations: int = 10_000,
    seed: int = 42,
) -> RiskEstimate:
    """Bootstrap the ticker's own historical daily returns to simulate `num_simulations`
    possible `horizon_days`-ahead paths, and summarize them as a win probability,
    expected return, and variance.

    Deliberately nonparametric: we resample real observed daily returns rather than
    fitting a normal/log-normal distribution, so we don't assume a shape the market
    hasn't actually shown us. Deliberately seeded: Monte Carlo is random by nature,
    but a fixed seed makes this function's output reproducible and testable.
    """
    if num_simulations < 1:
        raise ValueError("num_simulations must be at least 1.")
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1.")

    prices_arr = np.asarray(prices, dtype=float)
    if len(prices_arr) < 30:
        raise ValueError(
            f"Need at least 30 days of price history to bootstrap a reliable sample, got {len(prices_arr)}."
        )
    if np.any(prices_arr <= 0):
        raise ValueError("Price series contains non-positive values; cannot compute returns.")

    daily_returns = np.diff(prices_arr) / prices_arr[:-1]

    rng = np.random.default_rng(seed)
    # For each simulated path, draw `horizon_days` daily returns with replacement
    # and compound them into one total return for that path.
    sampled = rng.choice(daily_returns, size=(num_simulations, horizon_days), replace=True)
    path_returns = np.prod(1 + sampled, axis=1) - 1

    return RiskEstimate(
        win_probability=float(np.mean(path_returns > 0)),
        expected_return=float(np.mean(path_returns)),
        variance=float(np.var(path_returns)),
        num_simulations=num_simulations,
        horizon_days=horizon_days,
    )