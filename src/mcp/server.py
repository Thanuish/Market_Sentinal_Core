from mcp.server.fastmcp import FastMCP

from src.tools.monte_carlo import estimate_risk_monte_carlo as _estimate_risk_monte_carlo
from src.tools.sizing import compute_position_size as _compute_position_size

mcp = FastMCP("quant_tools")


@mcp.tool()
def estimate_risk_monte_carlo(
    prices: list[float], horizon_days: int = 5, num_simulations: int = 10_000
) -> dict:
    """Estimate win probability, expected return, and variance for the next
    `horizon_days` via historical bootstrap Monte Carlo simulation over daily
    returns. Read-only and informational — safe to bind to the evaluator LLM."""
    return _estimate_risk_monte_carlo(prices, horizon_days, num_simulations).model_dump()


@mcp.tool()
def compute_position_size(
    win_probability: float, expected_return: float, variance: float, bankroll: float
) -> dict:
    """Fractional-Kelly position sizing with a hard cap compiled into the code.
    Do not bind this tool to an LLM — only the graph's own deterministic code
    may call it."""
    return _compute_position_size(win_probability, expected_return, variance, bankroll).model_dump()


if __name__ == "__main__":
    mcp.run()