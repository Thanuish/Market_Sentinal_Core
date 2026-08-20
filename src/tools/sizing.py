from pydantic import BaseModel, Field

# Risk policy constants — deliberately NOT function parameters. Nothing that calls
# this function, human or machine, can loosen these without editing this file.
KELLY_FRACTION = 0.5       # half-Kelly: trade off some theoretical growth for much less variance
MAX_POSITION_PCT = 0.05    # hard cap: never more than 5% of bankroll in a single position


class PositionSizeResult(BaseModel):
    position_pct: float = Field(..., description="Fraction of bankroll to allocate, after Kelly fraction and hard cap.")
    position_dollars: float
    full_kelly_pct: float = Field(..., description="What uncapped full Kelly would have suggested — for audit/logging, never used directly.")
    capped: bool = Field(..., description="True if the hard cap, not the Kelly math, determined the final size.")


def compute_position_size(
    win_probability: float,
    expected_return: float,
    variance: float,
    bankroll: float,
) -> PositionSizeResult:
    """Fractional-Kelly position sizing using the continuous-return formulation
    (f* = expected_return / variance), appropriate for a continuous-return asset
    rather than a discrete win/loss bet. Must only ever be called by graph code —
    never bind this to an LLM's tool-calling loop.
    """
    if not (0.0 <= win_probability <= 1.0):
        raise ValueError(f"win_probability must be in [0, 1], got {win_probability}.")
    if variance <= 0:
        raise ValueError(f"variance must be positive, got {variance}.")
    if bankroll <= 0:
        raise ValueError(f"bankroll must be positive, got {bankroll}.")

    # Negative or zero expected return: no edge, no position. Don't let the
    # formula's algebra produce a nonsensical negative "short via Kelly" —
    # this system is long-only and simply sits out when there's no edge.
    if expected_return <= 0:
        return PositionSizeResult(
            position_pct=0.0, position_dollars=0.0, full_kelly_pct=0.0, capped=False
        )

    full_kelly_pct = expected_return / variance
    fractional_pct = KELLY_FRACTION * full_kelly_pct

    final_pct = min(fractional_pct, MAX_POSITION_PCT)
    capped = fractional_pct > MAX_POSITION_PCT

    return PositionSizeResult(
        position_pct=final_pct,
        position_dollars=final_pct * bankroll,
        full_kelly_pct=full_kelly_pct,
        capped=capped,
    )