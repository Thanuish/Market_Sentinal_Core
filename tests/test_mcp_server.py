import inspect

import pytest

from src.tools.monte_carlo import RiskEstimate, estimate_risk_monte_carlo
from src.tools.sizing import KELLY_FRACTION, MAX_POSITION_PCT, compute_position_size


def _uptrend_prices(n=60, start=100.0, daily_drift=0.001, noise=0.01, seed=1):
    import random

    rng = random.Random(seed)
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + daily_drift + rng.gauss(0, noise)))
    return prices


class TestEstimateRiskMonteCarlo:
    def test_returns_typed_result_with_reasonable_ranges(self):
        result = estimate_risk_monte_carlo(_uptrend_prices(), horizon_days=5, num_simulations=5000)
        assert isinstance(result, RiskEstimate)
        assert 0.0 <= result.win_probability <= 1.0
        assert result.variance > 0
        assert result.num_simulations == 5000
        assert result.horizon_days == 5

    def test_is_deterministic_given_same_seed(self):
        prices = _uptrend_prices()
        r1 = estimate_risk_monte_carlo(prices, seed=42)
        r2 = estimate_risk_monte_carlo(prices, seed=42)
        assert r1 == r2

    def test_too_few_price_points_raises(self):
        with pytest.raises(ValueError, match="at least 30 days"):
            estimate_risk_monte_carlo([100.0] * 10)

    def test_non_positive_price_raises(self):
        with pytest.raises(ValueError, match="non-positive"):
            estimate_risk_monte_carlo([100.0] * 20 + [-5.0] * 20)

    def test_zero_horizon_days_raises(self):
        with pytest.raises(ValueError, match="horizon_days"):
            estimate_risk_monte_carlo(_uptrend_prices(), horizon_days=0)

    def test_zero_simulations_raises(self):
        with pytest.raises(ValueError, match="num_simulations"):
            estimate_risk_monte_carlo(_uptrend_prices(), num_simulations=0)


class TestComputePositionSize:
    def test_strong_edge_gets_capped_at_hard_limit(self):
        result = compute_position_size(
            win_probability=0.7, expected_return=0.05, variance=0.001, bankroll=100_000
        )
        assert result.capped is True
        assert result.position_pct == MAX_POSITION_PCT
        assert result.position_dollars == MAX_POSITION_PCT * 100_000
        assert result.full_kelly_pct > MAX_POSITION_PCT

    def test_small_edge_stays_under_cap(self):
        result = compute_position_size(
            win_probability=0.51, expected_return=0.0005, variance=0.02, bankroll=100_000
        )
        assert result.capped is False
        assert result.position_pct == pytest.approx(KELLY_FRACTION * (0.0005 / 0.02))

    @pytest.mark.parametrize("expected_return", [-0.01, 0.0])
    def test_no_edge_means_no_position(self, expected_return):
        result = compute_position_size(
            win_probability=0.4, expected_return=expected_return, variance=0.01, bankroll=100_000
        )
        assert result.position_pct == 0.0
        assert result.position_dollars == 0.0
        assert result.capped is False

    @pytest.mark.parametrize("bad_prob", [-0.1, 1.5])
    def test_invalid_win_probability_raises(self, bad_prob):
        with pytest.raises(ValueError, match="win_probability"):
            compute_position_size(bad_prob, 0.02, 0.01, 100_000)

    @pytest.mark.parametrize("bad_var", [0.0, -0.01])
    def test_non_positive_variance_raises(self, bad_var):
        with pytest.raises(ValueError, match="variance"):
            compute_position_size(0.6, 0.02, bad_var, 100_000)

    @pytest.mark.parametrize("bad_bankroll", [0, -1000])
    def test_non_positive_bankroll_raises(self, bad_bankroll):
        with pytest.raises(ValueError, match="bankroll"):
            compute_position_size(0.6, 0.02, 0.01, bad_bankroll)

    def test_hard_cap_is_not_a_caller_settable_parameter(self):
        sig = inspect.signature(compute_position_size)
        assert "max_position_pct" not in sig.parameters
        assert "kelly_fraction" not in sig.parameters