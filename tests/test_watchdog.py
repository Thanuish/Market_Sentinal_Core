from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.tools.watchdog import run_watchdog


def _trending_price_df(n=260, start=100.0, daily_drift=0.0, noise=0.01, seed=7):
    rng = np.random.default_rng(seed)
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + daily_drift + rng.normal(0, noise)))
    return pd.DataFrame({"Close": prices})


class TestRunWatchdogStock:
    @patch("src.tools.watchdog.yf.Ticker")
    def test_uptrend_produces_golden_cross(self, mock_ticker):
        mock_instance = MagicMock()
        mock_instance.history.return_value = _trending_price_df(daily_drift=0.003)
        mock_ticker.return_value = mock_instance
        result = run_watchdog("AAPL", "stock")
        assert result.ticker == "AAPL"
        assert result.moving_average_cross is True
        assert result.sma_50 > result.sma_200

    @patch("src.tools.watchdog.yf.Ticker")
    def test_downtrend_produces_no_golden_cross(self, mock_ticker):
        mock_instance = MagicMock()
        mock_instance.history.return_value = _trending_price_df(daily_drift=-0.003)
        mock_ticker.return_value = mock_instance
        result = run_watchdog("AAPL", "stock")
        assert result.moving_average_cross is False
        assert result.sma_50 < result.sma_200

    @patch("src.tools.watchdog.yf.Ticker")
    def test_insufficient_history_raises(self, mock_ticker):
        mock_instance = MagicMock()
        mock_instance.history.return_value = _trending_price_df(n=10)
        mock_ticker.return_value = mock_instance
        with pytest.raises(ValueError, match="Insufficient market data"):
            run_watchdog("AAPL", "stock")

    @patch("src.tools.watchdog.yf.Ticker")
    def test_empty_history_raises(self, mock_ticker):
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_instance
        with pytest.raises(ValueError, match="Insufficient market data"):
            run_watchdog("AAPL", "stock")


class TestRunWatchdogCrypto:
    @patch("src.tools.watchdog.requests.get")
    def test_crypto_returns_typed_signals(self, mock_get):
        closes = _trending_price_df()["Close"].tolist()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [[0, "0", "0", "0", str(c), "0"] for c in closes]
        mock_get.return_value = mock_response
        result = run_watchdog("BTC", "crypto")
        assert result.ticker == "BTC"
        assert result.current_price > 0

    @patch("src.tools.watchdog.requests.get")
    def test_crypto_api_error_raises(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 451
        mock_get.return_value = mock_response
        with pytest.raises(ValueError, match="Binance API Error"):
            run_watchdog("BTC", "crypto")


class TestRunWatchdogForex:
    @patch("src.tools.watchdog.yf.Ticker")
    def test_forex_returns_typed_signals(self, mock_ticker):
        mock_instance = MagicMock()
        mock_instance.history.return_value = _trending_price_df()
        mock_ticker.return_value = mock_instance
        result = run_watchdog("EURUSD", "forex")
        assert result.ticker == "EURUSD"
        assert result.current_price > 0

    @patch("src.tools.watchdog.yf.Ticker")
    def test_forex_empty_history_raises(self, mock_ticker):
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_instance
        with pytest.raises(ValueError):
            run_watchdog("EURUSD", "forex")