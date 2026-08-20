import numpy as np
import pandas as pd
import requests
import yfinance as yf
from pydantic import BaseModel, Field

class TechnicalSignals(BaseModel):
    """The strict data contract passed from the Watchdog to the LLM State."""
    ticker: str = Field(..., description="Asset ticker symbol.")
    current_price: float = Field(..., description="The most recent closing price.")
    sma_50: float = Field(..., description="50-period Simple Moving Average.")
    sma_200: float = Field(..., description="200-period Simple Moving Average.")
    moving_average_cross: bool = Field(..., description="True if SMA 50 is above SMA 200 (Golden Cross).")
    upper_band: float = Field(..., description="Bollinger Upper Band (+2 Std Dev from 20-SMA).")
    lower_band: float = Field(..., description="Bollinger Lower Band (-2 Std Dev from 20-SMA).")
    volatility_state: str = Field(..., description="High, Low, or Normal based on price relative to bands.")
    volatility_index: float = Field(..., description="30-day annualized volatility.")

def _fetch_crypto_data(symbol: str) -> pd.DataFrame:
    """Fetches 1-year equivalent daily klines from Binance Public API."""
    clean_symbol = symbol.replace('-', '').replace('/', '').upper()
    if not clean_symbol.endswith('USDT'):
        clean_symbol += 'USDT'

    url = f'https://api.binance.com/api/v3/klines?symbol={clean_symbol}&interval=1d&limit=365'
    res = requests.get(url, timeout=5)

    if res.status_code != 200:
        raise ValueError(f'Binance API Error for crypto symbol: {symbol}')

    data = res.json()
    # Extract Close prices (index 4 in Binance kline payload)
    closes = [float(candle[4]) for candle in data]
    df = pd.DataFrame({'Close': closes})
    return df

def _fetch_forex_data(symbol: str) -> pd.DataFrame:
    """Fetches Forex daily data via yfinance ticker convention."""
    formatted_symbol = symbol.upper()
    if not formatted_symbol.endswith('=X'):
        clean = formatted_symbol.replace('/', '').replace('-', '')
        formatted_symbol = f'{clean}=X'

    stock = yf.Ticker(formatted_symbol)
    df = stock.history(period='1y')
    if df.empty:
        raise ValueError(f'Forex data not found for symbol: {symbol}')
    return df

def run_watchdog(ticker: str, asset_type: str = 'stock') -> TechnicalSignals:
    """Universal Watchdog Router computing deterministic math across asset classes."""
    asset_type = asset_type.lower()

    # 1. Universal Ingestion
    if asset_type == 'crypto':
        df = _fetch_crypto_data(ticker)
    elif asset_type == 'forex':
        df = _fetch_forex_data(ticker)
    else:
        stock = yf.Ticker(ticker)
        df = stock.history(period='1y')

    if df.empty or len(df) < 50:
        raise ValueError(f"Watchdog Error: Insufficient market data for '{ticker}'.")

    # 2. Trend Math: 50-day and 200-day Moving Averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # 3. Volatility Math: 20-day Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev_20'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['Std_Dev_20'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['Std_Dev_20'] * 2)

    # 4. Clean Data and Extract Latest
    df = df.dropna()
    latest = df.iloc[-1]
    
    current_price = latest['Close']
    sma_50 = latest['SMA_50']
    sma_200 = latest['SMA_200']
    upper = latest['Upper_Band']
    lower = latest['Lower_Band']

    # 5. Evaluate Trend (Golden Cross)
    is_bullish_cross = bool(sma_50 > sma_200)

    # 6. Evaluate Volatility State (Anomalies)
    if current_price > upper:
        vol_state = "HIGH ANOMALY: Price breached Upper Bollinger Band (Overbought/Breakout)"
    elif current_price < lower:
        vol_state = "HIGH ANOMALY: Price breached Lower Bollinger Band (Oversold/Crash)"
    else:
        vol_state = "NORMAL: Price contained within standard statistical bounds"

    # 7. Annualized Risk (30-day)
    daily_returns = df['Close'].pct_change().dropna()
    volatility = float(daily_returns.tail(30).std() * np.sqrt(252))

    return TechnicalSignals(
        ticker=ticker.upper(),
        current_price=round(current_price, 4 if asset_type == 'forex' else 2),
        sma_50=round(sma_50, 4 if asset_type == 'forex' else 2),
        sma_200=round(sma_200, 4 if asset_type == 'forex' else 2),
        moving_average_cross=is_bullish_cross,
        upper_band=round(upper, 4 if asset_type == 'forex' else 2),
        lower_band=round(lower, 4 if asset_type == 'forex' else 2),
        volatility_state=vol_state,
        volatility_index=round(volatility, 4)
        price_history=df['Close'].tolist(),
    )