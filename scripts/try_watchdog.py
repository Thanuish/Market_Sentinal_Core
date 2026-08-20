# try_watchdog.py -- one-off manual test, run and discard
import time
from src.tools.watchdog import run_watchdog

tests = [
    ("BTC", "crypto"),
    ("EURUSD", "forex"),
    ("NVDA", "stock"),
]

for ticker, asset_type in tests:
    print(f"--- {ticker} ({asset_type}) ---")
    start = time.perf_counter()
    try:
        result = run_watchdog(ticker, asset_type)
        elapsed = time.perf_counter() - start
        print(f"  time: {elapsed:.2f}s")
        print(f"  current_price: {result.current_price}")
        print(f"  sma_50: {result.sma_50}  sma_200: {result.sma_200}")
        print(f"  golden_cross: {result.moving_average_cross}")
        print(f"  upper_band: {result.upper_band}  lower_band: {result.lower_band}")
        print(f"  volatility_state: {result.volatility_state}")
        print(f"  volatility_index (30d annualized): {result.volatility_index}")
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"  time: {elapsed:.2f}s")
        print(f"  ERROR ({type(e).__name__}): {e}")
    print()
