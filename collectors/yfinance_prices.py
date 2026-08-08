"""
collectors/yfinance_prices.py — daily OHLCV via yfinance.

Primary source for the 15 ETFs/benchmark fundsand for ^VIX;
for the 108 equities/REITs it is the independent second source
the build layer can cross-check Sharadar against. Collected with
auto_adjust=False: Close and Dividends arrive on the current split basis,
dividends not baked in. `Adj Close` is rewritten retroactively by later
corporate actions — never point-in-time safe.

yfinance reads Yahoo's internal endpoints; there is no key and no official API.

    python -m collectors.yfinance_prices            # collect; skip jobs already ok
    python -m collectors.yfinance_prices --force    # re-fetch everything
    python -m collectors.yfinance_prices --verify   # coverage report, no fetching
"""

from __future__ import annotations
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yfinance as yf
from collectors._core import cli, collect as _collect, read_stored
from config.splits import FETCH_END, FETCH_START
from config.tickers import all_tickers

logging.getLogger("yfinance").setLevel(logging.CRITICAL)   # quieten scraping noise

SOURCE = "yfinance_prices"
WINDOW = f"{FETCH_START}_{FETCH_END}"
EXT = "parquet"
JOBS = [(t, WINDOW) for t in all_tickers()]

# yfinance `end` is exclusive — add a day so FETCH_END itself is included.
_REQUEST_END = (date.fromisoformat(FETCH_END) + timedelta(days=1)).isoformat()

# Full daily history for one symbol: unadjusted OHLCV + Adj Close + actions.
def fetch_one(symbol: str, window: str) -> pd.DataFrame:
    return yf.Ticker(symbol).history(
        start=FETCH_START, end=_REQUEST_END,
        auto_adjust=False, actions=True,
    )

# Stored history for one symbol, or None if never collected.
def load(symbol: str) -> pd.DataFrame | None:
    return read_stored(SOURCE, symbol, WINDOW, EXT)

# Collect every symbol in JOBS. Returns the run tally: ok/empty/error/skip.
def collect(force: bool = False) -> dict:
    return _collect(SOURCE, JOBS, fetch_one, EXT, sleep=0.25, force=force)

# Run the CLI for collecting yfinance prices.
if __name__ == "__main__":
    cli(SOURCE, JOBS, collect, "Collect daily OHLCV for the full universe via yfinance.")
