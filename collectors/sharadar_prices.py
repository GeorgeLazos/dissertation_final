"""
collectors/sharadar_prices.py — daily OHLCV for equities and REITs (primary).

PRICES ONLY. Sharadar keeps dividends, splits and spin-offs in a separate
typed table (sharadar_actions.py), unlike yfinance which bundles them into one
response.

Columns: open/high/low/close/volume on the CURRENT split basis, closeadj
(splits AND dividends baked in, rewritten retroactively — never point-in-time
safe), closeunadj (as traded on the day).

    python -m collectors.sharadar_prices            # collect; skip jobs already ok
    python -m collectors.sharadar_prices --force    # re-fetch everything
    python -m collectors.sharadar_prices --verify   # coverage report, no fetching
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from collectors import _sharadar
from collectors._core import cli, collect as _collect, read_stored
from config.splits import FETCH_END, FETCH_START
from config.tickers import sharadar_tickers

SOURCE = "sharadar_prices"
WINDOW = f"{FETCH_START}_{FETCH_END}"
EXT = "parquet"
JOBS = [(t, WINDOW) for t in sharadar_tickers()]

# Daily price history for one ticker over the fetch window, oldest first.
def fetch_one(ticker: str, window: str) -> pd.DataFrame:
    rows = _sharadar.get("sep", ticker,
                         **{"date.gte": FETCH_START, "date.lte": FETCH_END})["data"]
    df = pd.DataFrame(rows)
    return df.sort_values("date").reset_index(drop=True) if len(df) else df

# Stored price history for one ticker, or None if never collected.
def load(ticker: str) -> pd.DataFrame | None:
    return read_stored(SOURCE, ticker, WINDOW, EXT)

# Collect every ticker in JOBS. Returns the run tally: ok/empty/error/skip.
def collect(force: bool = False) -> dict:
    return _collect(SOURCE, JOBS, fetch_one, EXT, sleep=0.3, force=force)

# Runs the CLI for collecting Sharadar prices when the script is executed directly
if __name__ == "__main__":
    cli(SOURCE, JOBS, collect, "Collect daily Sharadar prices for equities and REITs.")
