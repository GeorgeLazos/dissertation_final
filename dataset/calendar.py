"""
dataset/calendar.py — the trading-day spine every other table aligns to.

A trading day is any day at least one of the 123 tradeable instruments
traded, taken from each instrument's PRIMARY price source: Sharadar for the
108 equities/REITs, yfinance for the 15 funds. ^VIX is excluded — an index
prints values on market holidays and would inject phantom sessions.

INPUT
    sharadar_prices.load(t)   DataFrame (~5,658 x 10), one per 108 tickers;
                              'date' as 'YYYY-MM-DD' strings
    yfinance_prices.load(t)   DataFrame (~5,658 x 10), one per 15 funds;
                              'Date' tz-aware datetime64[ms, America/New_York]

OUTPUT  data/processed/calendar.parquet
    date    datetime64, sorted, unique — one row per trading day

    python -m dataset.calendar            # build
    python -m dataset.calendar --show     # print the stored table
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from collectors import sharadar_prices, yfinance_prices
from config.splits import FETCH_END, FETCH_START
from config.tickers import fund_tickers, sharadar_tickers
from dataset._core import cli, load_table

NAME = "calendar"


# The union of every instrument's trading dates, bounded to the fetch window,
# as one sorted 'date' column.
def build() -> pd.DataFrame:
    days: set = set()
    for t in sharadar_tickers():
        d = sharadar_prices.load(t)
        days.update(pd.to_datetime(d["date"]))
    for t in fund_tickers():
        d = yfinance_prices.load(t)
        dates = pd.to_datetime(d["Date"]).dt.tz_localize(None).dt.normalize()
        days.update(dates)
    cal = pd.Series(sorted(days), name="date").astype("datetime64[us]")
    cal = cal[(cal >= FETCH_START) & (cal <= FETCH_END)]
    return cal.to_frame().reset_index(drop=True)


# The stored calendar as a DataFrame with one 'date' column.
def load() -> pd.DataFrame:
    return load_table(NAME)


if __name__ == "__main__":
    cli(NAME, build, "Build the trading-day calendar.")
