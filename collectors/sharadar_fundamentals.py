"""
collectors/sharadar_fundamentals.py — standardized quarterly fundamentals (primary).

Sharadar SF1, as-reported quarterly (ARQ): ~112 fields per quarter on one
template for every company, history back to the 1990s. `datekey` is the date
the row became available — the point-in-time key. Stored as JSON because the
schema drifts per company (banks and REITs report several fields as null, and
some numerics arrive as strings); the build layer decides types once.

    python -m collectors.sharadar_fundamentals            # collect; skip jobs already ok
    python -m collectors.sharadar_fundamentals --force    # re-fetch everything
    python -m collectors.sharadar_fundamentals --verify   # coverage report, no fetching
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collectors import _sharadar
from collectors._core import cli, collect as _collect, read_stored
from config.tickers import sharadar_tickers

SOURCE = "sharadar_fundamentals"
WINDOW = "ARQ"                       # as-reported quarterly, full history
EXT = "json"
JOBS = [(t, WINDOW) for t in sharadar_tickers()]

# Full ARQ history for one company, oldest quarter first.
def fetch_one(ticker: str, window: str) -> dict:
    return _sharadar.get("fundamentals", ticker, dimension=WINDOW,
                         sort_key=lambda r: r.get("calendardate") or "")

# Stored quarters for one company as a list of dicts, [] if never collected.
def load(ticker: str) -> list:
    stored = read_stored(SOURCE, ticker, WINDOW, EXT)
    return stored["data"] if stored else []

# Collect every company in JOBS. Returns the run tally: ok/empty/error/skip.
def collect(force: bool = False) -> dict:
    return _collect(SOURCE, JOBS, fetch_one, EXT, sleep=0.3, force=force)

# Runs the CLI for collecting Sharadar fundamentals
if __name__ == "__main__":
    cli(SOURCE, JOBS, collect, "Collect Sharadar ARQ fundamentals.")
