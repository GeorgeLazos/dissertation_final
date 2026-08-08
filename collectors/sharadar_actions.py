"""
collectors/sharadar_actions.py — corporate actions, typed per event (primary).

The other half of the price data: every dividend, split, spin-off and merger
as its own typed record — `action` is one of dividend / split / spinoff /
spinoffdividend / tickerchange / merger / ..., with `value` the cash amount,
split factor or spun-off value per share. The typing is the point: a split
rescales the price history and a spin-off does not, and a source that folds
both into one number cannot tell them apart.

    python -m collectors.sharadar_actions            # collect; skip jobs already ok
    python -m collectors.sharadar_actions --force    # re-fetch everything
    python -m collectors.sharadar_actions --verify   # coverage report, no fetching
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collectors import _sharadar
from collectors._core import cli, collect as _collect, read_stored
from config.tickers import sharadar_tickers

SOURCE = "sharadar_actions"
WINDOW = "all"                       # full event history; the build layer slices
EXT = "json"
JOBS = [(t, WINDOW) for t in sharadar_tickers()]

# Every recorded action for one ticker, oldest to newest.
def fetch_one(ticker: str, window: str) -> dict:
    return _sharadar.get("actions", ticker,
                         sort_key=lambda r: (r.get("date") or "", r.get("action") or ""))

# Stored actions for one ticker as a list of dicts, [] if never collected.
def load(ticker: str) -> list:
    stored = read_stored(SOURCE, ticker, WINDOW, EXT)
    return stored["data"] if stored else []

# Collect every ticker in JOBS. Returns the run tally: ok/empty/error/skip.
def collect(force: bool = False) -> dict:
    return _collect(SOURCE, JOBS, fetch_one, EXT, sleep=0.3, force=force)

# Runs the CLI for collecting Sharadar corporate actions
if __name__ == "__main__":
    cli(SOURCE, JOBS, collect, "Collect Sharadar corporate actions.")
