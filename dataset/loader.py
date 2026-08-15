"""
dataset/loader.py — the single read interface: everything downstream reads
through here.

Every downstream layer imports THIS module and nothing else from the dataset layer. Split slicing lives here too, wired
to config.splits, so no downstream code ever writes a date literal.

INPUT   the four processed tables in data/processed/ (see dataset.TABLES)

    calendar()              DatetimeIndex of the 5,658 trading days
    prices(split=None)      the daily panel, optionally one split
    macro(split=None)       the PIT macro table, optionally one split
    fundamentals(split=None)  filings PUBLISHED by the end of the split —
                            older filings stay: they are still knowledge
    matrices(split=None)    model-facing date x ticker grids:
                            {ret, close, tradeable, macro, dates, tickers}

    python -m dataset.loader          # smoke-read every table, print shapes
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config.splits import get_split
from dataset import calendar as _calendar
from dataset import fundamentals_dataset, macro_dataset, price_dataset


# Loads the calendar table, which is a DatetimeIndex of all trading days.
def calendar() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(_calendar.load()["date"])


# The daily price panel, sliced to a split when one is named.
def prices(split: str | None = None) -> pd.DataFrame:
    p = price_dataset.load()
    if split is not None:
        start, end = get_split(split)
        p = p[(p["date"] >= start) & (p["date"] <= end)]
    return p.reset_index(drop=True)


# The macro table, sliced to a split when one is named.
def macro(split: str | None = None) -> pd.DataFrame:
    m = macro_dataset.load()
    if split is not None:
        start, end = get_split(split)
        m = m[(m["date"] >= start) & (m["date"] <= end)]
    return m.reset_index(drop=True)


# The fundamentals rows PUBLISHED by the end of the split. The start does not
# cut: an old filing is still what a model is allowed to know.
def fundamentals(split: str | None = None) -> pd.DataFrame:
    f = fundamentals_dataset.load()
    if split is not None:
        _start, end = get_split(split)
        f = f[f["published"] <= end]
    return f.reset_index(drop=True)


# Reshapes the data into a grid of dates x tickers, with a column for each of the model's inputs.
def matrices(split: str | None = None) -> dict:
    p = prices(split)
    out = {
        "ret":       p.pivot(index="date", columns="ticker", values="ret"),
        "close":     p.pivot(index="date", columns="ticker", values="close"),
        "tradeable": p.pivot(index="date", columns="ticker",
                             values="tradeable").astype(bool),
        "macro":     macro(split).set_index("date"),
    }
    out["dates"] = out["ret"].index
    out["tickers"] = list(out["ret"].columns)
    return out


if __name__ == "__main__":
    from collectors._core import console_utf8
    console_utf8()
    print(f"calendar      : {len(calendar()):,} days")
    for name, fn in (("prices", prices), ("macro", macro),
                     ("fundamentals", fundamentals)):
        df = fn()
        print(f"{name:13s} : {df.shape}")
    m = matrices("train")
    print(f"matrices(train): ret {m['ret'].shape}, {len(m['tickers'])} tickers, "
          f"macro {m['macro'].shape}")
