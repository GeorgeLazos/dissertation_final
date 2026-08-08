"""
dataset/price_dataset.py — the daily panel: one row per (date, ticker).

Equities/REITs come from Sharadar: prices from the price table, cash flows
from the typed actions table — the two halves of a total return. Funds come
from yfinance, whose response carries prices and dividends together. Both
sources deliver prices on the CURRENT split basis with dividends on that same
basis, so the return needs no split factor:

    ret = (close + dividend + spinoff_value) / prev_close - 1

The panel is a FULL GRID: every ticker has a row on every calendar day, NaN
prices and tradeable=False before listing — never filled.

INPUT
    calendar.load()           (5,658 x 1) 'date'
    sharadar_prices.load(t)   DataFrame (~5,658 x 10) per 108 tickers:
                              date str, open/high/low/close/volume float,
                              closeadj/closeunadj float, lastupdated str
    sharadar_actions.load(t)  list of event dicts per 108 tickers:
                              {date, action, value, contraticker, ...} where
                              action is 'dividend'/'split'/'spinoffdividend'/...
                              and value sits on the CURRENT split basis
    yfinance_prices.load(t)   DataFrame (~5,658 x 10) per 15 funds:
                              'Date' tz-aware, Open..Close split-adjusted,
                              Dividends, Stock Splits (0 = no event),
                              Capital Gains (all zero — verified)

OUTPUT  data/processed/price_dataset.parquet   (date, ticker) x 12
    date, ticker              keys
    open high low close       float, current split basis. close is closeunadj
                              re-derived through the typed split chain — full
                              precision; open/high/low carry the vendor's
                              3-decimal rounding. A short list of convicted
                              vendor cells is corrected from yfinance
                              (VOLUME_FIXES / BAR_FIXES below).
    volume                    float
    dividend                  cash per share that day, 0.0 if none
    spinoff_value             value distributed per share, 0.0 if none
    split                     factor that day (new shares per old), 1.0 if none
    ret                       daily total return
    tradeable                 close exists that day

    python -m dataset.price_dataset            # build
    python -m dataset.price_dataset --show     # print the stored table
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from collectors import sharadar_actions, sharadar_prices, yfinance_prices
from config.tickers import fund_tickers, sharadar_tickers
from dataset import calendar
from dataset._core import cli, load_table

NAME = "price_dataset"

# Output shape for reference
# date|ticker|open|high|low|close|volume|dividend|spinoff_value|split|ret|tradeable

PRICE_COLS = ["open", "high", "low", "close", "volume"]

# Corrections to the vendor record — dates only, never values. Each cell is
# proven wrong against the ticker's own surrounding history; the replacement
# comes from yfinance, rescaled to Sharadar's counting basis. checks.py
# asserts the raw feed still carries each fault, so a vendor repair upstream
# raises a stale-correction failure instead of double-correcting.
VOLUME_FIXES = (
    ("BNY", "2026-01-02", "2026-02-09"),    # 26 sessions at 0.4-6.7% of median,
                                            # volume 0 on a +2.65% day
    ("BRK-B", "2010-02-12", "2010-02-12"),  # S&P 500 inclusion day, 50.0x low
    ("BRK-B", "2024-12-26", "2025-01-03"),  # year-end dropout, then 4.5x on 12-31
    ("MO", "2024-12-26", "2024-12-26"),     # 14% of median, price moves normally
    ("PM", "2024-12-26", "2024-12-26"),     # 13% of median
    ("GE", "2024-12-27", "2024-12-27"),     # 27% of median
    ("O", "2025-01-02", "2025-01-02"),      # 22% of median
    ("USB", "2024-12-31", "2024-12-31"),    # 14.5x inflated
    ("JPM", "2025-01-06", "2025-01-06"),    # 4.5% of real
    ("MO", "2025-12-10", "2025-12-10"),     # volume 0 on a +0.88% day
)
BAR_FIXES = (
    ("XOM", "2014-07-28"),   # the bar is a byte-copy of the ticker's own 07-30 bar
    ("GS", "2009-11-25"),    # open/high/low copied from 11-27 — the Dubai-crisis
                             # drop booked two days early
    ("WELL", "2014-08-19"),  # bar entirely outside the true session range
    ("NKE", "2014-08-01"),   # bar entirely outside the true session range
    ("PLD", "2009-03-24"),   # close ~7% high on a crisis day; the next open and
                             # both vendors' neighbours agree against it
    # Bad ticks: a low far below (or high far above) the bar's own open/close,
    # contradicted by the second vendor and by the neighbouring days. Each is
    # a fake single-day crash or spike a model could fit. The five 2008-09-19
    # bars are the short-sale-ban session, where Yahoo's 2008-era records
    # (Internet Archive) confirm the true bars; that day's true low IS the
    # close — the market opened up on the ban news and faded all session.
    ("TXN", "2004-03-01"),   # low 17.27 vs true 30.26
    ("PFE", "2004-04-20"),   # high 78.00 vs true ~37.4
    ("TXN", "2004-07-19"),   # high 29.91 vs true 21.09
    ("HON", "2004-12-10"),   # low 27% below the true session low
    ("MDT", "2006-02-03"),   # low 21.40 vs true 55.07
    ("GS", "2006-10-20"),    # low 108.20 vs true 178.85
    ("COST", "2007-08-13"),  # low 20.42 vs true 61.71
    ("BNY", "2008-04-22"),   # low 5.91 vs true 40.38
    ("BRK-B", "2008-07-15"), # low 42.95 vs true 74.60
    ("AMGN", "2008-09-19"),  # low 27.00 vs archived 59.97
    ("CMCSA", "2008-09-19"), # low 4.60 vs ~10.36 on the panel basis
    ("COST", "2008-09-19"),  # low 30.70 vs archived 66.09
    ("CSCO", "2008-09-19"),  # open and low wrong vs archived 24.29
    ("SCHW", "2008-09-19"),  # low 10.75 vs 25.25 — the close-equals-low day
    ("QCOM", "2008-09-29"),  # low 20.63 vs true 38.00
    ("BAC", "2008-10-24"),   # high 28.59 vs true 22.27
    ("LOW", "2009-02-05"),   # high 60.55 vs true 19.05
    ("PLD", "2009-03-16"),   # high 46.74 vs true 13.90
    ("CVX", "2009-05-28"),   # low 29.32 vs true 64.26
    ("TSLA", "2021-03-04"),  # high 291.31 vs true 222.82
)

# 2008-09-19 volume: BOTH live vendors carry this session divided by 100 (a
# shared upstream truncation; every corrupted value is the truth /100 rounded
# to thousands). Restored from Yahoo's own 2008-era pages preserved by the
# Internet Archive. CMCSA has no surviving archived page, so its cell is the
# corrupted value x100 — the restoration its three siblings prove exactly.
#   AMGN web.archive.org/web/20081203234606/finance.yahoo.com/q/hp?s=AMGN
#   CSCO web.archive.org/web/20090103173931 (paged y=66)
#   COST web.archive.org/web/20081103215648/finance.yahoo.com/q/hp?s=COST
#   MS   web.archive.org/web/20081028191401/finance.yahoo.com/q/hp?s=MS
#   SCHW web.archive.org/web/20081025011025/finance.yahoo.com/q/hp?s=SCHW
#   O    web.archive.org/web/20081029034212/finance.yahoo.com/q/hp?s=O
# CMCSA and BNY (BK in 2008) have no surviving archived page: their cells are
# the corrupted value x100, the restoration six archived siblings prove.
VOLUME_RESTORED = (
    ("AMGN", "2008-09-19", 20_283_700.0),
    ("CSCO", "2008-09-19", 92_078_400.0),
    ("COST", "2008-09-19", 10_335_800.0),
    ("MS", "2008-09-19", 121_010_300.0),
    ("SCHW", "2008-09-19", 24_523_400.0),
    ("O", "2008-09-19", 3_974_400.0),
    ("CMCSA", "2008-09-19", 77_500_000.0),
    ("BNY", "2008-09-19", 32_600_000.0),
)


# yfinance frame for one ticker, indexed by naive date.
def _yf(ticker: str) -> pd.DataFrame:
    y = yfinance_prices.load(ticker)
    idx = pd.to_datetime(y["Date"]).dt.tz_localize(None).dt.normalize()
    return y.set_index(idx.astype("datetime64[us]")).sort_index()


# Replace convicted bars with yfinance's, rescaled by the local close ratio —
# the two vendors can sit on different bases (spin-off conventions), and the
# ratio over nearby clean sessions converts between them.
def _fix_bars(ticker: str, out: pd.DataFrame) -> None:
    days = [pd.Timestamp(d) for t, d in BAR_FIXES if t == ticker]
    if not days:
        return
    y = _yf(ticker)
    for day in days:
        near = out["close"].drop(index=day)
        near = near.loc[day - pd.Timedelta(days=15):day + pd.Timedelta(days=15)]
        scale = float((near / y["Close"].reindex(near.index)).median())
        for col, ycol in (("open", "Open"), ("high", "High"),
                          ("low", "Low"), ("close", "Close")):
            out.loc[day, col] = float(y.loc[day, ycol]) * scale


# Replace convicted volume cells with yfinance's, rescaled by the ticker's
# own vendor volume ratio over the surrounding clean sessions.
def _fix_volume(ticker: str, out: pd.DataFrame) -> None:
    for t, d, v in VOLUME_RESTORED:
        if t == ticker and pd.Timestamp(d) in out.index:
            out.loc[pd.Timestamp(d), "volume"] = v
    ranges = [(pd.Timestamp(a), pd.Timestamp(b))
              for t, a, b in VOLUME_FIXES if t == ticker]
    if not ranges:
        return
    yvol = _yf(ticker)["Volume"].astype(float)
    for a, b in ranges:
        window = out.index[(out.index >= a) & (out.index <= b)]
        clean = out.index[(out.index >= a - pd.Timedelta(days=100))
                          & (out.index <= b + pd.Timedelta(days=100))]
        clean = clean.difference(window)
        ours = out.loc[clean, "volume"]
        theirs = yvol.reindex(clean)
        scale = float((ours / theirs[theirs > 0]).median())
        out.loc[window, "volume"] = yvol.reindex(window) * scale

# Build one equity/REIT's table. Sharadar keeps prices and corporate events in
# two places, so this reads both and folds the events into dividend /
# spinoff_value / split columns. Only the days this ticker traded.
def _equity(ticker: str) -> pd.DataFrame:
    d = sharadar_prices.load(ticker)
    d["date"] = pd.to_datetime(d["date"]).astype("datetime64[us]")
    out = d.set_index("date")[PRICE_COLS].astype(float).sort_index()

    # close re-derived at full precision. The vendor's split-adjusted close is
    # rounded to 3 decimals, which quantizes returns where the split chain is
    # deep: 120x of NVDA splits put 2004 at $0.087, so one rounding step is
    # ~1% and small real moves collapse to zero. closeunadj divided through
    # the typed split factors is the same number unrounded; the two must
    # agree to half a rounding step or the chain is wrong and the build stops.
    unadj = d.set_index("date")["closeunadj"].astype(float).sort_index()
    factor = pd.Series(1.0, index=out.index)
    for a in sharadar_actions.load(ticker):
        if a["action"] == "split" and a.get("value") is not None:
            day = pd.Timestamp(a["date"])
            factor.loc[factor.index < day] *= float(a["value"])
    # Allowance per row: the vendor close carries 3-decimal rounding and the
    # published closeunadj its own; the smallest real chain error (a missed
    # 1.02 stock-dividend factor) sits ~20x above this bound.
    precise = unadj / factor
    gap = (precise - out["close"]).abs()
    allow = 0.001 + 0.01 / factor
    if (gap > allow).any():
        worst = float((gap / allow).max())
        raise ValueError(f"{ticker}: split chain disagrees with the vendor "
                         f"close ({int((gap > allow).sum())} rows, worst "
                         f"{worst:.1f}x the rounding allowance)")
    # Dividing by the chain SHARPENS precision only where the factor exceeds
    # one (forward splits). Under a reverse split the factor is below one and
    # the same division AMPLIFIES closeunadj's own rounding 8-20x, so there
    # the vendor's close is the finer number and is kept.
    out["close"] = precise.where(factor > 1.0, out["close"])

    _fix_bars(ticker, out)
    _fix_volume(ticker, out)

    out["dividend"] = 0.0
    out["spinoff_value"] = 0.0
    out["split"] = 1.0
    for a in sharadar_actions.load(ticker):
        day = pd.Timestamp(a["date"])
        v = a.get("value")
        if v is None:
            continue
        if day not in out.index:
            # cash dated on a non-trading day lands on the NEXT trading day —
            # the holder receives it regardless. An event outside the ticker's
            # traded span has no session to land on and is dropped.
            pos = out.index.searchsorted(day)
            if pos == 0 or pos == len(out.index):
                continue
            day = out.index[pos]
        if a["action"] == "dividend":
            out.loc[day, "dividend"] += float(v)
        elif a["action"] == "spinoffdividend":
            out.loc[day, "spinoff_value"] += float(v)
        elif a["action"] == "split":
            out.loc[day, "split"] *= float(v)
    return out

# Build one fund's table. Funds have no spin-offs. Same columns as
# _equity, so build() can treat both the same from here on.
def _fund(ticker: str) -> pd.DataFrame:
    d = yfinance_prices.load(ticker)
    idx = pd.to_datetime(d["Date"]).dt.tz_localize(None).dt.normalize()
    out = pd.DataFrame({
        "open": d["Open"].values, "high": d["High"].values,
        "low": d["Low"].values, "close": d["Close"].values,
        "volume": d["Volume"].values, "dividend": d["Dividends"].values,
    }, index=idx.astype("datetime64[us]")).astype(float).sort_index()
    out["spinoff_value"] = 0.0
    out["split"] = d["Stock Splits"].values
    out.loc[out["split"] == 0.0, "split"] = 1.0
    return out


# Build one table per ticker, then stack them into the full (date, ticker)
# grid. Per-ticker because the return uses shift(1) — on a stacked frame that
# would reach across from one company's last day into the next one's first.
def build() -> pd.DataFrame:
    cal = pd.DatetimeIndex(calendar.load()["date"])
    frames = []
    jobs = [(t, _equity) for t in sharadar_tickers()] + \
           [(t, _fund) for t in fund_tickers()]
    for ticker, make in sorted(jobs):
        df = make(ticker)
        # cash the holder received that day, on the same basis as close
        cash = df["dividend"] + df["spinoff_value"]
        df["ret"] = (df["close"] + cash) / df["close"].shift(1) - 1
        df = df.reindex(cal)                        # full grid; NaN before listing
        df["tradeable"] = df["close"].notna()
        df[["dividend", "spinoff_value"]] = df[["dividend", "spinoff_value"]].fillna(0.0)
        df["split"] = df["split"].fillna(1.0)
        df.insert(0, "ticker", ticker)
        frames.append(df.rename_axis("date").reset_index())
    return pd.concat(frames, ignore_index=True)


# The stored panel: one row per (date, ticker), all 123 instruments.
def load() -> pd.DataFrame:
    return load_table(NAME)


if __name__ == "__main__":
    cli(NAME, build, "Build the daily price panel.")
