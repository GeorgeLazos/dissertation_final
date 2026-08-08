"""
MIDAS — date splits.

Single source of truth for train / validation / test boundaries. Every stage
that touches dates imports from here: the collectors (what range to fetch),
the feature layer, the training loop, and the backtest engine.
"""

import datetime as dt

# ---------------------------------------------------------------------------
# THE SPLITS
# ---------------------------------------------------------------------------

TRAIN_START = "2005-01-01"
TRAIN_END   = "2018-12-31"      # 14 years

VAL_START   = "2019-01-01"
VAL_END     = "2021-12-31"      # 3 years

TEST_START  = "2022-01-01"
TEST_END    = "2026-06-30"      # 4.5 years

SPLITS = {
    "train": (TRAIN_START, TRAIN_END),
    "val":   (VAL_START,   VAL_END),
    "test":  (TEST_START,  TEST_END),
}

# ---------------------------------------------------------------------------
# WHY THESE BOUNDARIES
# ---------------------------------------------------------------------------
# START 2005 — a compromise, not a clean line. GLD lists Nov 2004, but DBC and
#   SLV only in 2006 and DBA in 2007, so the commodity class is incomplete for
#   the first two years. Starting later would give a complete universe but lose
#   the run-up to 2008. Ragged entry is the better trade — assets join the
#   universe when they list, as in a real portfolio.
#
# TRAIN 2005-2018 — contains the 2008 financial crisis, so the agent trains on
#   a genuine crash rather than only calm markets. Also spans the 2009-2018
#   bull run, the low-rate era and the 2015-16 correction: enough regime
#   variety that the policy is not fitted to one market state.
#
# VAL 2019-2021 — COVID crash and recovery. Deliberately a different KIND of
#   shock from 2008: faster, policy-driven, sharper reversal. Hyperparameters
#   tuned here are not tuned on a repeat of the training regime.
#
# TEST 2022-2026 — rate hikes, inflation, and the bond-equity correlation
#   breakdown (2022 was rare in that stocks AND bonds fell together, which
#   punishes naive diversification).

# ---------------------------------------------------------------------------
# WARM-UP
# ---------------------------------------------------------------------------
# Indicators need history before they produce a value; the longest window is
# 12-month momentum. Warm-up data feeds indicator computation only — never
# training samples, never evaluation.

WARMUP_START = "2004-01-01"     # one year of run-up before TRAIN_START
WARMUP_DAYS  = 252

# ---------------------------------------------------------------------------
# FETCH RANGE — what the collectors request, and what names every raw file
# ---------------------------------------------------------------------------

FETCH_START = WARMUP_START      # 2004-01-01
FETCH_END   = TEST_END          # 2026-06-30


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

# The (start, end) dates of one split, or a ValueError naming the valid ones.
def get_split(name: str):
    if name not in SPLITS:
        raise ValueError(f"Unknown split {name!r}. Use one of {list(SPLITS)}.")
    return SPLITS[name]


# Slice a DataFrame to one split — by its index, or by date_col if named.
def filter_dates(df, split: str, date_col=None):
    start, end = get_split(split)
    if date_col is None:
        return df.loc[start:end]
    mask = (df[date_col] >= start) & (df[date_col] <= end)
    return df.loc[mask]


# Is a date inside a split? Accepts a str or a date/datetime.
def in_split(date, split: str) -> bool:
    start, end = get_split(split)
    if isinstance(date, (dt.date, dt.datetime)):
        date = date.isoformat()[:10]
    return start <= date <= end


# Which split a date falls in: 'train' / 'val' / 'test', or None for warm-up.
def which_split(date):
    for name in SPLITS:
        if in_split(date, name):
            return name
    return None


# One line per split, for logs and sanity checks.
def describe():
    lines = []
    for name, (start, end) in SPLITS.items():
        years = int(end[:4]) - int(start[:4]) + 1
        lines.append(f"  {name:5s} {start} -> {end}  ({years} years)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
SPLIT_NOTE = """
LEAKAGE CHECKLIST — verify before reporting any result.

  1. No feature at time t uses data from after t. Applies to every source:
     ALFRED vintages for macro, filing dates for fundamentals, and a
     total-return index compounded forward, never rewritten retroactively.

  2. Scalers and any fitted preprocessing are fit on TRAIN ONLY, then applied
     unchanged to val and test.

  3. The test window is touched once, at the end. If a hyperparameter or
     stopping rule was influenced by test performance, the result is no
     longer out-of-sample.

  4. Warm-up data (2004) computes indicators only — never training samples,
     never evaluation.

  5. Universe membership is fixed across all splits — see UNIVERSE_NOTE in
     tickers.py: survivorship bias inflates all strategies incl. benchmarks.
"""

if __name__ == "__main__":
    print("MIDAS splits:")
    print(describe())
    print(f"\n  fetch range: {FETCH_START} -> {FETCH_END}")
    print(f"  (includes {WARMUP_START} -> {TRAIN_START} warm-up for indicators)")
