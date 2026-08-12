"""
config/portfolio.py — cost rates, clocks and cash conventions for the model
layer. 

The single source of truth: portfolio/run.py and the future RL environment read these; 
the engine itself receives them only as parameters.
"""
from __future__ import annotations
import pandas as pd

# Proportional ONE-WAY costs in basis points, charged per leg on the value
# traded — an ALL-IN implementation cost (commission, half-spread, implicit
# impact), not a bid-ask spread. A round trip therefore costs twice the rate.
#
# Equities/REITs 10bp. Frazzini, Israel & Moskowitz (2018), $1.7tn of live
# institutional executions 1998-2016, put large-cap market impact at a mean
# of 8.90bp (median 5.54); ANcerno reports 10.5bp for trades near 0.5% of
# daily volume. DeMiguel, Nogales & Uppal (2014) adopt exactly 10bp as their
# modest-cost benchmark. 10bp is thus at the TOP of the measured institutional
# range and several times the pure mega-cap spread (Hasbrouck 2009 implies
# ~2bp), so the error runs against the strategies — the safe direction.
# The older 50bp convention traces to Balduzzi & Lynch (1999), calibrated on
# PRE-DECIMALISATION round-trip estimates, and is anachronistic here.
#
# ETFs 5bp is a deliberate conservative UPPER BOUND, not an estimate.
# Petajisto (2011) value-weighted quoted spreads halve to roughly 1.5bp
# one-way for precious metals and 0.5-3.5bp for the bond categories, and GLD
# is measured near 0.3bp one-way today. Only DBC and USO sit close to 5bp.
# DBA in its recent thin years is the one fund this may UNDERSTATE.
#
# Cash 1bp is a stated assumption with no literature behind it: T-bill
# transactions are near-free, but exactly zero would make flipping in and out
# of cash costless and an agent would learn to exploit that.

#Trading cost
COST_BP = {
    "equities": 10.0,
    "reits": 10.0,
    "bonds": 5.0,
    "commodities": 5.0,
    "cash": 1.0,
}

# The write-up commits to a sensitivity table at these levels, so no
# conclusion rests on the 10bp choice.
SENSITIVITY_BP = (0.0, 5.0, 10.0, 25.0)

# One definition of the cash column's name, owned by the engine.
from portfolio.engine import CASH

# ── Environment (the agents' world) ─────────────────
# Every knob in one place. None = the limit is off.

ASSET_CAP = None          # per-asset ceiling on portfolio weight. For a
                          # solo-training sleeve, pass the FINAL cap divided
                          # by the sleeve's assumed class share

BAND = 0.005              # agents' no-trade band (turnover); baselines run 0

REWARD_ETA = 0.01         # differential-Sharpe adaptation rate

TURNOVER_LAMBDA = 0.05    # churn penalty per unit executed turnover, in
                          # differential-Sharpe units (reward std ~1.4);
                          # validation grid spans ~1e-2 to 1

EPISODE_LEN = 252         # sessions per training episode

REWARD_WARMUP = 60        # sessions priming the reward statistics, reward
                          # withheld; shorter warm-ups pay ~2x-inflated
                          # rewards while the variance estimate is thin

AGENT_TRAIN_START = "2005-01-03"   # earliest training session: features are
                                   # warm from the first train day (the 2004
                                   # panel year feeds the 252-session windows);
                                   # remaining blanks are structural and
                                   # carried by the flag columns

# One-way cost rate per column, from the class lists. CASH is the cash rate.
def cost_rates(columns: list) -> pd.Series:
    from config.tickers import all_classes
    by_ticker = {}
    for cls, tickers in all_classes().items():
        for t in tickers:
            by_ticker[t] = COST_BP[cls] / 1e4
    by_ticker[CASH] = COST_BP["cash"] / 1e4
    missing = [c for c in columns if c not in by_ticker]
    if missing:
        raise ValueError(f"no cost rate for {missing[:5]}")
    return pd.Series({c: by_ticker[c] for c in columns})

# Daily cash return from DTB3 (percent, annualised): dtb3 / 100 / 252 on
# trading days. Both the engine's accrual and the Sharpe risk-free MUST come
# from this one function so they cannot drift apart. 
def cash_daily(macro: pd.DataFrame) -> pd.Series:
    s = macro.set_index("date")["dtb3"] if "date" in macro.columns else macro["dtb3"]
    return (s / 100.0 / 252.0).rename("cash_daily")

# Monthly rebalance decisions are taken at the close of the first trading
# day of each month.
def month_starts(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(dates)
    return pd.DatetimeIndex(s.groupby(s.dt.to_period("M")).min().values)
