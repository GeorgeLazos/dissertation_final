"""
portfolio/engine.py — the simulator every strategy runs through.

Decisions in, reality applied, money path out. The engine is a pure
mechanism: columns, cost rates and returns arrive as parameters; it imports
no config and no data layer, which is what lets the same code run a
three-asset test case or the full universe, and lets the RL environment
wrap step() rather than reimplement the accounting.

    validate_weights(w, columns)   the frame contract, enforced on entry
    police_rets(...)               one session's return sanitisation, shared
                                   with the RL environment
    step(...)                      one day of accounting
    run(...)                       validate, then a fold of step() over the
                                   window — deliberately nothing more, so a
                                   recorded weights frame replays to the
                                   identical path

Conventions (reasoning recorded in self_reports/layer3_design.md): a weights
row is indexed by the DECISION date; the fill is at that session's close,
after the day's returns land on the prior holdings, so the target's first
return exposure is the NEXT session. Cost is charged against pre-trade value
and holdings renormalise on the reduced value. CASH is the last column, and
its "return" is the daily accrual the caller assembles into the returns
frame, so the engine needs no cash logic of its own.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CASH = "CASH"

# The weights-frame contract. Violations raise; nothing is repaired.
def validate_weights(w: pd.DataFrame, columns: list, tol: float = 1e-6) -> None:

    # Is it a dataframe?
    if not isinstance(w, pd.DataFrame):
        raise TypeError("weights must be a DataFrame indexed by decision date")

    # Is the last column CASH?
    if list(columns)[-1] != CASH:
        raise ValueError(f"the column list must end with {CASH}; "
                         f"it ends with {list(columns)[-1]!r}")

    # Is the index a DatetimeIndex?
    if not isinstance(w.index, pd.DatetimeIndex):
        raise ValueError("weights must be indexed by decision DATES")

    # Do colomns match exacly, in order?
    if list(w.columns) != list(columns):
        missing = [c for c in columns if c not in w.columns]
        extra = [c for c in w.columns if c not in columns]
        raise ValueError(f"weights columns must match the run's columns "
                         f"exactly and in order; missing={missing[:5]} "
                         f"extra={extra[:5]}")

    # Is the index strictly increasing, no duplicates?
    if not w.index.is_monotonic_increasing or w.index.has_duplicates:
        raise ValueError("weights index must be strictly increasing, "
                         "no duplicates")

    # Are there any NaN?
    if w.isna().any().any():
        bad = w.index[w.isna().any(axis=1)][:3].tolist()
        raise ValueError(f"weights contain NaN — a blank is an error, never "
                         f"an implied zero; first offending rows: {bad}")

    # Are there any negative weights?
    if (w.to_numpy() < -tol).any():
        raise ValueError("negative weights: the engine is long-only")
    s = w.sum(axis=1).to_numpy()

    # Do the rows sum to 1?
    if np.abs(s - 1.0).max() > tol:
        i = int(np.abs(s - 1.0).argmax())
        raise ValueError(f"weights rows must sum to 1; {w.index[i].date()} "
                         f"sums to {s[i]:.10f}")

# Sanitise one session's return row for a held portfolio: the first session
# has no prior close so nothing accrues, and a NaN return may exist only on
# columns not currently held. Shared by run() and the RL environment, so
# training and evaluation police returns identically.
def police_rets(holdings: np.ndarray, rets: np.ndarray, columns: list, date, first: bool) -> np.ndarray:
    if first:
        return np.zeros(len(holdings))
    live = (holdings != 0.0) & np.isnan(rets)
    if live.any():
        j = int(np.argmax(live))
        raise ValueError(f"holding {columns[j]} on {date.date()}, where its "
                         f"return is undefined")
    return np.nan_to_num(rets, nan=0.0)

# One day. Returns land on what is held; then, if a target was decided,
# trade to it. A target closer to the drifted weights than the no-trade
# band is not executed. Returns the new holdings (value units) and the
# day's record.
def step(holdings: np.ndarray, rets: np.ndarray, target: np.ndarray | None, rates: np.ndarray, band: float = 0.0) -> tuple:
    h = holdings * (1.0 + rets)
    gross = float(h.sum())

    # If no decision to be made today, return
    if target is None:
        return h, {"value": gross, "cost": 0.0, "turnover": 0.0}

    drifted = h / gross
    delta = target - drifted
    turnover = 0.5 * float(np.abs(delta).sum())
    if turnover < band:
        return h, {"value": gross, "cost": 0.0, "turnover": 0.0}

    cost = float(np.abs(delta) @ rates) * gross
    net = gross - cost
    return target * net, {"value": net, "cost": cost, "turnover": turnover}

# The fold: step() over every session in ret's index. ret must carry the
# CASH column last (its values are the daily accrual). The portfolio starts
# entirely in cash, so the first allocation pays costs like any trade.
def run(weights: pd.DataFrame, ret: pd.DataFrame, rates: pd.Series,band: float = 0.0, start_value: float = 1.0) -> dict:
    columns = list(ret.columns)

    # Validate the inputs. 
    if not ret.index.is_monotonic_increasing or ret.index.has_duplicates:
        raise ValueError("ret index must be strictly increasing, no duplicates")
    validate_weights(weights, columns)
    if not weights.index.isin(ret.index).all():
        stray = weights.index[~weights.index.isin(ret.index)][:3].tolist()
        raise ValueError(f"weights dates outside the return index: {stray}")

    R = ret.to_numpy(dtype=float)
    if np.isnan(R[:, -1]).any():
        n = int(np.isnan(R[:, -1]).sum())
        raise ValueError(f"cash accrual missing on {n} sessions")
    rate_vec = rates.reindex(columns).to_numpy(dtype=float)
    if np.isnan(rate_vec).any():
        raise ValueError("every column needs a cost rate")

    dates = ret.index
    targets = {d: w.to_numpy(dtype=float) for d, w in weights.iterrows()}

    n, k = len(dates), len(columns)

    # Start with only cash
    holdings = np.zeros(k)
    holdings[-1] = start_value

    # Build the output arrays.
    value = np.empty(n)
    cost = np.zeros(n)
    turnover = np.zeros(n)
    held = np.empty((n, k))

    # Runs the engine over every session
    for i, d in enumerate(dates):
        rets = police_rets(holdings, R[i], columns, d, i == 0)
        holdings, rec = step(holdings, rets, targets.get(d), rate_vec, band)
        value[i], cost[i], turnover[i] = rec["value"], rec["cost"], rec["turnover"]
        held[i] = holdings / rec["value"]

    return {
        "value": pd.Series(value, index=dates, name="value"),
        "ret": pd.Series(value, index=dates).pct_change()
                 .fillna(value[0] / start_value - 1.0).rename("ret"),
        "cost": pd.Series(cost, index=dates, name="cost"),
        "turnover": pd.Series(turnover, index=dates, name="turnover"),
        "weights": pd.DataFrame(held, index=dates, columns=columns),
    }
