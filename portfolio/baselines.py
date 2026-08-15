"""
portfolio/baselines.py — the benchmark strategies: one estimator block,
the producers, and ARMS, the list run.py iterates.

A producer maps market data to a weights frame and returns (weights, info),
the second a dict of counts run.py reports. None knows the engine exists or
what trading costs.

Markowitz, min-variance and risk parity share the estimator block, so the
only difference between them is the objective — min-variance isolates the
damage the mean forecasts do only if everything else is identical.

Conventions (reasoning recorded in self_reports/layer3_design.md): an
optimiser sees only
assets with a full 252-session history ending at the decision date; others
get zero; 1/N uses the tradeable mask alone. Markowitz is long-only
tangency against the cash rate, falling back to min-variance then equal
weight. Covariance is LedoitWolf (scaled-identity target), solver SLSQP
from an equal-weight start; risk parity is closed-form inverse volatility.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from config import portfolio as cfg
from portfolio.engine import CASH

WINDOW = cfg.ESTIMATION_WINDOW   # trailing window for the estimators
FTOL = 1e-12    # the solver's function tolerance for convergence
MAXITER = 1000  # the solver's maximum iterations for convergence

# ── the shared estimator block ──────────────────────────────────────────

# Returns the list of investable tickers on a given date
def eligible(ret: pd.DataFrame, tradeable: pd.DataFrame, d: pd.Timestamp, window: int = WINDOW) -> list:
    i = ret.index.get_loc(d)
    if i + 1 < window:
        raise ValueError(f"only {i + 1} sessions of history before {d.date()}; "
                         f"the estimation window needs {window}")
    block = ret.iloc[i + 1 - window: i + 1]
    live = tradeable.loc[d]
    full = block.notna().all()
    return [t for t in ret.columns if live[t] and full[t]]

# Takes in a list of names tradable on a given date, returns the trailing window of returns,
# the mean vector and the covariance matrix. 
def estimates(ret: pd.DataFrame, tickers: list, d: pd.Timestamp, window: int = WINDOW) -> tuple:
    i = ret.index.get_loc(d)
    block = ret[tickers].iloc[i + 1 - window: i + 1]
    x = block.to_numpy(dtype=float)
    mu = x.mean(axis=0)
    sigma = LedoitWolf().fit(x).covariance_
    return block, mu, sigma

# ── solvers (deterministic: fixed start, fixed tolerances) ──────────────

# Minimum variance baseline solver: long-only minimum variance, ignoring the mean. Returns None if the solver fails.
def _solve_min_var(sigma: np.ndarray, cap: float = 1.0) -> np.ndarray | None:
    k = sigma.shape[0]
    res = minimize(
        lambda w: w @ sigma @ w, np.full(k, 1.0 / k),
        jac=lambda w: 2.0 * sigma @ w,
        bounds=[(0.0, cap)] * k,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                      "jac": lambda w: np.ones(k)}],
        method="SLSQP", options={"ftol": FTOL, "maxiter": MAXITER})
    return res.x if res.success else None

# Returns the best reward-to-risk ratio (tangency) portfolio, or None if no asset has a positive excess mean.
def _solve_tangency(mu_ex: np.ndarray, sigma: np.ndarray, cap: float = 1.0) -> np.ndarray | None:
    if (mu_ex <= 0).all():
        return None
    k = sigma.shape[0]

    def neg_sharpe(w):
        a = mu_ex @ w
        b = np.sqrt(w @ sigma @ w)
        return -a / b

    def jac(w):
        a = mu_ex @ w
        v = sigma @ w
        b = np.sqrt(w @ v)
        return -(mu_ex / b - a * v / b ** 3)

    res = minimize(
        neg_sharpe, np.full(k, 1.0 / k), jac=jac,
        bounds=[(0.0, cap)] * k,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                      "jac": lambda w: np.ones(k)}],
        method="SLSQP", options={"ftol": FTOL, "maxiter": MAXITER})
    return res.x if res.success and res.x @ mu_ex > 0 else None

# Weights vector over the full column set from a solution on a subset.
# The renormalisation only cleans solver dust (the sum constraint holds to
# ftol); a binding weight cap must be enforced in the solver's bounds —
# renormalising through a cap would breach it.
def _frame_row(columns: list, tickers: list, w: np.ndarray) -> pd.Series:
    row = pd.Series(0.0, index=columns)
    w = np.maximum(w, 0.0)
    row[tickers] = w / w.sum()
    return row

# ── producers ───────────────────────────────────────────────────────────
# data: ret (FULL-panel investable returns), tradeable, cash (daily rf),
# columns (investable + CASH last). Baselines are fully invested: CASH 0.

#1/N baseline: equal weight over the tradeable members, ignoring history.
def one_over_n(data: dict, reb: pd.DatetimeIndex) -> tuple:
    rows = []
    for d in reb:
        live = data["tradeable"].loc[d]
        row = pd.Series(0.0, index=data["columns"])
        row[live[live].index] = 1.0 / int(live.sum())
        rows.append(row)
    return pd.DataFrame(rows, index=reb, columns=data["columns"]), {}

# Markowitz baseline: long-only tangency against the cash rate, falling
# back to min-variance when no asset has a positive excess mean, then to
# equal weight if that fails too. Every fallback is counted.
def markowitz(data: dict, reb: pd.DatetimeIndex) -> tuple:
    rows, info = [], {"short_history_rebalances": 0, "minvar_fallbacks": 0,
                      "equal_weight_fallbacks": 0}
    for d in reb:
        tickers = eligible(data["ret"], data["tradeable"], d)
        live_n = int(data["tradeable"].loc[d].sum())
        if len(tickers) < live_n:
            info["short_history_rebalances"] += 1
        block, mu, sigma = estimates(data["ret"], tickers, d)
        rf = float(data["cash"].reindex(block.index).mean())
        w = _solve_tangency(mu - rf, sigma)
        if w is None:
            info["minvar_fallbacks"] += 1
            w = _solve_min_var(sigma)
        if w is None:
            info["equal_weight_fallbacks"] += 1
            w = np.full(len(tickers), 1.0 / len(tickers))
        rows.append(_frame_row(data["columns"], tickers, w))
    return pd.DataFrame(rows, index=reb, columns=data["columns"]), info

# Min-variance baseline: long-only minimum variance, ignoring the mean
def min_variance(data: dict, reb: pd.DatetimeIndex) -> tuple:
    rows, info = [], {"short_history_rebalances": 0,
                      "equal_weight_fallbacks": 0}
    for d in reb:
        tickers = eligible(data["ret"], data["tradeable"], d)
        if len(tickers) < int(data["tradeable"].loc[d].sum()):
            info["short_history_rebalances"] += 1
        _, _, sigma = estimates(data["ret"], tickers, d)
        w = _solve_min_var(sigma)
        if w is None:
            info["equal_weight_fallbacks"] += 1
            w = np.full(len(tickers), 1.0 / len(tickers))
        rows.append(_frame_row(data["columns"], tickers, w))
    return pd.DataFrame(rows, index=reb, columns=data["columns"]), info

# Risk parity baseline: long-only inverse volatility, ignoring the mean
def risk_parity(data: dict, reb: pd.DatetimeIndex) -> tuple:
    rows, info = [], {"short_history_rebalances": 0}
    for d in reb:
        tickers = eligible(data["ret"], data["tradeable"], d)
        if len(tickers) < int(data["tradeable"].loc[d].sum()):
            info["short_history_rebalances"] += 1
        block, _, _ = estimates(data["ret"], tickers, d)
        inv = 1.0 / block.std(ddof=1).to_numpy(dtype=float)
        rows.append(_frame_row(data["columns"], tickers, inv))
    return pd.DataFrame(rows, index=reb, columns=data["columns"]), info

# The fixed-mix producer. Class shares are a caller parameter; equal
# weight within each sleeve over its tradeable members. Class membership
# arrives in the data bundle — producers never read config themselves.
def fixed_mix(class_shares: dict):
    def produce(data: dict, reb: pd.DatetimeIndex) -> tuple:
        classes = data["classes"]
        rows = []
        for d in reb:
            live = data["tradeable"].loc[d]
            row = pd.Series(0.0, index=data["columns"])
            for cls, share in class_shares.items():
                if cls == "cash":
                    row[CASH] = share
                    continue
                members = [t for t in classes[cls] if live.get(t, False)]
                if not members:
                    raise ValueError(f"no tradeable member of {cls} on "
                                     f"{d.date()}")
                row[members] = share / len(members)
            rows.append(row)
        return pd.DataFrame(rows, index=reb, columns=data["columns"]), {}
    return produce

# The arms run.py iterates: name -> producer + parameters. fixed_mix
# joins once a prior allocation is set.
ARMS = {
    "one_over_n": {"producer": one_over_n, "clock": "monthly"},
    "markowitz": {"producer": markowitz, "clock": "monthly"},
    "min_variance": {"producer": min_variance, "clock": "monthly"},
    "risk_parity": {"producer": risk_parity, "clock": "monthly"},
}
