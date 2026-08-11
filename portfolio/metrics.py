"""
portfolio/metrics.py — the scoreboard: one run's money path in, the study's
statistics out.

Every formula is the exact variant recorded in self_reports/layer3_design.md;
none has discretion left. The risk-free series must be the SAME object the
engine accrued on cash — and it applies from session 1, matching the
engine, whose first session has no prior close and accrues nothing. Only
full run() outputs are accepted: the first return is measured against the
starting capital, which summary() verifies when given.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ANN = 252

def summary(result: dict, cash_daily: pd.Series, start_value: float | None = None) -> dict:
    v = result["value"]
    r = result["ret"]
    n = len(v)
    rf = cash_daily.reindex(v.index).to_numpy(dtype=float).copy()

    # If risk-free series has any missinf values, raise error
    if np.isnan(rf).any():
        raise ValueError("risk-free series missing inside the run window")
    rf[0] = 0.0

    v0 = v.iloc[0] / (1.0 + r.iloc[0])
    if start_value is not None and abs(v0 - start_value) > 1e-9:
        raise ValueError(f"result is not a full run: first return implies "
                         f"start {v0:.10f}, caller says {start_value:.10f}")
    total = v.iloc[-1] / v0
    cagr = total ** (ANN / n) - 1.0

    x = r.to_numpy(dtype=float)
    vol = float(np.std(x, ddof=1)) * np.sqrt(ANN) 

    excess = x - rf
    sharpe = float(np.mean(excess)) / float(np.std(excess, ddof=1)) * np.sqrt(ANN)

    downside = np.sqrt(np.mean(np.minimum(x, 0.0) ** 2)) * np.sqrt(ANN)
    sortino = float(np.mean(x)) * ANN / downside if downside > 0 else np.inf

    peak = v.cummax()
    mdd = float((v / peak - 1.0).min())
    calmar = cagr / abs(mdd) if mdd < 0 else np.inf

    # The first trade is the first executed rebalance, which need not be the
    # first session (a warm-up-delayed strategy buys in later).
    t_arr = result["turnover"].to_numpy(dtype=float)
    traded = np.nonzero(t_arr)[0]
    first = int(traded[0]) if traded.size else None

    # Cost drag as a fraction of the value it was charged against; the
    # engine's value series is post-cost, so pre-trade value is value + cost.
    c_arr = result["cost"].to_numpy(dtype=float)
    v_arr = v.to_numpy(dtype=float)
    drag = c_arr / (v_arr + c_arr)

    years = n / ANN
    return {
        "annual_return": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "turnover_annual": float(result["turnover"].sum()) / years,
        "costs_total": float(result["cost"].sum()),
        "costs_annual_bp": float(drag.sum()) / years * 1e4,
        "first_trade_cost": float(c_arr[first]) if first is not None else 0.0,
        "first_trade_turnover": float(t_arr[first]) if first is not None else 0.0,
        "max_asset_weight": float(result["weights"].iloc[:, :-1].max().max()),
        "sessions": n,
    }

# Calendar-year compounded returns; a year with fewer than 240 sessions is
# labelled partial.
def per_year(result: dict) -> pd.DataFrame:
    r = result["ret"]
    g = (1.0 + r).groupby(r.index.year)
    out = pd.DataFrame({
        "return": g.prod() - 1.0,
        "sessions": g.count(),
    })
    out["label"] = [f"{y}" + ("" if s >= 240 else " (partial)")
                    for y, s in zip(out.index, out["sessions"])]
    return out
