"""
portfolio/checks.py — is the arithmetic right?

Four families, each a function returning a list of violation strings;
the CLI exits non-zero if any family reports one.

    reference   an eight-day, three-asset-plus-cash scenario computed by an
                INDEPENDENT naive implementation (plain dict loops, no
                numpy, no shared code) — the engine must reproduce every
                value, cost, turnover and PER-DAY weight to 1e-12, with and
                without a no-trade band, and per-day cost must reconcile
                with turnover under the cost-rate bounds. Agreement between
                two separately written paths is the evidence; one path
                checking itself is not.
    contract    every rejection branch is exercised, by an input built to
                violate a single rule wherever one is constructible, so a
                branch cannot silently die behind another.
    metrics     the metric formulas against values computed longhand,
                including the per-year table and the session-1 risk-free
                convention.
    baselines   real data: producers honour their stated conventions at
                sampled dates; min-variance is verified against its own
                KKT optimality conditions, risk parity against the w*vol
                identity, fixed_mix against exact class shares.

The mask assertion (no investable ticker ever flips tradeable
True->False after entry) lives with the mask, in run.py, not here: the
engine never sees the mask.

    python -m portfolio.checks --all
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from portfolio import engine, metrics

# ── the reference scenario ──────────────────────────────────────────────
# dtb3 at 2.52 percent annualised is exactly one basis point per session.
COSTS = {"AAA": 0.0010, "BBB": 0.0005, "CCC": 0.0010, "CASH": 0.0001}
CASH_DAILY = 2.52 / 100.0 / 252.0
RETURNS = {
    1: {"AAA": +0.0100, "BBB": -0.0050, "CCC": None},
    2: {"AAA": -0.0200, "BBB": +0.0025, "CCC": None},
    3: {"AAA": +0.0050, "BBB": +0.0010, "CCC": None},
    4: {"AAA": +0.0150, "BBB": -0.0020, "CCC": None},
    5: {"AAA": -0.0100, "BBB": +0.0040, "CCC": +0.0200},
    6: {"AAA": +0.0080, "BBB": -0.0010, "CCC": -0.0150},
    7: {"AAA": +0.0020, "BBB": +0.0030, "CCC": +0.0050},
}
TARGETS = {
    0: {"AAA": 0.50, "BBB": 0.30, "CCC": 0.00, "CASH": 0.20},
    4: {"AAA": 0.40, "BBB": 0.25, "CCC": 0.25, "CASH": 0.10},
}
DAYS = 8
COLS = ["AAA", "BBB", "CCC", "CASH"]


# The independent implementation: explicit day-by-day dict arithmetic.
# Returns per-day value, cost, turnover AND per-day weights, so the
# engine's weights output is pinned on every session, not just the last.
def _naive(band: float = 0.0, costs: dict | None = None) -> dict:
    costs = costs or COSTS
    holdings = {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0, "CASH": 1.0}
    value, cost, turnover, shares = [], {}, {}, []
    for day in range(DAYS):
        if day > 0:
            for a in ("AAA", "BBB", "CCC"):
                r = RETURNS[day][a]
                if r is not None:
                    holdings[a] *= 1.0 + r
            holdings["CASH"] *= 1.0 + CASH_DAILY
        gross = sum(holdings.values())
        if day in TARGETS:
            target = TARGETS[day]
            drifted = {k: v / gross for k, v in holdings.items()}
            delta = {k: target[k] - drifted[k] for k in holdings}
            t = 0.5 * sum(abs(d) for d in delta.values())
            if t >= band:
                c = sum(abs(delta[k]) * costs[k] for k in holdings) * gross
                turnover[day], cost[day] = t, c
                gross -= c
                holdings = {k: target[k] * gross for k in holdings}
        value.append(gross)
        shares.append({k: v / gross for k, v in holdings.items()})
    return {"value": value, "cost": cost, "turnover": turnover,
            "holdings": holdings, "shares": shares}


def _scenario_frames() -> tuple:
    days = pd.date_range("2005-01-03", periods=DAYS, freq="B")
    ret = pd.DataFrame(
        {a: [np.nan] + [RETURNS[d][a] if RETURNS[d][a] is not None else np.nan
                        for d in range(1, DAYS)] for a in ("AAA", "BBB", "CCC")},
        index=days)
    ret["CASH"] = CASH_DAILY
    w = pd.DataFrame([[TARGETS[d][c] for c in COLS] for d in sorted(TARGETS)],
                     index=[days[d] for d in sorted(TARGETS)], columns=COLS)
    rates = pd.Series(COSTS).reindex(COLS)
    return w, ret, rates


def check_reference() -> list:
    bad = []
    w, ret, rates = _scenario_frames()
    for band, label in ((0.0, "band 0"), (0.30, "band 0.30")):
        naive = _naive(band)
        out = engine.run(w, ret, rates, band=band)
        dv = max(abs(naive["value"][i] - out["value"].iloc[i])
                 for i in range(DAYS))
        dc = max(abs(naive["cost"].get(i, 0.0) - out["cost"].iloc[i])
                 for i in range(DAYS))
        dt = max(abs(naive["turnover"].get(i, 0.0) - out["turnover"].iloc[i])
                 for i in range(DAYS))
        dw = max(abs(naive["shares"][i][c] - out["weights"].iloc[i][c])
                 for i in range(DAYS) for c in COLS)
        if max(dv, dc, dt, dw) > 1e-12:
            bad.append(f"reference ({label}): engine deviates from the naive "
                       f"path — value {dv:.2e}, cost {dc:.2e}, "
                       f"turnover {dt:.2e}, per-day weights {dw:.2e}")
        fh = out["weights"].iloc[-1] * out["value"].iloc[-1]
        dh = max(abs(naive["holdings"][c] - fh[c]) for c in COLS)
        if dh > 1e-12:
            bad.append(f"reference ({label}): final holdings deviate {dh:.2e}")

    # The band must actually bite: day-4 turnover 0.25 sits below 0.30.
    skipped = engine.run(w, ret, rates, band=0.30)
    if skipped["cost"].iloc[4] != 0.0 or skipped["turnover"].iloc[4] != 0.0:
        bad.append("reference: a 0.30 band failed to skip the 0.25-turnover "
                   "rebalance")
    if skipped["value"].iloc[-1] == engine.run(w, ret, rates)["value"].iloc[-1]:
        bad.append("reference: banded and unbanded paths ended identical, "
                   "so the band changed nothing")

    # Cost must reconcile with turnover: rate bounds on every day, and
    # exact equality when every instrument costs the same.
    out = engine.run(w, ret, rates)
    lo, hi = rates.min(), rates.max()
    for i in range(DAYS):
        t, c = out["turnover"].iloc[i], out["cost"].iloc[i]
        v_pre = out["value"].iloc[i] + c
        if not (2 * t * v_pre * lo - 1e-15 <= c <= 2 * t * v_pre * hi + 1e-15):
            bad.append(f"reconcile: day {i} cost {c:.2e} outside rate bounds "
                       f"for turnover {t:.4f}")
    flat = {k: 0.0010 for k in COSTS}
    nf = _naive(costs=flat)
    of = engine.run(w, ret, pd.Series(flat).reindex(COLS))
    for day, c in nf["cost"].items():
        t = nf["turnover"][day]
        v_pre = nf["value"][day] + c
        if abs(c - 2 * t * v_pre * 0.0010) > 1e-15:
            bad.append(f"reconcile: uniform-rate identity fails on day {day}")
        if abs(c - of["cost"].iloc[day]) > 1e-15:
            bad.append(f"reconcile: engine disagrees with naive uniform-rate "
                       f"cost on day {day}")
    return bad


def check_contract() -> list:
    bad = []
    w, ret, rates = _scenario_frames()

    def rejected(label, weights=None, ret_=None, rates_=None):
        try:
            engine.run(w if weights is None else weights,
                       ret if ret_ is None else ret_,
                       rates if rates_ is None else rates_)
            bad.append(f"contract: {label} was accepted")
        except (ValueError, TypeError):
            pass

    def mutate(fn):
        w2 = w.copy()
        fn(w2)
        return w2

    # Each case violates one rule wherever one is constructible, so a dead
    # branch cannot hide behind a neighbouring rejection. The missing-column
    # frame is renormalised so its rows still sum to 1 — otherwise the
    # row-sum guard would mask a dead column-match branch.
    rejected("not a DataFrame at all",
             weights=w.to_numpy())
    dropped = w.drop(columns=["CCC"])
    rejected("missing column (rows renormalised)",
             weights=dropped.div(dropped.sum(axis=1), axis=0))
    rejected("extra column",
             weights=pd.concat(
                 [w[["AAA", "BBB", "CCC"]], pd.DataFrame(
                     {"DDD": [0.0, 0.0]}, index=w.index), w[["CASH"]]], axis=1))
    rejected("columns in the wrong order",
             weights=w[["BBB", "AAA", "CCC", "CASH"]])
    rejected("NaN cell (sums untouched elsewhere)",
             weights=mutate(lambda x: x.iloc.__setitem__((0, 0), np.nan)))
    # The one negative sits in a row that sums to 1 AND holds nothing
    # undefined, so ONLY the long-only rule can reject it — a 0.40 CCC
    # slot here would hand the rejection to the NaN guard instead.
    rejected("negative weight in a row still summing to 1",
             weights=mutate(lambda x: x.iloc.__setitem__(
                 (0, slice(None)), [-0.10, 0.90, 0.00, 0.20])))
    rejected("row not summing to 1",
             weights=mutate(lambda x: x.iloc.__setitem__((0, 0), 0.70)))
    rejected("duplicated decision date",
             weights=pd.concat([w, w.iloc[[0]]]).sort_index())
    rejected("decreasing index",
             weights=w.iloc[::-1])
    rejected("integer index",
             weights=w.reset_index(drop=True))
    rejected("weights date outside the return index",
             weights=w.rename(index={w.index[1]: w.index[1]
                                     + pd.Timedelta(days=90)}))
    rejected("ret without CASH last",
             ret_=ret[["AAA", "BBB", "CASH", "CCC"]])
    rejected("decreasing ret index",
             ret_=ret.iloc[::-1])
    rejected("NaN in the cash accrual",
             ret_=ret.assign(CASH=[CASH_DAILY] * 3 + [np.nan]
                             + [CASH_DAILY] * 4))
    rejected("missing cost rate",
             rates_=rates.drop("BBB"))

    # Holding an asset on a day its return is undefined must raise: CCC has
    # no return until day 5, so demanding it on day 0 is exposure to a NaN.
    w3 = w.copy()
    w3.iloc[0] = [0.40, 0.30, 0.10, 0.20]
    try:
        engine.run(w3, ret, rates)
        bad.append("contract: a position held over an undefined return was "
                   "accepted")
    except ValueError:
        pass
    return bad


def check_metrics() -> list:
    bad = []
    # Four sessions with a NON-FLAT risk-free, so the session-1 convention
    # (no risk-free on the first session, matching the engine's accrual)
    # is actually discriminated.
    days = pd.date_range("2005-01-03", periods=4, freq="B")
    r = [0.01, -0.02, 0.015, 0.005]
    v = np.cumprod([1.0 + x for x in r])
    result = {
        "value": pd.Series(v, index=days),
        "ret": pd.Series(r, index=days),
        "cost": pd.Series([0.001, 0, 0, 0], index=days),
        "turnover": pd.Series([0.5, 0, 0, 0], index=days),
        "weights": pd.DataFrame(
            {"AAA": [0.6] * 4, "CASH": [0.4] * 4}, index=days),
    }
    rf_vals = [0.0400, 0.0001, 0.0002, 0.0003]
    rf = pd.Series(rf_vals, index=days)
    m = metrics.summary(result, rf, start_value=1.0)

    # Longhand recomputation of each formula. rf on session 0 is zero.
    total = float(np.prod([1.0 + x for x in r]))
    cagr = total ** (252 / 4) - 1.0
    vol = float(np.std(r, ddof=1)) * np.sqrt(252)
    ex = [r[0] - 0.0] + [r[i] - rf_vals[i] for i in (1, 2, 3)]
    sharpe = float(np.mean(ex)) / float(np.std(ex, ddof=1)) * np.sqrt(252)
    downside = float(np.sqrt(np.mean([min(x, 0.0) ** 2 for x in r]))) * np.sqrt(252)
    sortino = float(np.mean(r)) * 252 / downside
    running_peak = np.maximum.accumulate(v)
    mdd = float((v / running_peak - 1.0).min())
    drag_bp = (0.001 / (v[0] + 0.001)) / (4 / 252) * 1e4

    for name, want in (("annual_return", cagr), ("volatility", vol),
                       ("sharpe", sharpe), ("sortino", sortino),
                       ("max_drawdown", mdd), ("calmar", cagr / abs(mdd)),
                       ("turnover_annual", 0.5 / (4 / 252)),
                       ("costs_total", 0.001),
                       ("costs_annual_bp", drag_bp),
                       ("first_trade_cost", 0.001),
                       ("first_trade_turnover", 0.5),
                       ("max_asset_weight", 0.6)):
        got = m[name]
        if abs(got - want) > 1e-10:
            bad.append(f"metrics: {name} = {got:.10f}, longhand says "
                       f"{want:.10f}")

    # A delayed first trade must be reported at ITS date, not session 0.
    delayed = {**result,
               "cost": pd.Series([0, 0, 0.002, 0], index=days),
               "turnover": pd.Series([0, 0, 0.8, 0], index=days)}
    md = metrics.summary(delayed, rf)
    if md["first_trade_cost"] != 0.002 or md["first_trade_turnover"] != 0.8:
        bad.append("metrics: delayed first trade misreported "
                   f"({md['first_trade_cost']}, {md['first_trade_turnover']})")

    # start_value verification must fail loudly on a sliced sub-window.
    sliced = {k: (val.iloc[1:] if isinstance(val, (pd.Series, pd.DataFrame))
                  else val) for k, val in result.items()}
    try:
        metrics.summary(sliced, rf, start_value=1.0)
        bad.append("metrics: a sliced sub-window passed as a full run")
    except ValueError:
        pass

    # Per-year: five sessions across a year boundary, both years partial.
    days2 = pd.DatetimeIndex(["2005-12-28", "2005-12-29", "2005-12-30",
                              "2006-01-03", "2006-01-04"])
    r2 = pd.Series([0.01, -0.005, 0.002, 0.020, -0.010], index=days2)
    res2 = {"ret": r2}
    py = metrics.per_year(res2)
    want05 = (1.01 * 0.995 * 1.002) - 1.0
    want06 = (1.02 * 0.99) - 1.0
    if abs(py.loc[2005, "return"] - want05) > 1e-12 or \
       abs(py.loc[2006, "return"] - want06) > 1e-12:
        bad.append("metrics: per-year compounding wrong at a year boundary")
    if "(partial)" not in py.loc[2005, "label"] or \
       "(partial)" not in py.loc[2006, "label"]:
        bad.append("metrics: partial years not labelled")
    return bad


def check_baselines() -> list:
    """Real-data family: producers honour their stated conventions,
    and the min-variance solution satisfies its OWN optimality conditions —
    verified by mathematics (KKT), not by trusting the solver."""
    bad = []
    from portfolio import baselines
    from portfolio.run import load_bundle

    bundle = load_bundle()
    ret, tradeable = bundle["ret"], bundle["tradeable"]
    cols = bundle["columns"]
    inv = cols[:-1]

    # First train rebalance, a mid-sample date, and a short-history date
    # (2013-01-02 is ABBV's first session: tradeable, zero history).
    dates = ret.index
    sample = pd.DatetimeIndex([d for d in
                               [dates[(dates >= "2005-01-01")].min(),
                                pd.Timestamp("2010-06-01"),
                                pd.Timestamp("2013-01-02")]
                               if d in dates] +
                              [dates[dates <= pd.Timestamp("2013-01-02")].max()
                               ]).unique().sort_values()
    sample = pd.DatetimeIndex([d for d in sample if d in dates])

    for name in ("one_over_n", "markowitz", "min_variance", "risk_parity"):
        W, _ = baselines.ARMS[name]["producer"](bundle, sample)
        try:
            engine.validate_weights(W, cols)
        except (ValueError, TypeError) as e:
            bad.append(f"baselines: {name} frame fails the contract: {e}")
            continue
        for d in sample:
            live = tradeable.loc[d]
            row = W.loc[d]
            if name == "one_over_n":
                n = int(live.sum())
                held = row[row > 0]
                if len(held) != n or abs(held.iloc[0] - 1.0 / n) > 1e-12 \
                        or held.nunique() > 1:
                    bad.append(f"baselines: 1/N is not 1/{n} flat at "
                               f"{d.date()}")
            else:
                elig = set(baselines.eligible(ret, tradeable, d))
                short = [t for t in inv if live[t] and t not in elig
                         and row[t] != 0.0]
                if short:
                    bad.append(f"baselines: {name} gave weight to short-"
                               f"history {short[:3]} at {d.date()}")

    # KKT for long-only min-variance: with g = 2*sigma*w, every held asset
    # shares one gradient value and no excluded asset undercuts it.
    d = sample[-1]
    tickers = baselines.eligible(ret, tradeable, d)
    _, _, sigma = baselines.estimates(ret, tickers, d)
    w = baselines._solve_min_var(sigma)
    g = 2.0 * sigma @ w
    held = w > 1e-4
    lam = g[held].mean()
    # SLSQP's ftol leaves gradient spread near 5e-7 in daily-variance
    # units; a WRONG solution (e.g. 1/N passed off as min-var) spreads
    # ~2e-4 — two hundred times the floor.
    tol = max(0.02 * abs(lam), 1e-6)
    if g[held].max() - g[held].min() > 2 * tol:
        bad.append(f"baselines: min-var KKT fails — held-asset gradients "
                   f"spread {g[held].max() - g[held].min():.2e} at {d.date()}")
    if (g[~held] < lam - tol).any():
        bad.append(f"baselines: min-var KKT fails — an excluded asset "
                   f"undercuts the held gradient at {d.date()}")

    # Inverse-vol identity: weight x volatility constant across held assets.
    block, _, _ = baselines.estimates(ret, tickers, d)
    vol = block.std(ddof=1).to_numpy(dtype=float)
    W, _ = baselines.ARMS["risk_parity"]["producer"](bundle,
                                                     pd.DatetimeIndex([d]))
    wv = W.loc[d, tickers].to_numpy(dtype=float) * vol
    if (wv.max() - wv.min()) / wv.mean() > 1e-10:
        bad.append("baselines: risk parity breaks the w*vol identity")

    # The fixed-mix producer: valid frame, exact class shares, flat within
    # each sleeve over its tradeable members.
    shares = {"equities": 0.40, "bonds": 0.25, "commodities": 0.10,
              "reits": 0.15, "cash": 0.10}
    Wf, _ = baselines.fixed_mix(shares)(bundle, sample)
    try:
        engine.validate_weights(Wf, cols)
    except (ValueError, TypeError) as e:
        bad.append(f"baselines: fixed_mix frame fails the contract: {e}")
    for d2 in sample:
        row = Wf.loc[d2]
        if abs(row[engine.CASH] - 0.10) > 1e-12:
            bad.append(f"baselines: fixed_mix cash leg wrong at {d2.date()}")
        live = tradeable.loc[d2]
        for cls, share in shares.items():
            if cls == "cash":
                continue
            members = [t for t in bundle["classes"][cls] if live.get(t, False)]
            got = row[members]
            if abs(got.sum() - share) > 1e-12 or got.nunique() > 1:
                bad.append(f"baselines: fixed_mix {cls} sleeve not flat at "
                           f"{share} on {d2.date()}")
    return bad


FAMILIES = {
    "reference": check_reference,
    "contract": check_contract,
    "metrics": check_metrics,
    "baselines": check_baselines,
}

if __name__ == "__main__":
    from collectors._core import console_utf8
    console_utf8()
    args = sys.argv[1:]
    names = [n for n in FAMILIES if f"--{n}" in args] or \
            (list(FAMILIES) if "--all" in args or not args else [])
    violations = []
    for name in names:
        found = FAMILIES[name]()
        mark = "OK" if not found else f"{len(found)} VIOLATION(S)"
        print(f"{name:10s} {mark}")
        for f in found:
            print(f"    {f}")
        violations += found
    raise SystemExit(1 if violations else 0)
