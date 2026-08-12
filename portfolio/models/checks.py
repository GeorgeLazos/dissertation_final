"""
portfolio/models/checks.py — the environment proven with no agent in it.

Five families, each returning a list of violations; the CLI exits
non-zero if any family reports one.

    scaling     stats recompute from the train slice; blanks and
                degenerate columns handled as documented.
    interpret   malformed actions raise; masking and the cap projection
                match an independent reference.
    reward      dsr_step against the formula written a second time; an
                episode's reward reconciles with a longhand pass.
    episode     seeded resets reproduce; every floor and bound respected.
    wrap        a scripted 1/N through the environment equals the 1/N
                baseline through engine.run() bitwise, replay and band
                included.

    python -m portfolio.models.checks --all
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from config import portfolio as cfg
from config.splits import get_split
from portfolio import baselines, engine
from portfolio.models import scaling
from portfolio.models.environment import Environment, dsr_step, project_cap

_BUNDLE = None


# Load the shared data bundle once; every family reads the same objects.
def _bundle() -> dict:
    global _BUNDLE
    if _BUNDLE is None:
        from portfolio.run import load_bundle
        _BUNDLE = load_bundle()
    return _BUNDLE


# Fit a real ruler and verify it: train-only window, stats that recompute,
# blank/degenerate handling, and refusal of a wrong-assets file.
def check_scaling() -> list:
    bad = []
    from features import loader as fl
    bonds = _bundle()["classes"]["bonds"]
    feats = ["mom_63", "rv_21", "vix"]
    stats = scaling.fit("check_bonds", feats, bonds)

    start, end = get_split("train")
    if stats["window"] != [str(start), str(end)]:
        bad.append(f"scaling: fitted window {stats['window']} is not train")

    af = fl.features_asset("train", ["mom_63"])
    x = af[af["ticker"].isin(bonds)]["mom_63"].to_numpy(dtype=float)
    ok = x[~np.isnan(x)]
    want_mu, want_sd = float(np.mean(ok)), float(np.std(ok, ddof=1))
    got = stats["per_feature"]["mom_63"]
    if abs(got["mean"] - want_mu) > 1e-12 or abs(got["std"] - want_sd) > 1e-12:
        bad.append("scaling: mom_63 stats do not recompute from the train "
                   "slice")
    if got["count"] != len(ok):
        bad.append("scaling: row count disagrees with the train slice")

    mu, sd = scaling.vectors(stats, feats)
    raw = np.array([[want_mu, np.nan, 0.0]])
    s, f = scaling.apply(raw, mu, sd)
    if abs(s[0, 0]) > 1e-6:
        bad.append("scaling: the mean must scale to zero")
    if s[0, 1] != 0.0 or f[0, 1] != 1.0 or f[0, 0] != 0.0:
        bad.append("scaling: blank handling wrong — NaN must scale to 0 "
                   "with its flag set")

    if scaling._stats(np.array([3.0, 3.0, 3.0]))["std"] != 1.0:
        bad.append("scaling: a constant column must fall back to std 1")

    # A file fitted on other assets must be refused, not silently applied.
    try:
        Environment(_bundle()["classes"]["commodities"], feats,
                    scaling_name="check_bonds", bundle=_bundle())
        bad.append("scaling: a file fitted on different assets was accepted")
    except ValueError:
        pass
    return bad


# Action handling: masking zeroes dead assets, malformed actions raise,
# and the cap projection matches an independent reference.
def check_interpret() -> list:
    bad = []
    b = _bundle()
    env = Environment(b["classes"]["commodities"], ["ret_1"], cash=False,
                      window="train", bundle=b)
    # 2006-06-01: GLD, DBC, USO, SLV trade; DBA does not.
    env.i = int(env.dates.get_loc(env.dates[env.dates >= "2006-06-01"][0]))
    live = env.tradeable[env.i]
    if live.all() or not live.any():
        bad.append("interpret: chosen date does not split the sleeve")
    a = np.full(5, 0.2)
    t = env._interpret(a)
    if t[~live].sum() != 0.0:
        bad.append("interpret: weight survived on a non-tradeable asset")
    if abs(t.sum() - 1.0) > 1e-12:
        bad.append("interpret: renormalisation broke the sum")
    if abs(t[live].max() - 1.0 / live.sum()) > 1e-12:
        bad.append("interpret: surviving weights not evenly renormalised")

    for label, act in (("NaN", np.array([np.nan, .25, .25, .25, .25])),
                       ("negative", np.array([-.1, .4, .3, .2, .2])),
                       ("bad sum", np.full(5, 0.3)),
                       ("wrong width", np.full(4, 0.25))):
        try:
            env._interpret(act)
            bad.append(f"interpret: {label} action was accepted")
        except ValueError:
            pass

    # The projection against an independent reference: pin every breacher,
    # renormalise the rest, repeat — computed here with its own loop.
    rng = np.random.default_rng(7)
    for _ in range(200):
        k = int(rng.integers(4, 30))
        w = rng.dirichlet(np.ones(k) * 0.3)
        cap = float(rng.uniform(1.5 / k, 0.9))
        got = project_cap(w, cap)
        ref = w.copy()
        fixed = np.zeros(k, dtype=bool)
        while (ref[~fixed] > cap + 1e-15).any():
            hit = ~fixed & (ref > cap + 1e-15)
            ref[hit] = cap
            fixed |= hit
            free = ~fixed
            s = ref[free].sum()
            rem = 1.0 - cap * fixed.sum()
            ref[free] = rem / free.sum() if s <= 0 else ref[free] * rem / s
        if np.abs(got - ref).max() > 1e-12:
            bad.append("interpret: projection deviates from the reference")
            break
        if got.max() > cap + 1e-9 or abs(got.sum() - 1.0) > 1e-9:
            bad.append("interpret: projection breached the cap or the sum")
            break
    try:
        project_cap(np.full(4, 0.25), 0.2)
        bad.append("interpret: infeasible cap was accepted")
    except ValueError:
        pass

    # The cap binds live assets only — never redistributing onto masked
    # ones — and never binds CASH: the refuge stays reachable in full.
    env_c = Environment(b["classes"]["commodities"], ["ret_1"], cash=False,
                        window="train", bundle=b,
                        limits={"asset_cap": 0.5})
    env_c.i = int(env_c.dates.get_loc(
        env_c.dates[env_c.dates >= "2006-06-01"][0]))
    live_c = env_c.tradeable[env_c.i]
    t = env_c._interpret(np.full(5, 0.2))
    if t[~live_c].sum() != 0.0:
        bad.append("interpret: the cap pushed weight onto a masked asset")
    if t[live_c].max() > 0.5 + 1e-9 or abs(t.sum() - 1.0) > 1e-9:
        bad.append("interpret: capped projection broke the cap or the sum")

    env_s = Environment(b["classes"]["bonds"], ["ret_1"], cash=True,
                        window=("2010-01-01", "2010-12-31"), bundle=b,
                        limits={"asset_cap": 0.5})
    env_s.i = env_s.i_lo
    all_cash = np.zeros(7)
    all_cash[-1] = 1.0
    got = env_s._interpret(all_cash)
    if got[-1] != 1.0:
        bad.append("interpret: the cap bound CASH — full de-risking must "
                   "stay reachable")
    return bad


# The reward is right twice over: dsr_step vs the formula rewritten, and
# a whole episode reconciled against a longhand pass over its path.
def check_reward() -> list:
    bad = []
    # dsr_step against the formula written out a second time.
    rng = np.random.default_rng(3)
    A = B = 0.0
    eta = 0.01
    for r in rng.normal(0.0005, 0.01, 200):
        dA, dB = r - A, r * r - B
        var = B - A * A
        want = (B * dA - 0.5 * A * dB) / var ** 1.5 if var > 1e-12 else 0.0
        A2, B2, D = dsr_step(A, B, float(r), eta)
        if abs(D - want) > 1e-12 or abs(A2 - (A + eta * dA)) > 1e-15 \
                or abs(B2 - (B + eta * dB)) > 1e-15:
            bad.append("reward: dsr_step deviates from the written formula")
            break
        A, B = A2, B2

    # An episode's accumulated reward reconciles with a longhand pass over
    # the recorded path: same returns, same warm-up, same penalty.
    b = _bundle()
    env = Environment(b["classes"]["bonds"], ["mom_63"], cash=False,
                      window="train", bundle=b, seed=11,
                      episode_len=80, warmup=20, lam=1e-3)
    obs, info = env.reset()
    rng = np.random.default_rng(5)
    rewards, done = [], False
    while not done:
        raw = rng.dirichlet(np.ones(len(env.columns)))
        obs, r, done, info = env.step(raw)
        rewards.append(r)
    idx = pd.DatetimeIndex(sorted(env.path["value"]))
    rets = [env.path["ret"][d] for d in idx]
    tos = [env.path["turnover"][d] for d in idx]
    A = B = 0.0
    total = 0.0
    for n, (r, to) in enumerate(zip(rets, tos), start=1):
        A, B, D = dsr_step(A, B, r, env.eta)
        if n > env.warmup:
            total += D - env.lam * to
    if abs(sum(rewards) - total) > 1e-10:
        bad.append(f"reward: episode total {sum(rewards):.10f} does not "
                   f"reconcile with the longhand pass {total:.10f}")

    # The same reconciliation on a MONTHLY clock, where one step spans a
    # month of sessions and only landing days defer their pay.
    envm = Environment(b["classes"]["bonds"], ["mom_63"], cash=True,
                       clock="monthly", window=("2009-01-01", "2010-12-31"),
                       bundle=b, warmup=10, lam=1e-3)
    n_cols = len(envm.columns)
    rngm = np.random.default_rng(6)
    outm = envm.evaluate(lambda o, i: rngm.dirichlet(np.ones(n_cols)))
    # evaluate discards rewards, so re-walk the recorded path longhand and
    # against a fresh env stepping the same recorded decisions.
    env2 = Environment(b["classes"]["bonds"], ["mom_63"], cash=True,
                       clock="monthly", window=("2009-01-01", "2010-12-31"),
                       bundle=b, warmup=10, lam=1e-3)
    env2._reset_state(env2.i_lo, env2.i_hi)
    dec = outm["decisions"]
    rewards2, done = [], False
    obs2, info2 = env2._observe(), env2._info()
    k = 0
    while not done:
        obs2, r2, done, info2 = env2.step(dec.iloc[k].to_numpy())
        rewards2.append(r2)
        k += 1
    idx = outm["ret"].index
    A = B = 0.0
    total = 0.0
    n = 0
    for d in idx:
        r = outm["ret"].loc[d]
        A, B, D = dsr_step(A, B, float(r), env2.eta)
        n += 1
        if n > env2.warmup:
            total += D - env2.lam * float(outm["turnover"].loc[d])
    if abs(sum(rewards2) - total) > 1e-10:
        bad.append(f"reward: monthly episode total {sum(rewards2):.10f} does "
                   f"not reconcile with the longhand pass {total:.10f}")
    return bad


# Episode placement: seeded reproducibility, every floor and bound held,
# degenerate windows refused, no singleton commodity starts.
def check_episode() -> list:
    bad = []
    b = _bundle()
    mk = lambda seed: Environment(b["classes"]["bonds"], ["mom_63"],
                                  window="train", bundle=b, seed=seed)
    e1, e2, e3 = mk(4), mk(4), mk(5)
    starts1 = [e1.reset() is not None and e1.i for _ in range(12)]
    starts2 = [e2.reset() is not None and e2.i for _ in range(12)]
    starts3 = [e3.reset() is not None and e3.i for _ in range(12)]
    if starts1 != starts2:
        bad.append("episode: identical seeds drew different windows")
    if starts1 == starts3:
        bad.append("episode: different seeds drew identical windows")
    floor = e1.dates.searchsorted(pd.Timestamp(cfg.AGENT_TRAIN_START))
    lo, hi = min(starts1 + starts3), max(starts1 + starts3)
    if lo < floor:
        bad.append("episode: a window began before AGENT_TRAIN_START")
    if hi + e1.episode_len > e1.i_hi:
        bad.append("episode: a window overran the span")

    # A window that misses the calendar (or runs backwards) must raise —
    # the silent alternative spans the whole panel, test years included.
    for label, win in (("empty window", ("2030-01-01", "2030-12-31")),
                       ("inverted window", ("2010-12-31", "2010-01-01"))):
        try:
            Environment(b["classes"]["bonds"], ["mom_63"], window=win,
                        bundle=b)
            bad.append(f"episode: {label} was accepted")
        except ValueError:
            pass

    # A sleeve with one live member offers no decision: commodity episodes
    # must never start while GLD trades alone.
    ec = Environment(b["classes"]["commodities"], ["ret_1"], window="train",
                     bundle=b, seed=8)
    first_multi = int(np.nonzero(ec.tradeable.sum(axis=1) >= 2)[0][0])
    draws = []
    for _ in range(40):
        ec.reset()
        draws.append(ec.i)
    if min(draws) < first_multi:
        bad.append("episode: a commodity episode started on a "
                   "single-asset session")

    # A finished episode must refuse further steps — a silent re-trade on
    # the terminal session would corrupt the record.
    eg = Environment(b["classes"]["bonds"], ["mom_63"], window="train",
                     bundle=b, episode_len=30, warmup=5)
    obs_g, info_g = eg.reset()
    done_g = False
    while not done_g:
        obs_g, _, done_g, info_g = eg.step(np.full(6, 1.0 / 6))
    if not info_g["truncated"]:
        bad.append("episode: the done step must report truncated=True")
    try:
        eg.step(np.full(6, 1.0 / 6))
        bad.append("episode: a step after done was accepted")
    except RuntimeError:
        pass
    return bad


# The gold family: the environment IS the engine — sleeve stepping,
# scripted 1/N vs the baseline, and banded replay, all bitwise.
def check_wrap() -> list:
    bad = []
    b = _bundle()

    # 1. Decomposed stepping vs the engine's combined call, bitwise, on a
    #    cash-less sleeve driven by a seeded random policy.
    env = Environment(b["classes"]["bonds"], ["mom_63"], cash=False,
                      window=("2010-01-01", "2011-06-30"), bundle=b,
                      band=0.004, seed=2)
    rng = np.random.default_rng(9)
    acts = {}
    policy = lambda obs, info: acts.setdefault(
        str(info["date"]), rng.dirichlet(np.ones(len(env.columns))))
    out = env.evaluate(policy)

    cols = env.columns
    rates = env.rates
    dates = out["value"].index
    dec = out["decisions"]
    hold = None
    value = 1.0
    R = b["ret"][cols].reindex(dates).to_numpy(dtype=float)
    for n, d in enumerate(dates):
        target = dec.loc[d].to_numpy() if d in dec.index else None
        if hold is None:
            cost = float(np.abs(target) @ rates) * value
            hold = target * (value - cost)
            value -= cost
        else:
            rets = engine.police_rets(hold, R[n], cols, d, first=False)
            hold, rec = engine.step(hold, rets, target, rates, env.band)
            value = rec["value"]
        if abs(value - out["value"].loc[d]) > 0.0:
            bad.append(f"wrap: sleeve value deviates at {d.date()} by "
                       f"{abs(value - out['value'].loc[d]):.2e}")
            break

    # 2. Scripted 1/N through the full-universe environment == the 1/N
    #    baseline through engine.run, bitwise, decisions included.
    inv = [t for members in b["classes"].values() for t in members]
    env2 = Environment(inv, ["ret_1"], clock="monthly", cash=True,
                       window="train", bundle=b, band=0.0)
    n_assets = len(inv)

    def one_over_n_policy(obs, info):
        live = info["tradeable"]
        a = np.zeros(n_assets + 1)
        a[:n_assets][live] = 1.0 / live.sum()
        return a

    out2 = env2.evaluate(one_over_n_policy)

    start, end = get_split("train")
    window = b["dates"][(b["dates"] >= start) & (b["dates"] <= end)]
    reb = cfg.month_starts(window)
    W, _ = baselines.one_over_n(b, reb)
    ret_eval = b["ret"][inv].loc[window].copy()
    ret_eval[engine.CASH] = b["cash"].reindex(window).values
    ref = engine.run(W, ret_eval, cfg.cost_rates(inv + [engine.CASH]))

    for k in ("value", "cost", "turnover"):
        dv = float(np.max(np.abs(out2[k].values - ref[k].values)))
        if dv > 0.0:
            bad.append(f"wrap: scripted 1/N deviates from the baseline on "
                       f"{k} by {dv:.2e}")
    if not out2["decisions"].equals(W):
        bad.append("wrap: the recorded decision frame differs from the "
                   "baseline's weights frame")

    # 3. The recorded frame replays through engine.run to the identical
    #    path — including with a non-zero band.
    env3 = Environment(inv, ["ret_1"], clock="monthly", cash=True,
                       window=("2007-01-01", "2009-12-31"), bundle=b,
                       band=0.01)
    out3 = env3.evaluate(one_over_n_policy)
    win3 = b["dates"][(b["dates"] >= "2007-01-01") & (b["dates"] <= "2009-12-31")]
    ret3 = b["ret"][inv].loc[win3].copy()
    ret3[engine.CASH] = b["cash"].reindex(win3).values
    ref3 = engine.run(out3["decisions"], ret3,
                      cfg.cost_rates(inv + [engine.CASH]), band=0.01)
    for k in ("value", "cost", "turnover"):
        dv = float(np.max(np.abs(out3[k].values - ref3[k].values)))
        if dv > 0.0:
            bad.append(f"wrap: banded replay deviates on {k} by {dv:.2e}")
    return bad


FAMILIES = {
    "scaling": check_scaling,
    "interpret": check_interpret,
    "reward": check_reward,
    "episode": check_episode,
    "wrap": check_wrap,
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
