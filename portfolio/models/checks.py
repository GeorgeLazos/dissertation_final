"""
portfolio/models/checks.py — the environment proven with no agent in it.

Six families, each returning a list of violations; the CLI exits
non-zero if any family reports one.

    scaling     stats recompute from the train slice; blanks and
                degenerate columns handled as documented; windows past
                the train boundary refused at fit AND load time.
    interpret   malformed actions raise; masking and the cap projection
                match an independent reference.
    reward      dsr_step against the formula written a second time; an
                episode's reward reconciles with a longhand pass.
    episode     seeded resets reproduce; every floor and bound respected.
    wrap        a scripted 1/N through the environment equals the 1/N
                baseline through engine.run() bitwise, replay and band
                included.
    trainer     GAE longhand, agent configs valid, masked softmax exact,
                seeded determinism, checkpoint round-trip.

    python -m portfolio.models.checks --all
"""
from __future__ import annotations
import json
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


# Fit a real scaling file and verify it: train-only window, stats that
# recompute, blank/degenerate handling, and refusal of a wrong-assets file.
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

    # A window reaching past the train boundary must be refused at fit
    # time — fitting through validation leaks its distribution.
    try:
        scaling.fit("check_leak", feats, bonds,
                    window=("2005-01-01", "2021-12-31"))
        bad.append("scaling: a window past the train boundary was accepted")
    except ValueError:
        pass

    # ... and a stats FILE claiming such a window must be refused at load
    # time, whatever wrote it.
    forged = dict(stats)
    forged["window"] = ["2005-01-01", "2021-12-31"]
    fp = scaling.OUT_DIR / "check_forged.json"
    fp.write_text(json.dumps(forged), encoding="utf-8")
    try:
        Environment(bonds, feats, scaling_name="check_forged",
                    bundle=_bundle())
        bad.append("scaling: a file fitted outside train was accepted")
    except ValueError:
        pass
    finally:
        fp.unlink()
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
    # ones — and never binds CASH: full de-risking stays reachable.
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
        want = (B * dA - 0.5 * A * dB) / var ** 1.5 if var > 1e-8 else 0.0
        A2, B2, D = dsr_step(A, B, float(r), eta)
        if abs(D - want) > 1e-12 or abs(A2 - (A + eta * dA)) > 1e-15 \
                or abs(B2 - (B + eta * dB)) > 1e-15:
            bad.append("reward: dsr_step deviates from the written formula")
            break
        A, B = A2, B2

    # The variance floor: an all-cash sit returns the T-bill accrual, so
    # variance parks near 1e-11 — far past any warm-up — and the re-entry
    # trade's cost return divided by var**1.5 would be a reward in the
    # thousands. Every D on that path, the cost shock included, must be
    # exactly zero.
    A = B = 0.0
    worst = 0.0
    for r in [4.4e-6] * 80 + [-0.006]:
        A, B, D = dsr_step(A, B, r, 0.01)
        worst = max(worst, abs(D))
    if worst != 0.0:
        bad.append(f"reward: a sub-floor variance path emitted a reward "
                   f"(|D| up to {worst:.3g})")

    # ... and the floor must not swallow real trading: risky-scale returns
    # clear it within a couple of sessions.
    A = B = 0.0
    n_zero = 0
    for r in np.random.default_rng(12).normal(0.0005, 0.01, 100):
        A, B, D = dsr_step(A, B, float(r), 0.01)
        n_zero += D == 0.0
    if n_zero > 5:
        bad.append(f"reward: the variance floor suppressed {n_zero}/100 "
                   f"risky-scale rewards")

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

    # The same reconciliation on a monthly CASH-LESS walk — the one path
    # that combines the buy-in step, month-long spans and the terminal
    # landing in a single episode.
    envn = Environment(b["classes"]["bonds"], ["mom_63"], cash=False,
                       clock="monthly", window=("2011-01-01", "2012-12-31"),
                       bundle=b, warmup=10, lam=1e-3)
    n_cols_n = len(envn.columns)
    rngn = np.random.default_rng(14)
    outn = envn.evaluate(lambda o, i: rngn.dirichlet(np.ones(n_cols_n)))
    env3 = Environment(b["classes"]["bonds"], ["mom_63"], cash=False,
                       clock="monthly", window=("2011-01-01", "2012-12-31"),
                       bundle=b, warmup=10, lam=1e-3)
    env3._reset_state(env3.i_lo, env3.i_hi)
    dec_n = outn["decisions"]
    rewards3, done = [], False
    k = 0
    while not done:
        _, r3, done, _ = env3.step(dec_n.iloc[k].to_numpy())
        rewards3.append(r3)
        k += 1
    A = B = 0.0
    total = 0.0
    n = 0
    for d in outn["ret"].index:
        A, B, D = dsr_step(A, B, float(outn["ret"].loc[d]), env3.eta)
        n += 1
        if n > env3.warmup:
            total += D - env3.lam * float(outn["turnover"].loc[d])
    if abs(sum(rewards3) - total) > 1e-10:
        bad.append(f"reward: monthly cash-less total {sum(rewards3):.10f} "
                   f"does not reconcile with the longhand pass {total:.10f}")

    # The same reconciliation on a WEEKLY cash-less walk.
    envw = Environment(b["classes"]["bonds"], ["mom_63"], cash=False,
                       clock="weekly", window=("2011-01-01", "2011-12-31"),
                       bundle=b, warmup=10, lam=1e-3)
    n_cols_w = len(envw.columns)
    rngw = np.random.default_rng(15)
    outw = envw.evaluate(lambda o, i: rngw.dirichlet(np.ones(n_cols_w)))
    env4 = Environment(b["classes"]["bonds"], ["mom_63"], cash=False,
                       clock="weekly", window=("2011-01-01", "2011-12-31"),
                       bundle=b, warmup=10, lam=1e-3)
    env4._reset_state(env4.i_lo, env4.i_hi)
    dec_w = outw["decisions"]
    rewards4, done = [], False
    k = 0
    while not done:
        _, r4, done, _ = env4.step(dec_w.iloc[k].to_numpy())
        rewards4.append(r4)
        k += 1
    A = B = 0.0
    total = 0.0
    n = 0
    for d in outw["ret"].index:
        A, B, D = dsr_step(A, B, float(outw["ret"].loc[d]), env4.eta)
        n += 1
        if n > env4.warmup:
            total += D - env4.lam * float(outw["turnover"].loc[d])
    if abs(sum(rewards4) - total) > 1e-10:
        bad.append(f"reward: weekly cash-less total {sum(rewards4):.10f} "
                   f"does not reconcile with the longhand pass {total:.10f}")
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


# The environment IS the engine — sleeve stepping, scripted 1/N vs the
# baseline, and banded replay, all bitwise.
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

    # 4. The weekly grid: every full calendar year holds exactly its ISO
    #    week count of starts, and gaps never exceed a holiday-stretched
    #    week — a self-consistent but wrong grid cannot pass.
    ws = cfg.week_starts(b["dates"])
    yrs = pd.Series(ws.year)
    counts = yrs.value_counts()
    full = counts.drop([int(yrs.min()), int(yrs.max())], errors="ignore")
    if not full.between(52, 53).all():
        bad.append(f"wrap: weekly starts per full year outside 52-53 "
                   f"(min {int(full.min())}, max {int(full.max())})")
    gaps = np.diff(ws.values).astype("timedelta64[D]").astype(int)
    if gaps.min() < 1 or gaps.max() > 12:
        bad.append(f"wrap: weekly start gaps outside 1-12 days "
                   f"(min {gaps.min()}, max {gaps.max()})")

    # 5. Scripted 1/N on the WEEKLY clock == the 1/N baseline through
    #    engine.run on week-start rebalances, bitwise.
    env5 = Environment(inv, ["ret_1"], clock="weekly", cash=True,
                       window=("2010-01-01", "2011-12-31"), bundle=b,
                       band=0.0)
    out5 = env5.evaluate(one_over_n_policy)
    win5 = b["dates"][(b["dates"] >= "2010-01-01") & (b["dates"] <= "2011-12-31")]
    reb5 = cfg.week_starts(win5)
    W5, _ = baselines.one_over_n(b, reb5)
    ret5 = b["ret"][inv].loc[win5].copy()
    ret5[engine.CASH] = b["cash"].reindex(win5).values
    ref5 = engine.run(W5, ret5, cfg.cost_rates(inv + [engine.CASH]))
    for k in ("value", "cost", "turnover"):
        dv = float(np.max(np.abs(out5[k].values - ref5[k].values)))
        if dv > 0.0:
            bad.append(f"wrap: weekly scripted 1/N deviates from the "
                       f"baseline on {k} by {dv:.2e}")
    if not out5["decisions"].equals(W5):
        bad.append("wrap: the weekly decision frame differs from the "
                   "baseline's weights frame")
    return bad


# Trainer plumbing: seeded runs identical, checkpoints round-trip, the
# masked softmax zeroes dead columns exactly, GAE matches a longhand pass
# and bootstraps the truncated tail.
def check_trainer() -> list:
    bad = []
    import torch
    from portfolio.models import train as tr
    from portfolio.models.networks import PolicyValue, masked_weights

    # GAE against a longhand recursion, bootstrap included.
    rewards = np.array([1.0, -0.5, 2.0])
    values = np.array([0.5, 0.4, 0.3])
    last_v = 0.7
    adv, ret = tr.gae(rewards, values, last_v, gamma=0.9, lam=0.8)
    want = []
    nxt_adv = 0.0
    nxt_v = last_v
    for t in (2, 1, 0):
        delta = rewards[t] + 0.9 * nxt_v - values[t]
        nxt_adv = delta + 0.9 * 0.8 * nxt_adv
        want.append(nxt_adv)
        nxt_v = values[t]
    want = np.array(want[::-1])
    if np.abs(adv - want).max() > 1e-12:
        bad.append("trainer: GAE deviates from the longhand recursion")
    if abs(ret[-1] - (values[-1] + adv[-1])) > 1e-12:
        bad.append("trainer: returns are not values plus advantages")
    zero_adv, _ = tr.gae(rewards, values, 0.0, gamma=0.9, lam=0.8)
    if abs(adv[-1] - zero_adv[-1] - 0.9 * last_v) > 1e-12:
        bad.append("trainer: the truncated tail does not bootstrap last_v")

    # Every agent file loads, declares its own sleeve, and carries exactly
    # the legal keys; universe-scoped agents resolve their asset lists.
    from config.tickers import all_classes
    for sleeve in (*all_classes(), "flat", "allocator"):
        try:
            acfg_s = tr.agent_config(sleeve)
            n_assets = len(tr.agent_assets(acfg_s))
            if sleeve == "flat" and n_assets != 119:
                bad.append(f"trainer: flat resolves {n_assets} assets, "
                           f"not 119")
        except Exception as e:
            bad.append(f"trainer: config/agents/{sleeve}.py invalid: {e}")

    # The config guard itself: a stray uppercase name (a typo'd section)
    # and an empty FEATURES list must both be refused — accepted, either
    # silently changes the observation set.
    import types
    probe = types.ModuleType("config.agents.zz_probe")
    probe.SLEEVE = "zz_probe"
    probe.NETWORK = {"hidden": 8, "layers": 2, "dropout": 0.0}
    probe.PPO = {"lr": 1e-4, "clip": 0.2, "gamma": 0.99, "gae_lambda": 0.95,
                 "epochs": 1, "minibatches": 1, "entropy_coef": 0.0,
                 "value_coef": 0.5, "grad_clip": 0.5, "weight_decay": 0.0}
    probe.ENV = {"clock": "daily", "cash": False, "band": 0.0, "eta": 0.01,
                 "lam": 0.0, "warmup": 0, "episode_len": 10}
    probe.TRAIN = {"updates": 1, "episodes_per_update": 1, "eval_every": 1,
                   "seed": 0}
    probe.FEATURES = None
    probe.FEATURE = ["mom_63"]
    sys.modules["config.agents.zz_probe"] = probe
    try:
        try:
            tr.agent_config("zz_probe")
            bad.append("trainer: a stray uppercase name in an agent file "
                       "was accepted")
        except ValueError:
            pass
        del probe.FEATURE
        probe.FEATURES = []
        try:
            tr.agent_config("zz_probe")
            bad.append("trainer: FEATURES = [] was accepted")
        except ValueError:
            pass
    finally:
        del sys.modules["config.agents.zz_probe"]

    # Every sleeve's observation list must carry asset features — a sleeve
    # seeing only market state cannot distinguish its own assets.
    from config import feature_registry as registry
    for sleeve in all_classes():
        feats = tr.feature_list(sleeve)
        n_asset = sum(1 for n in feats
                      if registry.spec(n)["grain"] == "asset")
        if n_asset == 0:
            bad.append(f"trainer: {sleeve} feature list has no asset "
                       f"features")

    # Masked softmax: dead columns exactly zero, live sum exactly-ish 1.
    z = torch.tensor([0.3, -0.2, 1.0, 0.0])
    mask = np.array([True, False, True, True])
    w = masked_weights(z, mask)
    if w[1] != 0.0 or abs(w.sum() - 1.0) > 1e-9:
        bad.append("trainer: masked softmax leaks weight or breaks the sum")

    # Static covariates: the sector block is 8 exact one-hots covering
    # every equity; the environment appends it between the flags and the
    # weights unchanged, grows obs_size by exactly its size, and refuses
    # a block whose rows do not match the assets.
    eq = _bundle()["classes"]["equities"]
    ss = tr.sector_static(eq)
    if ss.shape != (len(eq), 8):
        bad.append(f"trainer: sector block shape {ss.shape}, want "
                   f"({len(eq)}, 8)")
    if not (np.isin(ss, (0.0, 1.0)).all()
            and np.array_equal(ss.sum(axis=1),
                               np.ones(len(eq), dtype=np.float32))):
        bad.append("trainer: sector rows must be exact one-hots")
    bnd = _bundle()["classes"]["bonds"]
    st = np.eye(len(bnd), 3, dtype=np.float32)
    env_s = Environment(bnd, ["mom_63"], window="train", bundle=_bundle(),
                        static=st)
    env_p = Environment(bnd, ["mom_63"], window="train", bundle=_bundle())
    if env_s.obs_size != env_p.obs_size + st.size:
        bad.append("trainer: static block does not grow obs_size by its "
                   "own size")
    obs_s, _ = env_s.reset()
    k = len(env_s.columns)
    seg = obs_s[-(st.size + k + 1):-(k + 1)]
    if not np.array_equal(seg, st.ravel()):
        bad.append("trainer: static block not found intact in the "
                   "observation")
    try:
        Environment(bnd, ["mom_63"], window="train", bundle=_bundle(),
                    static=np.zeros((3, 2), dtype=np.float32))
        bad.append("trainer: a misaligned static block was accepted")
    except ValueError:
        pass

    # Dead action columns carry no probability weight: changing a masked
    # column's z must not move the log-probability, and the policy outputs
    # feeding that column must receive exactly zero gradient.
    from portfolio.models import networks as nw
    tr.seed_everything(13)
    netz = PolicyValue(6, 3, hidden=8)
    obs_z = torch.randn(4, 6)
    mu_z, _ = netz(obs_z)
    d_z = netz.dist(mu_z)
    z_z = d_z.sample()
    m_z = torch.tensor([[1.0, 1.0, 0.0]] * 4)
    lp = nw.masked_logp(d_z, z_z, m_z)
    z_moved = z_z.clone()
    z_moved[:, 2] += 7.0
    if not torch.equal(lp, nw.masked_logp(d_z, z_moved, m_z)):
        bad.append("trainer: a dead column's z moved the log-probability")
    (-lp.sum()).backward()
    out_layer = netz.pi[-1]
    if float(out_layer.weight.grad[2].abs().max()) != 0.0 \
            or float(out_layer.bias.grad[2].abs()) != 0.0 \
            or float(netz.log_std.grad[2].abs()) != 0.0:
        bad.append("trainer: a dead column received policy gradient")

    # Weight decay must reach weight matrices only. Decaying log_std drags
    # exploration toward exp(0) = 1, and decayed biases shift outputs; a
    # decay step with zero gradients must move every weight and nothing
    # else.
    tr.seed_everything(9)
    netw = PolicyValue(5, 3, hidden=8)
    optw = tr.make_optimizer(netw, lr=0.1, weight_decay=0.5)
    (sum(p.sum() for p in netw.parameters()) * 0.0).backward()
    before = {n: p.detach().clone() for n, p in netw.named_parameters()}
    optw.step()
    for n, p in netw.named_parameters():
        changed = not torch.equal(before[n], p.detach())
        if n.endswith("weight") and not changed:
            bad.append(f"trainer: weight decay never reached {n}")
        if not n.endswith("weight") and changed:
            bad.append(f"trainer: weight decay leaked into {n}")

    # Mode discipline: deterministic evaluation must not consume RNG or
    # vary (dropout live), and update() must hand the net back in eval
    # mode — train mode exists only inside its gradient epochs.
    tr.seed_everything(11)
    netd = PolicyValue(6, 3, hidden=8, dropout=0.5)
    pol = tr.det_policy(netd, 3)
    info_d = {"tradeable": np.ones(3, dtype=bool), "cash": False}
    obs_d = np.zeros(6, dtype=np.float32)
    state = torch.get_rng_state()
    a1 = pol(obs_d, info_d)
    a2 = pol(obs_d, info_d)
    if not torch.equal(state, torch.get_rng_state()):
        bad.append("trainer: deterministic evaluation consumed torch RNG")
    if not np.array_equal(a1, a2):
        bad.append("trainer: deterministic evaluation varies call to call")
    roll_d = {"obs": np.zeros((8, 6), dtype=np.float32),
              "mask": np.ones((8, 3), dtype=bool),
              "z": np.zeros((8, 3), dtype=np.float32),
              "logp": np.zeros(8, dtype=np.float32),
              "adv": np.zeros(8, dtype=np.float32),
              "ret": np.zeros(8, dtype=np.float32)}
    h_d = {"epochs": 1, "minibatches": 2, "clip": 0.2, "value_coef": 0.5,
           "entropy_coef": 0.0, "grad_clip": 0.5}
    tr.update(netd, tr.make_optimizer(netd, 1e-3, 0.0), roll_d, h_d)
    if netd.training:
        bad.append("trainer: update() left the net in train mode")

    # Seeded determinism: two tiny runs, identical parameters after.
    def tiny_run():
        tr.seed_everything(7)
        net = PolicyValue(10, 4, hidden=8)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        rng = np.random.default_rng(7)
        for _ in range(3):
            obs = torch.as_tensor(rng.normal(size=(16, 10)),
                                  dtype=torch.float32)
            mu, v = net(obs)
            loss = mu.pow(2).mean() + v.pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return torch.cat([p.detach().ravel() for p in net.parameters()])
    if not torch.equal(tiny_run(), tiny_run()):
        bad.append("trainer: identical seeds diverge")

    # Checkpoint round-trip: identical deterministic output. Scratch lives
    # in a temp dir so a failing save/load leaves nothing behind.
    import tempfile
    tr.seed_everything(3)
    net = PolicyValue(12, 5, hidden=8)
    opt = torch.optim.Adam(net.parameters())
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "check_roundtrip.pt"
        tr.save(p, net, opt, {"obs_size": 12, "action_size": 5,
                              "network": {"hidden": 8, "layers": 2,
                                          "dropout": 0.0}})
        net2, _ = tr.load(p)
    obs = torch.as_tensor(np.random.default_rng(1).normal(size=12),
                          dtype=torch.float32)
    with torch.no_grad():
        a, _ = net(obs)
        b, _ = net2(obs)
    if not torch.equal(a, b):
        bad.append("trainer: checkpoint round-trip changes the policy")
    return bad


# The anchored-tilt identity: a zero policy head must reproduce the
# anchor exactly on every decision, or the tilt's never-worse-than-
# anchor floor does not exist. Skips silently if the anchor artifact
# is not built.
def check_anchor() -> list:
    import torch
    from portfolio.models import anchors, train as tr
    from portfolio.models.environment import Environment
    from portfolio.models.networks import PolicyValue
    bad = []
    acfg = tr.agent_config("tilt")
    if not anchors.path(acfg.ANCHOR).exists():
        print("    anchor artifact not built — family skipped")
        return bad
    from portfolio.run import load_bundle
    b = load_bundle()
    e = acfg.ENV
    env = Environment(tr.agent_assets(acfg), tr.sleeve_features(acfg),
                      clock=e["clock"], cash=e["cash"],
                      scaling_name=None, window="val", bundle=b,
                      band=e["band"], eta=e["eta"], lam=e["lam"],
                      warmup=e["warmup"], episode_len=e["episode_len"])
    import numpy as _np
    env.mu_a = _np.zeros(len(env.a_names), dtype=_np.float32)
    env.sd_a = _np.ones(len(env.a_names), dtype=_np.float32)
    env.mu_m = _np.zeros(len(env.m_names), dtype=_np.float32)
    env.sd_m = _np.ones(len(env.m_names), dtype=_np.float32)
    A = anchors.load(acfg.ANCHOR, env.columns)
    net = PolicyValue(env.obs_size, env.action_size,
                      acfg.NETWORK["hidden"], acfg.NETWORK["layers"],
                      acfg.NETWORK["dropout"])
    with torch.no_grad():
        net.pi[-1].weight.zero_()
        net.pi[-1].bias.zero_()
    dev = []

    def probe(obs, info):
        w = tr.tilt(tr.det_policy(net, env.action_size)(obs, info),
                    info, A, acfg.TAU)
        i = A.index.searchsorted(info["date"], side="right") - 1
        anchor = A.iloc[i].to_numpy(dtype=float)
        live = _np.ones(len(w), dtype=bool)
        live[: len(info["tradeable"])] = info["tradeable"]
        ref = _np.where(live, anchor, 0.0)
        s = ref.sum()
        ref = ref / s if s > 0 else anchor
        dev.append(float(_np.abs(w - ref).max()))
        return w

    env.evaluate(probe, "val")
    worst = max(dev) if dev else float("nan")
    if not dev or worst > 1e-10:
        bad.append(f"anchor: zero-tilt deviates from the anchor by "
                   f"{worst:.2e} (must be < 1e-10)")
    return bad


FAMILIES = {
    "scaling": check_scaling,
    "interpret": check_interpret,
    "reward": check_reward,
    "episode": check_episode,
    "wrap": check_wrap,
    "trainer": check_trainer,
    "anchor": check_anchor,
}

if __name__ == "__main__":
    from collectors._core import console_utf8
    console_utf8()
    args = sys.argv[1:]
    known = {f"--{n}" for n in FAMILIES} | {"--all"}
    stray = [a for a in args if a not in known]
    if stray:
        raise SystemExit(f"unknown arguments {stray} — use "
                         f"{' '.join(sorted(known))}")
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
