"""
portfolio/models/allocate.py — the allocation agent's world, its
verification, and its training run.

The four deployed sleeve agents are FROZEN: each best.pt is evaluated
once, deterministically, over 2005–2021, and its daily portfolio return
becomes one synthetic asset. The allocator trades those four synthetic
assets plus CASH on the monthly clock, observing the market feature
block (synthetic assets carry no asset-grain features). Costs charge
class-level rates on share changes — an approximation of the underlying
rebalancing, declared as such; sleeve-internal costs are already inside
the return series. The test split is never evaluated here.

    python -m portfolio.models.allocate --build    # sleeve returns cache
    python -m portfolio.models.allocate --check    # reconciliation + wrap
    python -m portfolio.models.allocate --promote  # the training run
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from config import portfolio as cfg
from config.splits import get_split
from config.tickers import all_classes
from portfolio import engine
from portfolio.models import train as tr
from portfolio.models.environment import Environment

CLASSES = tuple(all_classes())
_ACFG = tr.agent_config("allocator")
DEPLOYED = dict(_ACFG.DEPLOYED)
DEPLOYED_ALLOCATOR = _ACFG.DEPLOYED_RUN

# Sleeve returns span train + validation; the test split stays untouched.
SPAN = (cfg.AGENT_TRAIN_START, get_split("val")[1])

OUT = tr.AGENT_RUNS / "allocator"
CACHE = OUT / "sleeve_returns.parquet"
CACHE_META = OUT / "sleeve_returns.json"

# The allocator's observation: the config's market-block FEATURES.
MARKET = tr.sleeve_features(_ACFG)

# One frozen model, evaluated over the span through its own deployed
# environment; returns the daily portfolio return series.
def _sleeve_returns(sleeve: str, bundle: dict) -> pd.Series:
    run_dir = tr.AGENT_RUNS / sleeve / DEPLOYED[sleeve]
    net, meta = tr.load(run_dir / "best.pt")
    assets = all_classes()[sleeve]
    env = Environment(assets, meta["features"], clock=meta["env"]["clock"],
                      cash=False, scaling_name=meta["scaling"],
                      window=SPAN, bundle=bundle,
                      band=meta["env"]["band"], eta=meta["env"]["eta"],
                      lam=meta["env"]["lam"], warmup=meta["env"]["warmup"],
                      episode_len=meta["env"]["episode_len"])
    out = env.evaluate(tr.det_policy(net, env.action_size), SPAN)
    return out["ret"].rename(sleeve)

# Caches all four models return series plus the checkpoints and span they came from.
def build() -> None:
    from portfolio.run import load_bundle
    bundle = load_bundle()
    cols = {}
    for s in CLASSES:
        print(f"evaluating frozen {s} ({DEPLOYED[s]})", flush=True)
        cols[s] = _sleeve_returns(s, bundle)
    df = pd.DataFrame(cols)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    CACHE_META.write_text(json.dumps({"deployed": DEPLOYED, "span": SPAN}, indent=2), encoding="utf-8")
    print(f"cached {df.shape} -> {CACHE}")

# Creates the allocator's bundle: the four sleeve return series plus
# the market features, cash, and class mapping.
def make_bundle() -> dict:
    from portfolio.run import load_bundle
    b = load_bundle()

    if not CACHE.exists():
        raise FileNotFoundError("run --build first")

    fp = json.loads(CACHE_META.read_text(encoding="utf-8"))
    if fp["deployed"] != DEPLOYED or list(fp["span"]) != list(SPAN):
        raise ValueError("sleeve-returns cache was built from other runs or another span — re-run --build")

    rets = pd.read_parquet(CACHE)
    ret = rets.reindex(b["dates"])
    classes = all_classes()
    trad = pd.DataFrame({s: b["tradeable"][classes[s]].any(axis=1) for s in CLASSES}, index=b["dates"])
    trad &= ret.notna()
    return {"dates": b["dates"], "ret": ret, "tradeable": trad,
            "cash": b["cash"], "classes": {s: [s] for s in CLASSES}}

# The allocator's environment, from its config's ENV; the callers pass
# their own lam.
def _env(ab: dict, window, lam: float, seed: int, scaling_name: str | None) -> Environment:
    e = _ACFG.ENV
    return Environment(list(CLASSES), MARKET, clock=e["clock"],
                       cash=e["cash"], scaling_name=scaling_name,
                       window=window, seed=seed, bundle=ab,
                       band=e["band"], eta=e["eta"], lam=lam,
                       warmup=e["warmup"], episode_len=e["episode_len"])

# Deployed checkpoints reproduce their reports, the cache matches a
# fresh recompute, the blend holds, the environment matches engine.run.
# Cached series are CONTINUOUS passes — their val slice differs from a
# fresh-start one by design.
def check() -> list:
    bad = []
    ab = make_bundle()
    from portfolio.run import load_bundle
    b = load_bundle()
    for s in CLASSES:
        run_dir = tr.AGENT_RUNS / s / DEPLOYED[s]
        want = None
        for line in (run_dir / "report.md").read_text(encoding="utf-8").splitlines():
            if "agent (best)" in line:
                want = float(line.split("|")[3])
                break
        net, meta = tr.load(run_dir / "best.pt")

        env = Environment(all_classes()[s], meta["features"],
                          clock=meta["env"]["clock"], cash=False,
                          scaling_name=meta["scaling"], window="val",
                          bundle=b, band=meta["env"]["band"],
                          eta=meta["env"]["eta"], lam=meta["env"]["lam"],
                          warmup=meta["env"]["warmup"],
                          episode_len=meta["env"]["episode_len"])
        
        s_v = tr.score(env, tr.det_policy(net, env.action_size), b, "val")[0]
        if want is None or abs(s_v["sharpe"] - want) > 6e-4:
            bad.append(f"allocator: {s} fresh val Sharpe "
                       f"{s_v['sharpe']:+.4f} does not reproduce the "
                       f"report's {want}")

    run_dir = OUT / DEPLOYED_ALLOCATOR
    want = None

    for line in (run_dir / "report.md").read_text(encoding="utf-8").splitlines():
        if "agent (best)" in line:
            want = float(line.split("|")[3])
            break

    net, meta = tr.load(run_dir / "best.pt")

    env = Environment(list(CLASSES), meta["features"],
                      clock=meta["env"]["clock"],
                      cash=meta["env"]["cash"],
                      scaling_name=meta["scaling"], window="val",
                      bundle=ab, band=meta["env"]["band"],
                      eta=meta["env"]["eta"], lam=meta["env"]["lam"],
                      warmup=meta["env"]["warmup"],
                      episode_len=meta["env"]["episode_len"])
    
    policy = tr.det_policy(net, env.action_size)
    if meta.get("kappa", 1.0) != 1.0:
        policy = tr.blended(policy, meta["kappa"])

    s_v = tr.score(env, policy, ab, "val")[0]
    if want is None or abs(s_v["sharpe"] - want) > 6e-4:
        bad.append(f"allocator: the deployed run's fresh val Sharpe "
                   f"{s_v['sharpe']:+.4f} does not reproduce the "
                   f"report's {want}")

    fresh = _sleeve_returns("bonds", b)
    cached = pd.read_parquet(CACHE)["bonds"].dropna()

    if not fresh.equals(cached.reindex(fresh.index)):
        bad.append("allocator: the bonds cache deviates from a fresh recompute")

    # kappa=1 is the identity, and the equal-shares prior is
    # a fixed point at every kappa.
    info_b = {"tradeable": np.array([True, True, True, True])}
    w = np.array([0.7, 0.1, 0.1, 0.0, 0.1])

    if not np.allclose(tr.blend(w, info_b, 1.0), w):
        bad.append("allocator: blend at kappa=1 is not the identity")
    prior = np.array([0.25, 0.25, 0.25, 0.25, 0.0])

    for kp in (0.25, 0.5, 1.0):

        if not np.allclose(tr.blend(prior, info_b, kp), prior):
            bad.append(f"allocator: the prior is not a blend fixed point at kappa {kp}")
        out_w = tr.blend(w, info_b, kp)

        if abs(out_w.sum() - 1.0) > 1e-12 or (out_w < -1e-12).any():
            bad.append(f"allocator: blend at kappa {kp} broke the simplex")

    env = _env(ab, ("2010-01-01", "2011-12-31"), lam=cfg.TURNOVER_LAMBDA,
               seed=0, scaling_name=None)
    k = env.action_size

    def equal(obs, info):
        live = info["tradeable"]
        a = np.zeros(k)
        a[:len(live)][live] = 1.0 / live.sum()
        return a

    out = env.evaluate(equal)
    win = ab["dates"][(ab["dates"] >= "2010-01-01") & (ab["dates"] <= "2011-12-31")]

    ret = ab["ret"].loc[win].copy()
    ret[engine.CASH] = ab["cash"].reindex(win).values
    ref = engine.run(out["decisions"], ret,
                     cfg.cost_rates(list(CLASSES) + [engine.CASH]),
                     band=_ACFG.ENV["band"])

    for key in ("value", "cost", "turnover"):
        dv = float(np.max(np.abs(out[key].values - ref[key].values)))
        if dv > 0.0:
            bad.append(f"allocator: scripted replay deviates on {key} by {dv:.2e}")
    return bad

# The allocator's training run: the manufactured world through the
# standard ceremony. This is the agent's ONE decision-bearing
# validation read.
def promote() -> dict:
    return tr.train("allocator", bundle=make_bundle(), extra_meta={"deployed_sleeves": DEPLOYED})

if __name__ == "__main__":
    from collectors._core import console_utf8
    console_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()

    if args.build:
        build()
    elif args.promote:
        promote()
    elif args.check:
        found = check()
        for f in found:
            print(f"    {f}")
        print("allocator " + ("OK" if not found
                              else f"{len(found)} VIOLATION(S)"))
        raise SystemExit(1 if found else 0)
    else:
        ap.error("pass one of --build, --check, --promote")
