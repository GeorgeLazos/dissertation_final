"""
portfolio/models/evaluate.py — score a saved run on a chosen split.

Loads a run folder's best.pt, rebuilds its environment from the FROZEN
meta (never live config), and scores the deterministic policy on the
requested window with the study's metrics. Validation is a free read of
a recorded quantity; test refuses without the acknowledgement flag, and
every test invocation is appended to self_reports/test_invocations.log.

The allocator's synthetic world ends at the validation boundary, so its
test pass belongs to the final composed evaluation, not here.

    python -m portfolio.models.evaluate --run agent_runs/bonds/<run_id>
    python -m portfolio.models.evaluate --run ... --window train
    python -m portfolio.models.evaluate --run ... --window test \
        --acknowledge-single-test-pass
"""
from __future__ import annotations
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from portfolio.models import train as tr
from portfolio.models.environment import Environment

ROOT = Path(__file__).resolve().parents[2]
TEST_LOG = ROOT / "self_reports" / "test_invocations.log"


# Score one run folder's best checkpoint on one window; returns the
# summary dict. The environment comes from the checkpoint's own meta.
def evaluate_run(run_dir: Path, window: str = "val") -> dict:
    net, meta = tr.load(run_dir / "best.pt")
    sleeve = meta["sleeve"]
    acfg = tr.agent_config(sleeve)

    if sleeve == "allocator":
        if window == "test":
            raise ValueError("the allocator's synthetic world ends at the "
                             "validation boundary; its test pass is the "
                             "final composed evaluation")
        from portfolio.models import allocate
        bundle = allocate.make_bundle()
        assets = list(allocate.CLASSES)
    else:
        from portfolio.run import load_bundle
        bundle = load_bundle()
        assets = tr.agent_assets(acfg)

    e = meta["env"]
    static = (tr.sector_static(assets)
              if meta.get("static") == "sectors" else None)
    env = Environment(assets, meta["features"], clock=e["clock"],
                      cash=e.get("cash", acfg.ENV["cash"]),
                      scaling_name=meta["scaling"], window=window,
                      bundle=bundle, band=e["band"], eta=e["eta"],
                      lam=e["lam"], warmup=e["warmup"],
                      episode_len=e["episode_len"], static=static)
    policy = tr.det_policy(net, env.action_size)
    if meta.get("kappa", 1.0) != 1.0:
        policy = tr.blended(policy, meta["kappa"])
    return tr.score(env, policy, bundle, window)[0]


if __name__ == "__main__":
    from collectors._core import console_utf8
    console_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="a run folder under agent_runs/<agent>/")
    ap.add_argument("--window", default="val",
                    choices=["train", "val", "test"])
    ap.add_argument("--acknowledge-single-test-pass", action="store_true")
    args = ap.parse_args()

    if args.window == "test" and not args.acknowledge_single_test_pass:
        raise SystemExit("test is touched ONCE, at the end, every strategy "
                         "together. Pass --acknowledge-single-test-pass if "
                         "this is that moment.")

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if not (run_dir / "best.pt").exists():
        raise SystemExit(f"no best.pt under {run_dir}")

    if args.window == "test":
        TEST_LOG.parent.mkdir(exist_ok=True)
        with TEST_LOG.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  test read  "
                    f"{run_dir.name}\n")

    s = evaluate_run(run_dir, args.window)
    print(f"{run_dir.parent.name}/{run_dir.name}  [{args.window}]")
    print(f"  sharpe {s['sharpe']:+.4f}  ann {s['annual_return']*100:+.2f}%  "
          f"vol {s['volatility']*100:.2f}%  mdd {s['max_drawdown']*100:.1f}%  "
          f"turnover {s['turnover_annual']*100:.0f}%")
