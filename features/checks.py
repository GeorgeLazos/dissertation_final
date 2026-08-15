"""
features/checks.py — verification of the two feature tables.

Families:
  registry   the catalogue is structurally sound
  market     bounds, anchors, warm-up honesty, columns == registry
  asset      bounds, anchors, class coverage, warm-up honesty
  crosstab   both tables align with the layer-1 grids

    python -m features.checks              # the light families
    python -m features.checks --pit        # the truncation leak test (~4 min)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from collectors._core import console_utf8
from config.splits import get_split
from config.tickers import all_classes, fund_tickers
from dataset import loader as dloader
from config import feature_registry as registry
from features import asset_features, market_features

_ANCHOR_PATH = Path(__file__).with_name("anchors.py")

TRUNCATION_DATE = "2016-06-30"

# Strict warm-up applies where the window is driven by the price history
# itself; fundamental windows are filing-driven floors, not exact counts.
_STRICT_WARMUP_GROUPS = ("momentum", "meanrev", "vol", "liquidity")

# Runs validate from the feature registry and reports the number of violations, if any.
def check_registry() -> list:
    bad = registry.validate()
    print(f"  registry   : {len(registry.REGISTRY)} features "
          f"({len(registry.names('asset'))} asset, "
          f"{len(registry.names('market'))} market), "
          f"violations {len(bad)}")
    return [f"registry: {b}" for b in bad]

# Check the market table for bounds, anchors, warm-up honesty, and column alignment with the registry.
def check_market() -> list:
    bad = []
    m = market_features.load().set_index("date")
    cal = pd.DatetimeIndex(dloader.calendar())
    want = registry.names("market")
    if list(m.columns) != want:
        bad.append(f"market: columns differ from the registry "
                   f"({set(m.columns) ^ set(want)})")
    if len(m) != len(cal) or not m.index.equals(cal):
        bad.append("market: rows != trading calendar")

    n_neg_vix = int((m["vix"] <= 0).sum())
    bear_vals = set(m["spy_bear_504"].dropna().unique())
    if n_neg_vix or not bear_vals <= {0.0, 1.0}:
        bad.append(f"market: impossible values (vix<=0 {n_neg_vix}, "
                   f"bear flags {bear_vals - {0.0, 1.0}})")

    # vix must equal the layer-1 column exactly — a frozen or re-joined vix
    # passes every bound but cannot survive an identity
    mac = dloader.macro().set_index("date")["vixcls"]
    n_vix = int(((m["vix"] - mac).abs() > 1e-12).sum() + (m["vix"].isna() != mac.isna()).sum())
    if n_vix:
        bad.append(f"market: vix differs from macro vixcls on {n_vix} days")

    # anchors against known market history
    vrp_mean = float(m["vrp_21"].mean())
    sb_pre = float(m.loc[:"2019-12-31", "sb_corr_63"].median())
    disp_ratio = float(m.loc["2008-11-03", "cs_disp_21"]
                       / m.loc["2017-06-01", "cs_disp_21"])
    bear09 = float(m.loc["2009", "spy_bear_504"].mean())
    bull17 = float(m.loc["2017", "spy_bear_504"].mean())
    print(f"  market     : vrp mean {vrp_mean:+.5f} (>0), pre-2020 "
          f"stock-bond corr {sb_pre:+.2f} (<0), crisis/calm dispersion "
          f"{disp_ratio:.1f}x, bear'09 {bear09:.2f}, bear'17 {bull17:.2f}")
    if vrp_mean <= 0:
        bad.append("market: the variance risk premium averages <= 0")
    if sb_pre >= 0:
        bad.append("market: pre-2020 stock-bond correlation not negative")
    if disp_ratio < 2:
        bad.append("market: crisis dispersion not above calm")
    if bear09 < 0.9 or bull17 > 0.1:
        bad.append("market: the bear flag misses 2009 or fires in 2017")

    # warm-up honesty for the price-windowed columns
    late = []
    for name in want:
        s = registry.spec(name)
        if s["window"] <= 1 or s["inputs"][0].startswith("macro."):
            continue
        first = m[name].first_valid_index()
        if first is not None and first < cal[min(s["window"] - 1, len(cal) - 1)]:
            late.append(name)
    if late:
        bad.append(f"market: values BEFORE the declared warm-up in {late}")
    holes = [c for c in want
             if m[c].loc[m[c].first_valid_index():].isna().any()]
    if holes:
        bad.append(f"market: interior NaN in {holes}")
    return bad

# Check the asset table for bounds, anchors, class coverage, and warm-up honesty.
def check_asset() -> list:
    bad = []
    a = asset_features.load()
    p = dloader.prices()
    want = registry.names("asset")
    got = [c for c in a.columns if c not in ("date", "ticker")]
    if got != want:
        bad.append(f"asset: columns differ from the registry "
                   f"({set(got) ^ set(want)})")
    if len(a) != len(p):
        bad.append(f"asset: {len(a):,} rows != panel {len(p):,}")

    # bounds that must hold by definition
    n_bad = 0
    for col, lo, hi in (("rsi_14", 0, 100), ("stoch_k_14", 0, 100),
                        ("stoch_d_3", 0, 100), ("adx_14", 0, 100),
                        ("size_rank", 0, 1), ("value_rank", 0, 1),
                        ("quality_rank", 0, 1), ("mom_rank", 0, 1)):
        n_bad += int((~a[col].dropna().between(lo - 1e-9, hi + 1e-9)).sum())
    n_bad += int((a["dd_252"].dropna() > 1e-12).sum())
    n_bad += int((a["mdd_252"].dropna() > 1e-9).sum())
    for col in ("rv_5", "rv_21", "rv_63", "ewma_vol", "downside_dev_63",
                "bb_bw_20"):
        n_bad += int((a[col].dropna() < 0).sum())
    if n_bad:
        bad.append(f"asset: {n_bad} rows violate definitional bounds")

    # no value may exist on a day the instrument did not trade — the mask
    # applied at assembly, asserted here so it cannot silently regress
    tr_all = a.merge(p[["date", "ticker", "tradeable"]], on=["date", "ticker"])
    off_rows = tr_all[~tr_all["tradeable"]]
    n_ghost = int(off_rows.drop(columns=["date", "ticker", "tradeable"])
                  .notna().sum().sum())
    if n_ghost:
        bad.append(f"asset: {n_ghost} feature values on non-tradeable rows")

    # class scope straight FROM THE REGISTRY: a feature declared for some
    # classes must be NaN everywhere else — no hand-maintained column list
    cls_members = {c: set(v) for c, v in all_classes().items()}
    scope_bad = []
    for name in want:
        s = registry.spec(name)
        if s["classes"] == ("all",):
            continue
        allowed = set().union(*[cls_members[c] for c in s["classes"]])
        n_out = int(a[~a["ticker"].isin(allowed)][name].notna().sum())
        if n_out:
            scope_bad.append((name, n_out))
    if scope_bad:
        bad.append(f"asset: values outside the declared class scope {scope_bad}")

    # anchors
    spy = a[a["ticker"] == "SPY"].set_index("date")
    beta_dev = float((spy["beta_252"].dropna() - 1).abs().max())
    ko = a[a["ticker"] == "KO"].set_index("date")
    ko_dp = float(ko.loc["2015-06-01", "dp_ttm"])
    aapl = a[a["ticker"] == "AAPL"].set_index("date")
    aapl_ep = float(aapl.loc["2016-06-01", "ep_ttm"])
    dp_max = float(a["dp_ttm"].max())
    print(f"  asset      : bounds bad {n_bad}, SPY beta dev {beta_dev:.1e}, "
          f"KO dp'15 {ko_dp:.3f} (~0.03), AAPL ep'16 {aapl_ep:.3f} (~0.09), "
          f"dp max {dp_max:.2f} (AIG'09 crash yield, declared < 2)")
    if beta_dev > 1e-9:
        bad.append(f"asset: SPY's own beta deviates from 1 by {beta_dev:.1e}")
    if not 0.02 < ko_dp < 0.045:
        bad.append(f"asset: KO 2015 dividend yield {ko_dp:.4f} implausible")
    if not 0.06 < aapl_ep < 0.13:
        bad.append(f"asset: AAPL 2016 earnings yield {aapl_ep:.4f} implausible")
    if dp_max > 2:
        bad.append(f"asset: dp_ttm max {dp_max:.2f} beyond the declared "
                   f"crash-yield ceiling")

    # beta must be BETA, not its bounded lookalike: correlation can never
    # leave [-1,1], real betas do
    beta_max = float(a["beta_252"].abs().max())
    if beta_max < 1.5:
        bad.append(f"asset: beta_252 never exceeds 1.5 in magnitude "
                   f"({beta_max:.2f}) — it may be a bounded lookalike")

    # the four rank columns must BE percentiles of their source, per class:
    # uniform across [0,1] and monotone with the source on sampled days
    rank_src = {"size_rank": "mktcap_log", "value_rank": "ep_ttm",
                "quality_rank": "cop_at", "mom_rank": "mom_252"}
    tr_rows = tr_all[tr_all["tradeable"]]
    days = tr_rows["date"].drop_duplicates().sort_values()
    sample_days = days.iloc[np.linspace(300, len(days) - 1, 12).astype(int)]
    rank_bad = []
    for rk, src in rank_src.items():
        pooled = tr_rows[rk].dropna()
        q1, q3 = pooled.quantile(0.25), pooled.quantile(0.75)
        if not (0.17 <= q1 <= 0.33 and 0.67 <= q3 <= 0.83):
            rank_bad.append((rk, f"not uniform (q1 {q1:.2f}, q3 {q3:.2f})"))
        worst = 1.0
        for d in sample_days:
            for cname, members in all_classes().items():
                s = tr_rows[(tr_rows["date"] == d)
                            & tr_rows["ticker"].isin(members)][[rk, src]].dropna()
                if len(s) >= 5:
                    rho = s[rk].corr(s[src], method="spearman")
                    worst = min(worst, rho)
        if worst < 0.97:
            rank_bad.append((rk, f"not monotone with {src} (rho {worst:.2f})"))
    if rank_bad:
        bad.append(f"asset: rank columns broken {rank_bad}")

    # fundamentals must ARRIVE on time: a served step in a price-independent
    # ratio may occur only on a filing's first admissible day — a join one
    # quarter late moves every step off that set and fails here
    f_all = dloader.fundamentals()
    cal_idx = pd.DatetimeIndex(dloader.calendar())
    step_bad = []
    for tk in ("AAPL", "JPM", "KO", "XOM", "PG", "O", "MSFT", "PM"):
        srv = a[a["ticker"] == tk].set_index("date")["cop_at"].dropna()
        steps = srv.index[srv.diff().abs() > 1e-12]
        pubs = f_all[f_all["ticker"] == tk]["published"]
        admissible = {cal_idx[cal_idx.searchsorted(d1, side="right")]
                      for d1 in pubs if d1 < cal_idx[-1]}
        stray = [d for d in steps if d not in admissible]
        if stray:
            step_bad.append((tk, len(stray), str(stray[0].date())))
    print(f"  asset      : ghost values {n_ghost}, scope violations "
          f"{len(scope_bad)}, beta max {beta_max:.2f}, rank issues "
          f"{len(rank_bad)}, off-schedule fundamental steps "
          f"{step_bad or 'none'}")
    if step_bad:
        bad.append(f"asset: fundamental values step on non-admissible days "
                   f"{step_bad} — the point-in-time join is off schedule")

    # class coverage: funds must carry NO filing-driven values; FFO is REITs
    # only; every instrument carries technicals
    funds = set(fund_tickers())
    fsub = a[a["ticker"].isin(funds)]
    leak_cols = [c for c in ("ep_ttm", "bm", "ebitda_ev", "cop_at", "nsi_12m",
                             "asset_growth", "sue_q", "mktcap_log", "cash_at",
                             "net_payout_yield_ttm")
                 if fsub[c].notna().any()]
    if leak_cols:
        bad.append(f"asset: fund rows carry fundamental values in {leak_cols}")
    reits = set(all_classes()["reits"])
    ffo_wrong = a[a["ffo_yield"].notna() & ~a["ticker"].isin(reits)]
    if len(ffo_wrong):
        bad.append(f"asset: ffo_yield outside REITs "
                   f"({sorted(ffo_wrong['ticker'].unique())[:4]})")
    cov = a[a["ret_1"].notna()]["mom_252"].notna().mean()
    print(f"  asset      : fund fundamental leak-cols {len(leak_cols)}, "
          f"ffo outside REITs {len(ffo_wrong)}, mom_252 coverage on "
          f"traded rows {cov:.1%}")
    if cov < 0.90:
        bad.append(f"asset: mom_252 coverage {cov:.1%} below 90% — "
                   f"a technical block hole")

    # warm-up honesty: no technical value before its window is filled,
    # measured from each ticker's own first TRADED session (0-based index n;
    # a window of w sessions allows the first value at n = w-1)
    tr = a.merge(p[["date", "ticker", "tradeable"]], on=["date", "ticker"])
    tr = tr[tr["tradeable"]].copy()
    tr["n"] = tr.groupby("ticker").cumcount()
    early = []
    for name in want:
        s = registry.spec(name)
        if s["group"] not in _STRICT_WARMUP_GROUPS or s["window"] <= 1:
            continue
        n_early = int((tr[name].notna() & (tr["n"] < s["window"] - 1)).sum())
        if n_early:
            early.append((name, n_early))
    if early:
        bad.append(f"asset: values before the declared warm-up {early[:5]}")
    else:
        print(f"  asset      : warm-up honest for all "
              f"{sum(1 for n in want if registry.spec(n)['group'] in _STRICT_WARMUP_GROUPS)} "
              f"price-windowed features")
    return bad

# Check that the asset and market tables align with the layer-1 grids.
def check_crosstab() -> list:
    bad = []
    a = asset_features.load()
    p = dloader.prices()
    keys_a = set(zip(a["date"], a["ticker"]))
    keys_p = set(zip(p["date"], p["ticker"]))
    if keys_a != keys_p:
        bad.append(f"crosstab: asset keys differ from the panel by "
                   f"{len(keys_a ^ keys_p)}")
    m = market_features.load()
    cal = pd.DatetimeIndex(dloader.calendar())
    if not pd.DatetimeIndex(m["date"]).equals(cal):
        bad.append("crosstab: market dates != calendar")
    print(f"  crosstab   : asset keys == panel keys "
          f"({'yes' if keys_a == keys_p else 'NO'}), market dates == calendar")
    return bad

# ── the distribution anchors ─────────────────────────────────────────────

# Quartiles of every column per split, from the current tables.
def _split_quartiles() -> dict:
    out = {}
    for grain, df in (("asset", asset_features.load()),
                      ("market", market_features.load())):
        for split in ("train", "val", "test"):
            s, e = get_split(split)
            sub = df[(df["date"] >= s) & (df["date"] <= e)]
            for c in registry.names(grain):
                v = sub[c].dropna()
                if len(v) < 50:
                    continue
                out.setdefault(c, {})[split] = tuple(
                    round(float(v.quantile(q)), 6) for q in (0.25, 0.5, 0.75))
    return out

# Freeze the current quartiles as the snapshot the anchor check compares to.
def write_anchors() -> None:
    q = _split_quartiles()
    lines = ['"""AUTO-GENERATED distribution snapshot — the anchor check',
             'compares every column, every split, against these quartiles.',
             'Regenerate ONLY after an intended data change, and only once',
             'the full check suite is green:',
             '    python -m features.checks --write-anchors',
             '"""', "", "ANCHORS = {"]
    for c in sorted(q):
        lines.append(f"    {c!r}: {q[c]!r},")
    lines.append("}")
    _ANCHOR_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  anchors    : wrote {sum(len(v) for v in q.values())} "
          f"(column, split) snapshots -> {_ANCHOR_PATH.name}")

# Every column's median must sit inside its snapshot band and its spread must
# not collapse or explode: one mechanism that catches sign flips, unit
# errors, frozen columns and swaps — in EVERY split, including test.
def check_anchors() -> list:
    bad = []
    if not _ANCHOR_PATH.exists():
        return ["anchors: no snapshot — run python -m features.checks "
                "--write-anchors on a verified build"]
    from features.anchors import ANCHORS
    q_now = _split_quartiles()
    off = []
    for c, splits in ANCHORS.items():
        for split, (p25s, p50s, p75s) in splits.items():
            if c not in q_now or split not in q_now[c]:
                off.append((c, split, "no longer computable"))
                continue
            p25, p50, p75 = q_now[c][split]
            def _floor(lo, mid, hi):
                return max(hi - lo, 0.05 * abs(mid), 1e-6)
            iqr_s = _floor(p25s, p50s, p75s)
            if abs(p50 - p50s) > 0.75 * iqr_s:
                off.append((c, split, f"median {p50:.4g} vs {p50s:.4g}"))
            iqr = _floor(p25, p50, p75)
            if not 0.25 <= iqr / iqr_s <= 4.0:
                off.append((c, split, f"spread x{iqr / iqr_s:.2f}"))
    print(f"  anchors    : {sum(len(v) for v in ANCHORS.values())} "
          f"(column, split) bands checked, off: {len(off)} "
          f"{off[:3] if off else ''}")
    if off:
        bad.append(f"anchors: {len(off)} distribution shifts {off[:6]}")
    return bad

# ── the truncation leak test ─────────────────────────────────────────────

# Compare a truncated rebuild against the stored table on the surviving
# rows: numeric difference AND NaN-pattern difference both count.
def compare_frames(full: pd.DataFrame, trunc: pd.DataFrame,
                   key_cols: list) -> dict:
    cols = [c for c in trunc.columns if c not in key_cols]
    f = full.merge(trunc[key_cols], on=key_cols, how="inner")
    f = f.sort_values(key_cols).reset_index(drop=True)
    t = trunc.sort_values(key_cols).reset_index(drop=True)
    verdict = {}
    for c in cols:
        num = int(((f[c] - t[c]).abs() > 1e-9).sum())
        pat = int((f[c].isna() != t[c].isna()).sum())
        if num or pat:
            verdict[c] = (num, pat)
    return verdict

def run_pit_gate() -> list:
    bad = []
    T = pd.Timestamp(TRUNCATION_DATE)
    print(f"  pit        : truncating every layer-1 input at "
          f"{TRUNCATION_DATE} and rebuilding both tables ...")

    full_m = market_features.load()
    full_a = asset_features.load()

    real = {"prices": dloader.prices, "macro": dloader.macro,
            "fundamentals": dloader.fundamentals, "calendar": dloader.calendar}
    P = real["prices"]()
    M = real["macro"]()
    F = real["fundamentals"]()
    CAL = real["calendar"]()

    # any loader accessor NOT truncation-patched must fail LOUDLY if a
    # builder reaches it: a leak through an unpatched channel would rebuild
    # from identical data and show a zero difference
    def _raiser(attr):
        def fail(*a_, **k_):
            raise RuntimeError(f"PIT gate: dataset.loader.{attr} is not "
                               f"truncation-patched — extend the gate")
        return fail

    try:
        dloader.prices = lambda split=None: P[P["date"] <= T].reset_index(drop=True)
        dloader.macro = lambda split=None: M[M["date"] <= T].reset_index(drop=True)
        dloader.fundamentals = lambda split=None: (
            F[F["published"] <= T].reset_index(drop=True))
        dloader.calendar = lambda: CAL[CAL <= T]
        for attr in dir(dloader):
            if attr.startswith("_") or attr in real:
                continue
            obj = getattr(dloader, attr)
            if callable(obj) and getattr(obj, "__module__", "") == "dataset.loader":
                real[attr] = obj
                setattr(dloader, attr, _raiser(attr))
        trunc_m = market_features.build()
        trunc_a = asset_features.build()
    finally:
        for k, v in real.items():
            setattr(dloader, k, v)

    vm = compare_frames(full_m, trunc_m, ["date"])
    va = compare_frames(full_a, trunc_a, ["date", "ticker"])
    print(f"  pit        : market — {len(vm)} leaking columns "
          f"{dict(list(vm.items())[:4]) if vm else ''}")
    print(f"  pit        : asset  — {len(va)} leaking columns "
          f"{dict(list(va.items())[:4]) if va else ''}")
    if vm:
        bad.append(f"pit: market columns change when the future is deleted: "
                   f"{sorted(vm)}")
    if va:
        bad.append(f"pit: asset columns change when the future is deleted: "
                   f"{sorted(va)}")
    return bad

FAMILIES = {"registry": check_registry, "market": check_market,
            "asset": check_asset, "crosstab": check_crosstab,
            "anchors": check_anchors}

if __name__ == "__main__":
    console_utf8()
    ap = argparse.ArgumentParser(description="Verify the feature tables.")
    for name in FAMILIES:
        ap.add_argument(f"--{name}", action="store_true",
                        help=f"check {name} only")
    ap.add_argument("--pit", action="store_true",
                    help="run the truncation leak-gate (heavy)")
    ap.add_argument("--write-anchors", action="store_true",
                    help="freeze the current distributions as the snapshot")
    args = ap.parse_args()
    if args.write_anchors:
        write_anchors()
        raise SystemExit(0)
    chosen = [n for n in FAMILIES if getattr(args, n)]
    if not chosen and not args.pit:
        chosen = list(FAMILIES)

    failures = []
    for name in chosen:
        print(f"{name.upper()}")
        failures += FAMILIES[name]()
    if args.pit:
        print("PIT GATE")
        failures += run_pit_gate()
    print()
    if failures:
        print(f"FAILED - {len(failures)} check(s):")
        for b in failures:
            print(f"  {b}")
    else:
        ran = chosen + (["pit"] if args.pit else [])
        print(f"all checks passed ({', '.join(ran)})")
    raise SystemExit(1 if failures else 0)
