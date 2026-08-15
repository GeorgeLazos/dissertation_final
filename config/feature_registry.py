"""
config/feature_registry.py — the feature catalogue as data.

One entry per feature: what it is, which table it lives in, which block it
belongs to (the ablation unit), how much history it needs before its first
honest value, what it reads, and which asset classes it applies to. The
layer-2 builders implement EXACTLY this list — checks fail on any mismatch
in either direction — and layers 3 and 4 request observations by these
names.

    grain   asset  -> features_asset.parquet   (date, ticker)
            market -> features_market.parquet  (date)

    window  trading days of history before the first non-NaN value; features
            stay NaN through their warm-up, never filled

    classes which instruments carry a value; "all" = every panel instrument.
            Funds have no filings, so fundamental features are permanently
            NaN for them — declared here, not a gap.

    python -m config.feature_registry      # validate + summary
"""

from __future__ import annotations
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GRAINS = ("asset", "market")
GROUPS = ("momentum", "meanrev", "vol", "liquidity", "value", "growth",
          "quality", "factor", "sector", "fear", "rates", "economy",
          "cross_asset", "direction")

# Ablation blocks: the units switched on/off in the pre-registered arms.
BLOCKS = {
    "technical": ("momentum", "meanrev", "vol", "liquidity", "sector"),
    "fundamental": ("value", "growth", "quality"),
    "factor": ("factor",),
    "market": ("fear", "rates", "economy", "cross_asset", "direction"),
}

EQ = ("equities", "reits")          # instruments with filings
ALL = ("all",)
CLASSES = ("all", "equities", "bonds", "commodities", "reits")

# Input
def _f(grain, group, window, inputs, classes, note):
    return {"grain": grain, "group": group, "window": window,
            "inputs": inputs, "classes": classes, "note": note}


REGISTRY = {
    # ── asset · momentum ─────────────────────────────────────────────────
    "ret_1": _f("asset", "momentum", 1, ("ret",), ALL,
                "yesterday's total return, served raw"),

    "mom_21": _f("asset", "momentum", 21, ("ret",), ALL,
                 "1-month total return; doubles as the reversal channel"),

    "mom_63": _f("asset", "momentum", 63, ("ret",), ALL,
                 "3-month total return"),

    "mom_126": _f("asset", "momentum", 126, ("ret",), ALL,
                  "6-month total return"),

    "mom_252": _f("asset", "momentum", 252, ("ret",), ALL,
                  "12-month total return"),

    "mom_12_1": _f("asset", "momentum", 252, ("ret",), ALL,
                   "12-month return skipping the most recent month"),

    "sma50_gap": _f("asset", "momentum", 50, ("ret",), ALL,
                    "total-return index vs its 50d average"),

    "sma200_gap": _f("asset", "momentum", 200, ("ret",), ALL,
                     "total-return index vs its 200d average"),

    "sma_50_200": _f("asset", "momentum", 200, ("ret",), ALL,
                     "50d vs 200d average — the continuous golden cross"),

    "macd_norm": _f("asset", "momentum", 26, ("ret",), ALL,
                    "EMA12-EMA26 gap, price-normalized"),

    "macd_hist_norm": _f("asset", "momentum", 34, ("ret",), ALL,
                         "MACD vs its own 9d signal EMA, price-normalized — "
                         "the chained EMAs complete on session 26+9-1"),

    "adx_14": _f("asset", "momentum", 27, ("high", "low", "close"), ALL,
                 "Wilder trend strength 0-100; carries intraday range; the "
                 "chained smoothings complete on session 27 (measured)"),

    # high_52w is NOT served

    "seas_echo": _f("asset", "momentum", 525, ("ret",), ALL,
                    "mean same-calendar-month return over up to 5 annual "
                    "lags, >=2 required — declared long-window exception"),

    "sector_mom_126": _f("asset", "sector", 126, ("ret", "close"),
                         ("equities",),
                         "cap-weighted peer 126d momentum, self-excluded; "
                         "2025 GICS applied historically (declared: ~10 of "
                         "98 labels are future information; LIN NaN)"),

    # ── asset · mean reversion / oscillators ─────────────────────────────
    "rsi_14": _f("asset", "meanrev", 15, ("ret",), ALL,
                 "Wilder RSI; thin standalone evidence — the technical "
                 "on/off ablation arm adjudicates the whole block"),

    "stoch_k_14": _f("asset", "meanrev", 14, ("high", "low", "close"), ALL,
                     "stochastic %K on raw OHLC"),

    "stoch_d_3": _f("asset", "meanrev", 16, ("high", "low", "close"), ALL,
                    "3d smoothed %D"),

    "bb_z_20": _f("asset", "meanrev", 20, ("ret",), ALL,
                  "z-score inside the 20d Bollinger band"),

    "bb_bw_20": _f("asset", "meanrev", 20, ("ret",), ALL,
                   "Bollinger band width — 20d vol on a price scale"),

    # ── asset · volatility & risk ────────────────────────────────────────
    "rv_1": _f("asset", "vol", 2, ("ret",), ALL, "|yesterday's return|, annualized"),

    "rv_5": _f("asset", "vol", 5, ("ret",), ALL, "5d realized vol (HAR fast leg)"),

    "rv_21": _f("asset", "vol", 21, ("ret",), ALL, "21d realized vol"),

    "rv_63": _f("asset", "vol", 63, ("ret",), ALL, "63d realized vol"),

    "ewma_vol": _f("asset", "vol", 21, ("ret",), ALL,
                   "RiskMetrics EWMA vol, lambda 0.94"),

    "dd_252": _f("asset", "vol", 252, ("ret",), ALL,
                 "drawdown from the trailing-year total-return peak"),

    "mdd_252": _f("asset", "vol", 252, ("ret",), ALL,
                  "worst peak-to-trough inside the trailing year — "
                  "path-dependent, not recoverable from dd_252"),

    "downside_dev_63": _f("asset", "vol", 63, ("ret",), ALL,
                          "vol of losing days only (Sortino denominator)"),

    "beta_252": _f("asset", "vol", 252, ("ret",), ALL,
                   "252d beta vs SPY"),

    "corr_252": _f("asset", "vol", 252, ("ret",), ALL,
                   "252d correlation vs SPY — served beside beta because the "
                   "two components carry different information"),

    "coskew_252": _f("asset", "vol", 252, ("ret",), ALL,
                     "252d co-skewness with SPY, normalized"),

    # ── asset · liquidity ────────────────────────────────────────────────
    "amihud_21": _f("asset", "liquidity", 21, ("ret", "close", "volume"), ALL,
                    "log10 Amihud illiquidity — state, never expected alpha"),

    "dollar_vol_21": _f("asset", "liquidity", 21, ("close", "volume"), ALL,
                        "log10 of 21d average dollar volume"),

    "turnover_vol_63": _f("asset", "liquidity", 63, ("close", "volume"), ALL,
                          "63d std of log dollar volume — instability of "
                          "trading activity; admitted at R2 0.18 vs the "
                          "served liquidity/vol set (train)"),

    # ── asset · value ────────────────────────────────────────────────────
    "ep_ttm": _f("asset", "value", 300, ("fund", "close"), EQ,
                 "TTM net income to common / daily market cap"),

    "bm": _f("asset", "value", 0, ("fund", "close"), EQ,
             "book equity (parent-only) / daily market cap"),

    "dp_ttm": _f("asset", "value", 252, ("dividend", "close"), ALL,
                 "trailing-year panel dividends / close — the one valuation "
                 "feature covering funds; spin-off value excluded by column; "
                 "crash yields kept as signal (AIG 151% in 2009); trailing "
                 "cash over a post-spin price overstates ~1y after a large "
                 "spin (declared: MO 2008)"),

    "ebitda_ev": _f("asset", "value", 300, ("fund", "close"), EQ,
                    "TTM EBITDA / (mcap + debt - cash), EV floored at 20% of "
                    "mcap — near-zero EV explodes the ratio otherwise"),

    "ffo_yield": _f("asset", "value", 300, ("fund", "close"), ("reits",),
                    "REIT funds-from-operations (NI + D&A) / market cap"),

    "mktcap_log": _f("asset", "value", 0, ("fund", "close"), EQ,
                     "log10 daily market cap"),

    "reme": _f("asset", "value", 0, ("fund", "close"), EQ,
               "retained earnings / market cap"),

    "ocf_me": _f("asset", "value", 300, ("fund", "close"), EQ,
                 "TTM operating cash flow / market cap"),

    "net_payout_yield_ttm": _f("asset", "value", 300, ("fund", "close"), EQ,
                               "TTM dividends plus net repurchases / market "
                               "cap, both legs from the cash-flow statement; "
                               "admitted at R2 0.09 vs {dp_ttm, nsi_12m} — "
                               "the dollar buyback channel is genuinely new"),

    # ── asset · growth ───────────────────────────────────────────────────
    "asset_growth": _f("asset", "growth", 300, ("fund",), EQ,
                       "year-over-year total-asset growth (the investment "
                       "factor, sign negative in the literature)"),

    "rev_growth_ttm": _f("asset", "growth", 570, ("fund",), EQ,
                         "TTM revenue vs the prior year's TTM revenue"),

    "d_inv": _f("asset", "growth", 300, ("fund",), EQ,
                "year-over-year inventory change / average assets; "
                "permanently NaN for non-inventory holders by construction"),

    # ── asset · quality ──────────────────────────────────────────────────
    "cop_at": _f("asset", "quality", 300, ("fund",), EQ,
                 "TTM operating cash flow / average assets — the cash-based "
                 "profitability variant that survives value-weighting; "
                 "quality_rank ranks THIS column"),

    "op_prof": _f("asset", "quality", 300, ("fund",), EQ,
                  "TTM operating profitability / book equity (floored at 2% "
                  "of assets) — kept solely as the leverage channel"),

    "op_lev": _f("asset", "quality", 300, ("fund",), EQ,
                 "TTM (cost of revenue + SG&A) / assets — cost structure"),

    "noa_at": _f("asset", "quality", 0, ("fund",), EQ,
                 "net operating assets level / assets — balance-sheet bloat"),

    "accruals_ta": _f("asset", "quality", 300, ("fund",), EQ,
                      "(TTM net income - TTM operating cash flow) / average "
                      "assets — the cash-flow-statement accruals method"),

    "nsi_12m": _f("asset", "quality", 300, ("fund",), EQ,
                  "log change in split-adjusted shares over 4 quarters — "
                  "issuers lag, repurchasers lead"),

    "sue_q": _f("asset", "quality", 660, ("fund",), EQ,
                "standardized unexpected earnings: year-over-year quarterly "
                "EPS change / its own 8-quarter volatility"),

    "cash_at": _f("asset", "quality", 0, ("fund",), EQ,
                  "cash and equivalents / total assets; admitted at R2 0.41 "
                  "vs the served quality set; banks run structurally higher "
                  "(median 0.10 vs 0.07) — declared, not an error"),

    # ── asset · factor ranks ─────────────────────────────────────────────
    "size_rank": _f("asset", "factor", 0, ("mktcap_log",), EQ,
                    "same-day percentile of market cap within class"),

    "value_rank": _f("asset", "factor", 300, ("ep_ttm",), EQ,
                     "same-day percentile of earnings yield within class"),

    "quality_rank": _f("asset", "factor", 300, ("cop_at",), EQ,
                       "same-day percentile of cash profitability within class"),

    "mom_rank": _f("asset", "factor", 252, ("mom_252",), ALL,
                   "same-day percentile of 12m momentum WITHIN asset class"),

    # ── market · fear ────────────────────────────────────────────────────
    "vix": _f("market", "fear", 1, ("macro.vixcls",), ALL,
              "VIX close, lag 1 as served by layer 1"),

    "vix_chg_21": _f("market", "fear", 22, ("macro.vixcls",), ALL,
                     "21d change in VIX points"),

    "vrp_21": _f("market", "fear", 21, ("macro.vixcls", "ret"), ALL,
                 "variance risk premium: VIX-implied 21d variance minus SPY "
                 "realized 21d variance — the one market signal with "
                 "surviving out-of-sample forecast evidence"),

    # ── market · rates ───────────────────────────────────────────────────
    "dtb3": _f("market", "rates", 1, ("macro.dtb3",), ALL, "3m T-bill level"),

    "dtb3_chg_63": _f("market", "rates", 64, ("macro.dtb3",), ALL,
                      "quarterly change in the 3m rate — tightening/easing"),

    "dff": _f("market", "rates", 2, ("macro.dff",), ALL,
              "Fed funds effective rate, lag 2"),

    "term_spread": _f("market", "rates", 1, ("macro.dgs10", "macro.dtb3"), ALL,
                      "10y minus 3m yield — the curve slope"),

    "def_spread": _f("market", "rates", 1, ("macro.dbaa", "macro.daaa"), ALL,
                     "Baa minus Aaa yield — the credit-stress level"),

    # ── market · economy ─────────────────────────────────────────────────
    "cpi_yoy": _f("market", "economy", 0, ("macro.cpi_yoy",), ALL,
                  "inflation year-over-year, within-vintage, release-dated"),

    "unrate": _f("market", "economy", 0, ("macro.unrate",), ALL,
                 "unemployment rate as released"),

    "unrate_chg_12m": _f("market", "economy", 0, ("macro.unrate_chg_12m",), ALL,
                         "12m change in unemployment, within-vintage"),

    "gdpc1_yoy": _f("market", "economy", 0, ("macro.gdpc1_yoy",), ALL,
                    "real GDP growth year-over-year, within-vintage"),

    # ── market · cross-asset ─────────────────────────────────────────────
    "cred_ig_21": _f("market", "cross_asset", 21, ("ret",), ALL,
                     "21d LQD minus IEF return — the credit-spread flow"),

    "term_ret_21": _f("market", "cross_asset", 21, ("ret",), ALL,
                      "21d TLT minus SHY return — the duration flow"),

    "sb_corr_63": _f("market", "cross_asset", 63, ("ret",), ALL,
                     "63d SPY-TLT correlation — the bond sleeve's "
                     "diversification value right now"),

    "cs_disp_21": _f("market", "cross_asset", 21, ("ret",), ALL,
                     "21d mean cross-sectional std of equity returns — "
                     "lockstep vs spread-out market"),

    # ── market · direction ───────────────────────────────────────────────
    "spy_mom_252": _f("market", "direction", 252, ("ret",), ALL,
                      "SPY 12-month total return — the market's own trend"),

    "spy_bear_504": _f("market", "direction", 504, ("ret",), ALL,
                       "1.0 when SPY's trailing 2-year return is negative — "
                       "the momentum-crash panic state"),
}

# Names in registry order, filtered by grain and/or group.
def names(grain: str | None = None, group: str | None = None) -> list:
    return [n for n, s in REGISTRY.items()
            if (grain is None or s["grain"] == grain)
            and (group is None or s["group"] == group)]

# One feature's spec; KeyError on an unknown name is the desired failure.
def spec(name: str) -> dict:
    return REGISTRY[name]

# The longest warm-up among the given features (build-planning helper).
def max_window(feature_names=None) -> int:
    picked = feature_names if feature_names is not None else list(REGISTRY)
    return max(REGISTRY[n]["window"] for n in picked)

# Structural validation. Returns a list of violations, empty when sound.
def validate() -> list:
    bad = []
    for n, s in REGISTRY.items():
        if s["grain"] not in GRAINS:
            bad.append(f"{n}: unknown grain {s['grain']}")
        if s["group"] not in GROUPS:
            bad.append(f"{n}: unknown group {s['group']}")
        if not isinstance(s["window"], int) or s["window"] < 0:
            bad.append(f"{n}: bad window {s['window']}")
        if not s["inputs"]:
            bad.append(f"{n}: empty inputs")
        if not s["note"]:
            bad.append(f"{n}: empty note")
        if s["grain"] == "market" and s["classes"] != ALL:
            bad.append(f"{n}: market features have no per-class scope")
        unknown_cls = [c for c in s["classes"] if c not in CLASSES]
        if unknown_cls:
            bad.append(f"{n}: unknown classes {unknown_cls}")
    grouped = {g for gs in BLOCKS.values() for g in gs}
    missing = [g for g in GROUPS if g not in grouped]
    if missing:
        bad.append(f"groups outside every ablation block: {missing}")
    stray = [g for gs in BLOCKS.values() for g in gs if g not in GROUPS]
    if stray:
        bad.append(f"BLOCKS name groups that do not exist: {stray}")
    bad += _source_duplicates(Path(__file__))
    return bad


# A repeated key in a dict literal is LEGAL Python: the earlier entry is
# silently discarded and the parsed dict shows nothing, so a duplicated
# feature would vanish without a word. Only the SOURCE TEXT still holds the
# evidence — the keys are counted as written in the file.
def _source_duplicates(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") == "REGISTRY"):
            keys = [k.value for k in node.value.keys]
            dup = sorted({k for k in keys if keys.count(k) > 1})
            return [f"duplicate entries in the source: {dup}"] if dup else []
    return ["REGISTRY dict literal not found in the source"]

if __name__ == "__main__":
    from collectors._core import console_utf8
    console_utf8()
    problems = validate()
    n_asset, n_market = len(names("asset")), len(names("market"))
    print(f"registry: {len(REGISTRY)} features — {n_asset} asset, "
          f"{n_market} market")
    for g in GROUPS:
        row = names(group=g)
        if row:
            print(f"  {g:12s} {len(row):3d}  {', '.join(row[:6])}"
                  f"{' ...' if len(row) > 6 else ''}")
    if problems:
        print(f"INVALID - {len(problems)}:")
        for p in problems:
            print(f"  {p}")
    raise SystemExit(1 if problems else 0)
