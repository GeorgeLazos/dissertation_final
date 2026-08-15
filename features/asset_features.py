"""
features/asset_features.py — the per-(date,ticker) feature table.

Columns are exactly registry.names("asset"), in registry order, on the full
123-instrument grid. build() runs the sections in dependency order — price
grids, technicals, fundamentals, sector momentum, factor ranks — and
_assemble applies the tradeable mask and the registry contract in ONE place,
so a new section is masked and checked automatically.

Bases: return-driven features run on a per-ticker total-return index
compounded from `ret` (dividends included, base arbitrary); ADX, stochastics
and liquidity read the raw split-basis OHLCV. Daily market cap is the
filing's own figure scaled by the price move since its filing-date anchor:
mcap_t = mcap_filed * close_t / price_filed.

A filing is admissible from the first trading day AFTER it was published.
The newest fiscal period always wins; a restatement of an older period never
overwrites a newer one. TTM = the newest four distinct quarters as known at
the filing, NaN when the run is broken (span outside 250-400 days).

INPUT
    dataset.loader.prices()          (695,934 x 12)
    dataset.loader.fundamentals()    (12,568 x 77) filing grain
    config.tickers                   classes, sector map

OUTPUT  data/processed/asset_features.parquet   (695,934 x 59)
    date, ticker + the 57 registry asset columns, float64

    python -m features.asset_features            # build
    python -m features.asset_features --show     # print the stored table
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config.tickers import EQUITIES, all_classes
from dataset import loader
from dataset._core import cli, load_table
from config import feature_registry as registry

NAME = "asset_features"

# Fiscal items each filing must carry into the point-in-time stream.
_Q_ITEMS = ("netinccmn", "netinc", "revenue", "ncfo", "ebitda", "depamor",
            "cor", "sgna", "intexp", "inventory", "epsdil", "eps",
            "ncfdiv", "ncfcommon")
_LEVEL_ITEMS = ("assets", "equity", "retearn", "cashneq", "debt",
                "liabilities", "marketcap", "price", "sharesbas",
                "sharefactor")

# Real filings that are no basis for a year-over-year comparison: LIN's
# pre-merger shell (2018-09-30: $8.9M assets, zero revenue) is genuine
# history, but a ratio across it reads the Praxair merger as x10,000 growth.
# The shell may not serve as a comparison base; the quarter itself stays.
SHELL_QUARTERS = {("LIN", pd.Timestamp("2018-09-30"))}

# ── the point-in-time fundamentals stream ────────────────────────────────

#  the four newest quarters REPORTED SO FAR, added up 
# a window that slides forward with each filing, not the calendar year. 
# end_idx=4 gives the four before those, the previous year.
def _ttm(quarters: list, values: dict, item: str, end_idx: int = 0):
    qs = quarters[len(quarters) - 4 - end_idx: len(quarters) - end_idx]
    if len(qs) < 4:
        return np.nan
    span = (qs[-1] - qs[0]).days
    if not 250 <= span <= 400:
        return np.nan
    vals = [values[q].get(item, np.nan) for q in qs]
    return np.nan if any(pd.isna(v) for v in vals) else float(sum(vals))

# An item as it stood n QUARTERS back (n=4 is a year)
def _back(quarters: list, values: dict, item: str, n: int):
    if len(quarters) <= n:
        return np.nan
    q = quarters[-1 - n]
    prev = quarters[-1]
    if not 330 <= (prev - q).days <= 400:
        return np.nan
    return values[q].get(item, np.nan)

# The per-filing event stream for one ticker: at each accepted filing, every
# scalar the daily join needs, stamped with the publish date.
def _events(f: pd.DataFrame, ticker: str) -> pd.DataFrame:
    rows = []
    quarters, values = [], {}
    newest = pd.Timestamp.min
    # filings in the order they became PUBLIC, so the replay only ever knows
    # what the market knew
    for r in f.sort_values(["published", "quarter"]).itertuples():
        q = r.quarter
        # remember this filing's numbers; a correction overwrites the quarter
        rec = {i: getattr(r, i, np.nan) for i in _Q_ITEMS + _LEVEL_ITEMS}
        values[q] = rec
        if q not in quarters:
            quarters.append(q)
            quarters.sort()
        if q < newest:
            continue                    # restatement of an older period
        newest = q
        cur = values[q]

        # earnings per share, diluted first
        eps_now = cur.get("epsdil", np.nan)
        if pd.isna(eps_now):
            eps_now = cur.get("eps", np.nan)

        idx = quarters.index(q)

        # 13 quarters of EPS, newest first, blank where history runs out
        eps_hist = []
        for k in range(0, 13):
            j = idx - k
            if j < 0:
                eps_hist.append(np.nan)
                continue
            v = values[quarters[j]].get("epsdil", np.nan)
            if pd.isna(v):
                v = values[quarters[j]].get("eps", np.nan)
            eps_hist.append(v)

        # each quarter vs the same quarter a year earlier — cancels seasonality
        chgs = [eps_hist[k] - eps_hist[k + 4] for k in range(0, 8)]
        chgs = [c for c in chgs if not pd.isna(c)]
        # the newest change, in units of this company's OWN typical swing
        sue = (chgs[0] / np.std(chgs, ddof=1)
               if len(chgs) >= 6 and np.std(chgs, ddof=1) > 0 else np.nan)

        qs_here = quarters[:idx + 1]    # the past only, never a later quarter
        # levels now and a year ago
        assets = cur.get("assets", np.nan)
        assets_4 = _back(qs_here, values, "assets", 4)
        inv_4 = _back(qs_here, values, "inventory", 4)
        # a declared shell cannot be the comparison base, nor sit inside the
        # prior-year TTM span
        base_q = qs_here[-5] if len(qs_here) >= 5 else None
        prior_span = qs_here[-8:-4] if len(qs_here) >= 8 else []
        shell_base = base_q is not None and (ticker, base_q) in SHELL_QUARTERS
        shell_prior = any((ticker, q) in SHELL_QUARTERS for q in prior_span)
        if shell_base:
            assets_4 = np.nan
            inv_4 = np.nan
        # the year's midpoint — a company that grew mid-year is not judged
        # against its end-of-year size alone
        avg_assets = (assets + assets_4) / 2 if not (
            pd.isna(assets) or pd.isna(assets_4)) else np.nan
        # share counts on ONE split basis, or a 4-for-1 reads as a 300% issue
        shares = cur.get("sharesbas", np.nan)
        factor = cur.get("sharefactor", np.nan)
        shares_adj = shares * (factor if not pd.isna(factor) else 1.0)
        sh4 = _back(qs_here, values, "sharesbas", 4)
        sf4 = _back(qs_here, values, "sharefactor", 4)
        shares_adj_4 = (sh4 * (sf4 if not pd.isna(sf4) else 1.0)
                        if not pd.isna(sh4) else np.nan)

        rows.append({
            "published": r.published,
            "mcap_filed": cur.get("marketcap", np.nan),
            "price_filed": cur.get("price", np.nan),
            "equity": cur.get("equity", np.nan),
            "retearn": cur.get("retearn", np.nan),
            "assets": assets, "avg_assets": avg_assets,
            "cashneq": cur.get("cashneq", np.nan),
            "debt": cur.get("debt", np.nan),
            "liabilities": cur.get("liabilities", np.nan),
            "ttm_ni": _ttm(qs_here, values, "netinccmn"),
            "ttm_ni_total": _ttm(qs_here, values, "netinc"),
            "ttm_rev": _ttm(qs_here, values, "revenue"),
            "ttm_rev_prior": (np.nan if shell_prior
                              else _ttm(qs_here, values, "revenue", end_idx=4)),
            "ttm_ncfo": _ttm(qs_here, values, "ncfo"),
            "ttm_ebitda": _ttm(qs_here, values, "ebitda"),
            "ttm_depamor": _ttm(qs_here, values, "depamor"),
            "ttm_cor": _ttm(qs_here, values, "cor"),
            "ttm_sgna": _ttm(qs_here, values, "sgna"),
            "ttm_intexp": _ttm(qs_here, values, "intexp"),
            "ttm_ncfdiv": _ttm(qs_here, values, "ncfdiv"),
            "ttm_ncfcommon": _ttm(qs_here, values, "ncfcommon"),
            "cash_at_now": (cur.get("cashneq", np.nan) / assets
                            if not pd.isna(assets) and assets else np.nan),
            "inventory": cur.get("inventory", np.nan),
            "inventory_4": inv_4,
            "assets_4": assets_4,
            "shares_adj": shares_adj, "shares_adj_4": shares_adj_4,
            "sue": sue,
        })
    return pd.DataFrame(rows)

# Serve every event scalar onto the calendar: admissible strictly AFTER the
# publish date, carried forward until the next accepted filing.
def _serve(f_all: pd.DataFrame, cal: pd.DatetimeIndex, tickers: list) -> dict:
    grids = {}
    frames = {}
    for t in tickers:
        sub = f_all[f_all["ticker"] == t]
        if not len(sub):
            continue
        ev = _events(sub, t)
        if not len(ev):
            continue
        served = pd.merge_asof(
            pd.DataFrame({"date": cal}), ev.rename(columns={"published": "date"}),
            on="date", direction="backward", allow_exact_matches=False)
        served.index = cal
        frames[t] = served.drop(columns=["date"])
    if not frames:
        return grids
    items = next(iter(frames.values())).columns
    for item in items:
        grids[item] = pd.DataFrame(
            {t: frames[t][item] if t in frames else np.nan for t in tickers},
            index=cal)
    return grids

# ── technical engine (vectorized on date x ticker grids) ─────────────────

# Wilder's smoothing: an exponential average fading at 1/n a day.
def _wilder(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

# Trend STRENGTH, 0-100, blind to direction
def _adx(H, L, C, n=14) -> pd.DataFrame:
    # each day's push, up or down — the larger move wins, ties count as neither
    up, dn = H.diff(), -L.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    # true range: the day's full travel, overnight gaps included
    tr = pd.concat([(H - L), (H - C.shift()).abs(), (L - C.shift()).abs()]
                   ).groupby(level=0).max()
    atr = _wilder(tr, n)
    # what share of the movement went each way, and how lopsided that is
    pdi = 100 * _wilder(plus_dm, n) / atr
    mdi = 100 * _wilder(minus_dm, n) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return _wilder(dx, n)

# Recent gains against recent losses, mapped to 0-100
def _rsi(P, n=14) -> pd.DataFrame:
    d = P.diff()
    gain = _wilder(d.clip(lower=0), n)
    loss = _wilder(-d.clip(upper=0), n)
    return 100 - 100 / (1 + gain / loss)

# The deepest fall below the running high inside one window.
def _window_mdd(win: np.ndarray) -> float:
    peak = np.maximum.accumulate(win)
    return float(np.min(win / peak - 1))

# ── the build sections ───────────────────────────────────────────────────

# The shared ingredients, built once from the panel.
class _Grids(NamedTuple):
    R: pd.DataFrame          # daily total returns
    C: pd.DataFrame          # close, current split basis
    H: pd.DataFrame          # high
    L: pd.DataFrame          # low
    V: pd.DataFrame          # volume
    DIV: pd.DataFrame        # cash dividends per share
    P: pd.DataFrame          # total-return index, base 1 at listing
    spy: pd.Series           # SPY's return series
    cal: pd.DatetimeIndex
    tickers: list
    tradeable: pd.DataFrame  # close-presence, the universal mask

def _price_grids(p: pd.DataFrame) -> _Grids:
    tickers = sorted(p["ticker"].unique())
    grid = {c: p.pivot(index="date", columns="ticker", values=c)
            for c in ("ret", "close", "high", "low", "volume", "dividend")}
    R, C = grid["ret"], grid["close"]
    # total-return index per ticker: NaN before listing, base 1 at listing
    P = np.exp(np.log1p(R).fillna(0).cumsum()).where(C.notna())
    return _Grids(R, C, grid["high"], grid["low"], grid["volume"],
                  grid["dividend"], P, R["SPY"], R.index, tickers, C.notna())

# The 33 price-driven columns.
def _technical(g: _Grids) -> dict:
    R, C, H, L, V, P, spy = g.R, g.C, g.H, g.L, g.V, g.P, g.spy
    out = {}

    # One day-return
    out["ret_1"] = R 

    # total return over 1, 3, 6 and 12 months
    for w in (21, 63, 126, 252):
        out[f"mom_{w}"] = P / P.shift(w) - 1

    # 12 months ending a month ago — skips the short-term reversal effect
    out["mom_12_1"] = P.shift(21) / P.shift(252) - 1
    ma50 = P.rolling(50, min_periods=50).mean()
    ma200 = P.rolling(200, min_periods=200).mean()

    # how far above/below its own trend the price sits
    out["sma50_gap"] = P / ma50 - 1
    out["sma200_gap"] = P / ma200 - 1

    # fast trend vs slow trend — the golden cross as a continuous number
    out["sma_50_200"] = ma50 / ma200 - 1
    e12 = P.ewm(span=12, min_periods=26, adjust=False).mean()
    e26 = P.ewm(span=26, min_periods=26, adjust=False).mean()
    macd_line = e12 - e26
    signal = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()

    # fast minus slow EMA, and its gap from its own signal line; over price
    # so the scale is comparable across instruments
    out["macd_norm"] = macd_line / P
    out["macd_hist_norm"] = (macd_line - signal) / P
    out["adx_14"] = _adx(H, L, C)

    # this calendar month's return in each of the past 5 years, averaged
    r21 = R.rolling(21, min_periods=21).sum()
    lags = [r21.shift(252 * k) for k in range(1, 6)]
    cnt = sum(x.notna().astype(int) for x in lags)
    total = sum(x.fillna(0) for x in lags)
    out["seas_echo"] = (total / cnt).where(cnt >= 2)

    out["rsi_14"] = _rsi(P)
    lo14 = L.rolling(14, min_periods=14).min()
    hi14 = H.rolling(14, min_periods=14).max()

    # where the close sits inside the last 14 days' range, 0-100.
    out["stoch_k_14"] = (100 * (C - lo14) / (hi14 - lo14)).clip(0, 100)
    out["stoch_d_3"] = out["stoch_k_14"].rolling(3, min_periods=3).mean()
    ma20 = P.rolling(20, min_periods=20).mean()
    sd20 = P.rolling(20, min_periods=20).std()

    # standard deviations from the 20d mean, and the band's own width
    out["bb_z_20"] = (P - ma20) / sd20
    out["bb_bw_20"] = 4 * sd20 / ma20

    # volatility: yesterday's move, then rolling std at 5/21/63 days, all
    # annualized so they read as yearly percentages
    ann = np.sqrt(252)
    out["rv_1"] = R.abs() * ann
    for w in (5, 21, 63):
        out[f"rv_{w}"] = R.rolling(w, min_periods=w).std() * ann

    # the fast-reacting variant: recent days weighted most
    out["ewma_vol"] = np.sqrt(
        (R ** 2).ewm(alpha=0.06, adjust=False, min_periods=21).mean()) * ann
    
    # distance below the year's peak now, and the worst fall within the year
    peak252 = P.rolling(252, min_periods=252).max()
    out["dd_252"] = P / peak252 - 1
    out["mdd_252"] = P.rolling(252, min_periods=252).apply(_window_mdd, raw=True)

    # volatility of LOSING days only — the Sortino denominator
    out["downside_dev_63"] = np.sqrt(
        252 * (R.clip(upper=0) ** 2).rolling(63, min_periods=63).mean())

    # market relation over a year, as rolling-moment algebra since pandas
    # has no three-way rolling operation
    w = 252
    mR = R.rolling(w, min_periods=w).mean()
    mS = spy.rolling(w, min_periods=w).mean()
    cov = R.mul(spy, axis=0).rolling(w, min_periods=w).mean() - mR.mul(mS, axis=0)
    varS = spy.rolling(w, min_periods=w).var(ddof=0)
    sdR = R.rolling(w, min_periods=w).std(ddof=0)

    # how much it moves per unit of market move
    out["beta_252"] = cov.div(varS, axis=0)

    # how RELIABLY it moves with the market — beta conflates the two
    out["corr_252"] = cov.div(sdR.mul(np.sqrt(varS), axis=0))
    ES2 = (spy ** 2).rolling(w, min_periods=w).mean()
    ERS2 = R.mul(spy ** 2, axis=0).rolling(w, min_periods=w).mean()
    ERS = R.mul(spy, axis=0).rolling(w, min_periods=w).mean()

    # payoff during large market moves in either direction
    cosk_num = (ERS2 - mR.mul(ES2, axis=0) - 2 * ERS.mul(mS, axis=0)
                + 2 * mR.mul(mS ** 2, axis=0))
    out["coskew_252"] = cosk_num.div(sdR.mul(varS, axis=0))

    # liquidity, all log10: price impact per dollar traded, the dollar
    # volume itself, and how unstable that volume is
    dv = (C * V).where(V > 0)
    out["amihud_21"] = np.log10(
        (R.abs() / dv).rolling(21, min_periods=21).mean())
    out["dollar_vol_21"] = np.log10(dv.rolling(21, min_periods=21).mean())
    out["turnover_vol_63"] = np.log10(dv).rolling(63, min_periods=63).std()
    return out

# The 19 filing-driven columns, plus the daily market cap the sector and
# rank sections reuse.
def _fundamental(g: _Grids, served: dict) -> tuple:
    C = g.C
    out = {}

    # daily market cap: the filing's own figure moved by the price since its
    # filing-date anchor. Every valuation ratio divides by this.
    mcap = served["mcap_filed"] * C / served["price_filed"]
    out["mktcap_log"] = np.log10(mcap)

    # earnings and book equity per dollar of market value
    out["ep_ttm"] = served["ttm_ni"] / mcap
    out["bm"] = served["equity"] / mcap

    # dividend yield from the PANEL's own cash — so ETFs get one too
    div = g.DIV.where(C.notna())
    out["dp_ttm"] = div.rolling(252, min_periods=252).sum() / C

    # enterprise value = market cap + debt - cash, floored at 20% of mcap:
    # a bank's cash rivals its market cap and a near-zero EV explodes this
    ev = (mcap + served["debt"] - served["cashneq"]).clip(lower=0.2 * mcap)
    out["ebitda_ev"] = served["ttm_ebitda"] / ev

    # REIT earnings yield: profit with depreciation added back
    # (depreciation overstates the economic decline of property). REITs only.
    reit_set = set(all_classes()["reits"])
    ffo = (served["ttm_ni_total"] + served["ttm_depamor"]) / mcap
    out["ffo_yield"] = ffo[[t for t in g.tickers]].where(
        pd.DataFrame({t: t in reit_set for t in g.tickers},
                     index=g.cal, dtype=bool))
    
    # accumulated past profits, and cash-basis earnings, per unit of value
    out["reme"] = served["retearn"] / mcap
    out["ocf_me"] = served["ttm_ncfo"] / mcap

    # SF1 signs: both legs are cash OUTFLOWS (negative), so payout = -(sum)
    out["net_payout_yield_ttm"] = -(served["ttm_ncfdiv"]
                                    + served["ttm_ncfcommon"]) / mcap

    # year-over-year growth in the balance sheet and the top line, plus
    # inventory build-up relative to company size
    out["asset_growth"] = served["assets"] / served["assets_4"] - 1
    out["rev_growth_ttm"] = served["ttm_rev"] / served["ttm_rev_prior"] - 1
    out["d_inv"] = (served["inventory"]
                    - served["inventory_4"]) / served["avg_assets"]

    # cash generated per dollar of assets — the quality measure that survives
    # value-weighting in a mega-cap universe

    out["cop_at"] = served["ttm_ncfo"] / served["avg_assets"]
    # the same idea over EQUITY (so leverage shows through), floored at 2%
    # of assets because heavy buybacks can drive book equity negative

    eq_floor = served["equity"].clip(lower=0.02 * served["assets"])
    out["op_prof"] = (served["ttm_rev"] - served["ttm_cor"] - served["ttm_sgna"]
                      - served["ttm_intexp"]) / eq_floor
    
    # cost base relative to size — how much operating leverage it carries
    out["op_lev"] = (served["ttm_cor"] + served["ttm_sgna"]) / served["assets"]

    # capital tied up in operations rather than cash — balance-sheet bloat
    out["noa_at"] = ((served["assets"] - served["cashneq"])
                     - (served["liabilities"] - served["debt"])) / served["assets"]
    
    # profit that has NOT turned into cash — the earnings-quality warning
    out["accruals_ta"] = (served["ttm_ni_total"]
                          - served["ttm_ncfo"]) / served["avg_assets"]
    
    # share count change over a year: issuing is negative news, buying back
    # positive. Log so issuance and buybacks are symmetric.
    out["nsi_12m"] = np.log(served["shares_adj"] / served["shares_adj_4"])
    out["sue_q"] = served["sue"]
    out["cash_at"] = served["cash_at_now"]
    return out, mcap

# Cap-weighted peer momentum per sector, self-excluded. Equities only; a
# one-member sector (LIN) has no peer group and stays NaN.
def _sector_mom(mom126: pd.DataFrame, mcap: pd.DataFrame,
                cal: pd.DatetimeIndex, tickers: list) -> pd.DataFrame:
    sectors = pd.Series(EQUITIES)
    sec_mom = pd.DataFrame(np.nan, index=cal, columns=tickers)
    cap_for_w = mcap.copy()

    # a missing cap falls back to the row's mean cap, never a near-zero weight
    row_mean = cap_for_w.mean(axis=1)
    for sec in sectors.unique():
        members = [t for t in sectors[sectors == sec].index if t in tickers]
        if len(members) < 2:
            continue
        m_mom = mom126[members]
        m_cap = cap_for_w[members].apply(
            lambda col: col.fillna(row_mean))
        
        # sector totals once, then each member subtracts its own contribution
        w_sum = (m_mom * m_cap).sum(axis=1)
        c_sum = m_cap.where(m_mom.notna()).sum(axis=1)
        for t in members:
            wt = m_cap[t].where(m_mom[t].notna())
            num = w_sum - (m_mom[t] * wt).fillna(0)
            den = c_sum - wt.fillna(0)
            sec_mom[t] = num / den.where(den > 0)
    return sec_mom

# Same-day percentile ranks within each asset class, tradeable rows only.
def _ranks(out: dict, g: _Grids) -> dict:
    classes = all_classes()

    def _rank(source: pd.DataFrame, class_names: list) -> pd.DataFrame:
        ranked = pd.DataFrame(np.nan, index=g.cal, columns=g.tickers)
        for cname in class_names:
            members = [t for t in classes[cname] if t in g.tickers]
            sub = source[members].where(g.tradeable[members])
            ranked[members] = sub.rank(axis=1, pct=True)
        return ranked

    return {
        "size_rank": _rank(out["mktcap_log"], ["equities", "reits"]),
        "value_rank": _rank(out["ep_ttm"], ["equities", "reits"]),
        "quality_rank": _rank(out["cop_at"], ["equities", "reits"]),
        "mom_rank": _rank(out["mom_252"], list(classes)),
    }

# mask every column by the instrument's own existence, enforce the registry contract both ways, reshape to long.
def _assemble(out: dict, g: _Grids) -> pd.DataFrame:
    for c in out:
        out[c] = out[c].where(g.tradeable)
    cols = registry.names("asset")
    if set(out) != set(cols):
        raise ValueError(
            f"built columns != registry: extra {sorted(set(out) - set(cols))}, "
            f"missing {sorted(set(cols) - set(out))}")
    long = {c: out[c].stack() for c in cols}    # new-style stack keeps NaN rows
    result = pd.DataFrame(long).astype("float64")
    result.index.names = ["date", "ticker"]
    return result.reset_index().sort_values(["ticker", "date"]).reset_index(drop=True)

# ── build ────────────────────────────────────────────────────────────────

# Add a section's columns to the accumulator, refusing a name collision:
# dict.update would silently overwrite, and the registry check compares
# SETS of names so it cannot see one section clobbering another.
def _merge(out: dict, new: dict) -> None:
    clash = set(out) & set(new)
    if clash:
        raise ValueError(f"two sections both produced {sorted(clash)}")
    out.update(new)

def build() -> pd.DataFrame:
    g = _price_grids(loader.prices())
    out = _technical(g)
    fund, mcap = _fundamental(g, _serve(loader.fundamentals(), g.cal, g.tickers))
    _merge(out, fund)
    _merge(out, {"sector_mom_126": _sector_mom(out["mom_126"], mcap, g.cal, g.tickers)})
    _merge(out, _ranks(out, g))
    return _assemble(out, g)

# The stored asset feature table: one row per (date, ticker).
def load() -> pd.DataFrame:
    return load_table(NAME, deps=("calendar", "price_dataset",
                                  "fundamentals_dataset"), package="features")

if __name__ == "__main__":
    cli(NAME, build, "Build the per-asset feature table.")
