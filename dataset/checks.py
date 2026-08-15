"""
dataset/checks.py — verification of the processed tables. Separate from the
builds on purpose: a build never runs checks implicitly, and checks never
modify anything. Run before committing results, not on every rebuild.

Three kinds of check:
  internal      a table obeys its own rules (ret reconciles, no weekends)
  cross-source  our tables vs an INDEPENDENT source (yfinance for prices,
                Sharadar's own closeadj for returns, raw FRED vintages for
                the revised macro series)
  cross-table   the tables agree with each other (panel dates == calendar)

Known vendor faults are DECLARED in the constants below, each with its
evidence. A check fails on anything outside its declared set — and also when
a declared fault disappears, so a stale declaration cannot linger.

Exit code: 0 iff every check passes. Cross-vendor disagreement is expected at
a low rate (the second vendor has its own errors); those checks fail only
above a ceiling, and always PRINT what they saw.

    python -m dataset.checks               # everything
    python -m dataset.checks --prices      # one family
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from collectors import (fred_macro, sharadar_actions, sharadar_fundamentals,
                        sharadar_prices, yfinance_prices)
from collectors._core import console_utf8
from config.tickers import fund_tickers, sharadar_tickers
from dataset import calendar, fundamentals_dataset, macro_dataset, price_dataset
from dataset.macro_dataset import DAILY_LAGS, DERIVED, _daily


# Quarter gaps that are real history, not extraction faults: the 2006
# options-backdating filing delays. Quarters absent from the record itself.
KNOWN_QUARTER_GAPS = {
    ("AAPL", "2006-03-31", "2006-09-30"),
    ("AMT", "2006-03-31", "2006-09-30"),
    ("UNH", "2006-03-31", "2006-12-31"),
}

# Days where FRED's VIXCLS and yfinance's ^VIX close disagree by more than
# 0.5. No third source exists to arbitrate; FRED is served. On 2026-02-06
# FRED prints below yahoo's own daily low, so FRED is the likely outlier.
KNOWN_VIX_DAYS = {"2008-11-28", "2014-10-15", "2026-02-06"}

# Volume cells corrupt in the vendor feed and waived rather than corrected.
# Currently empty: every proven-wrong cell has a repair source — yfinance for
# the dropouts, Yahoo's archived 2008-era records for 2008-09-19 — and is
# corrected in the build (price_dataset.VOLUME_FIXES / VOLUME_RESTORED /
# BAR_FIXES). The stale-correction check asserts each is still needed.
KNOWN_VOLUME_DEFECTS = ()

# Extreme bars VERIFIED REAL: both vendors agree and the day is a documented
# crisis session (AIG's bailout week, the MS runs, PG's flash crash). The
# bar-plausibility check requires every extreme bar to be either fixed or
# on this list — and complains if a listed bar stops being extreme.
KNOWN_EXTREME_BARS = {
    ("AIG", "2008-09-15"), ("AIG", "2008-09-16"), ("AIG", "2008-09-17"),
    ("AIG", "2008-09-19"), ("ORCL", "2008-09-19"), ("MS", "2008-09-17"),
    ("MS", "2008-09-18"), ("MS", "2008-10-10"), ("MS", "2008-10-28"),
    ("BNY", "2008-09-18"), ("AMD", "2008-09-29"), ("BAC", "2009-02-20"),
    ("USB", "2009-01-21"), ("PG", "2010-05-06"), ("WELL", "2020-03-18"),
}

# Days where the SECOND vendor's volume is the wrong one. Skipped, not
# waived: nothing of ours is at fault on these days. BRK-B 2021: yfinance
# 100x low while Sharadar matches its own rhythm. The 2008-09-19 four:
# yfinance's live feed still carries the /100 truncation that the panel has
# restored from the archived record.
KNOWN_YF_VOLUME_DAYS = {
    ("BRK-B", "2021-03-16"), ("BRK-B", "2021-03-17"),
    ("AMGN", "2008-09-19"), ("CMCSA", "2008-09-19"),
    ("COST", "2008-09-19"), ("CSCO", "2008-09-19"),
    ("MS", "2008-09-19"), ("SCHW", "2008-09-19"),
    ("O", "2008-09-19"), ("BNY", "2008-09-19"),
}

_VOL_RANGES = [(t, pd.Timestamp(a), pd.Timestamp(b))
               for t, a, b in KNOWN_VOLUME_DEFECTS]


def _declared_volume(ticker: str, day: pd.Timestamp) -> bool:
    return any(t == ticker and a <= day <= b for t, a, b in _VOL_RANGES)


# ── calendar ─────────────────────────────────────────────────────────────

def check_calendar() -> list:
    bad = []
    cal = calendar.load()["date"]
    n_we = int((cal.dt.weekday >= 5).sum())
    n_dup = int(cal.duplicated().sum())
    if n_we:
        bad.append(f"calendar: {n_we} weekend rows")
    if n_dup:
        bad.append(f"calendar: {n_dup} duplicate dates")

    # A full US market year has ~252 sessions. The final year is partial, so
    # its band is pro-rated by how far into the year the calendar reaches.
    per_year = cal.dt.year.value_counts().sort_index()
    last_year = int(cal.dt.year.max())
    frac = float(cal.max().dayofyear) / 365.25
    off = {}
    for y, n in per_year.items():
        lo, hi = ((245, 260) if y < last_year
                  else (int(245 * frac) - 4, int(260 * frac) + 4))
        if not lo <= n <= hi:
            off[int(y)] = int(n)
    if off:
        bad.append(f"calendar: implausible sessions/year {off}")
    full = per_year[per_year.index < last_year]
    print(f"  calendar   : {len(cal):,} days, weekends {n_we}, dups {n_dup}, "
          f"{full.min()}-{full.max()} sessions/yr, "
          f"final year {int(per_year[last_year])} in [{int(245 * frac) - 4},"
          f"{int(260 * frac) + 4}]")
    return bad


# ── prices ───────────────────────────────────────────────────────────────

def check_prices() -> list:
    bad = []
    p = price_dataset.load()
    cal = calendar.load()["date"]

    # internal: complete grid, unique keys, date sets identical both ways,
    # no session where nothing trades, mask exactly equals close-presence
    n_days, n_tk = p["date"].nunique(), p["ticker"].nunique()
    short = len(p) - n_days * n_tk
    if short:
        bad.append(f"prices: grid incomplete {len(p):,} != {n_days * n_tk:,}")
    n_dupk = int(p.duplicated(["date", "ticker"]).sum())
    if n_dupk:
        # duplicate keys make every downstream join ill-defined — stop here
        bad.append(f"prices: {n_dupk} duplicate (date,ticker) rows")
        print(f"  prices     : {n_dupk} duplicate keys — remaining checks skipped")
        return bad
    if set(p["date"]) != set(cal):
        bad.append("prices: panel dates != calendar dates")
    n_dead = int((p.groupby("date")["tradeable"].sum() == 0).sum())
    if n_dead:
        bad.append(f"prices: {n_dead} sessions with no tradeable instrument")
    if (p["tradeable"] != p["close"].notna()).any():
        bad.append("prices: tradeable mask != close presence")

    # internal: every traded bar is a possible bar. close carries full
    # precision while open/high/low keep the vendor's 3-decimal rounding, so
    # the ordering is tested with half-a-cent tolerance.
    t_ = p[p["tradeable"]]
    tol = 0.006
    n_ohlc = int(((t_["low"] - t_[["open", "close"]].min(axis=1) > tol) |
                  (t_[["open", "close"]].max(axis=1) - t_["high"] > tol) |
                  (t_[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum())
    if n_ohlc:
        bad.append(f"prices: {n_ohlc} bars violate 0 < low <= open,close <= high")
    n_negv = int((t_["volume"] < 0).sum())
    if n_negv:
        bad.append(f"prices: {n_negv} bars with negative volume")

    # internal: bar plausibility. A low far below (or high far above) the
    # bar's own open/close is either a vendor bad tick — fixed in the build —
    # or a real crisis session, declared by name above. Both directions:
    # a new extreme bar and a declared one that vanished each fail.
    rl = t_["low"] / t_[["open", "close"]].min(axis=1)
    rh = t_["high"] / t_[["open", "close"]].max(axis=1)
    ext = {(r.ticker, str(r.date.date()))
           for r in t_[(rl < 0.75) | (rh > 1.33)].itertuples()}
    new_ext = sorted(ext - KNOWN_EXTREME_BARS)
    gone_ext = sorted(KNOWN_EXTREME_BARS - ext)
    print(f"  prices     : extreme bars {len(ext)} "
          f"({len(ext) - len(new_ext)} declared real), new: {new_ext or 'none'}")
    if new_ext:
        bad.append(f"prices: NEW extreme bars (bad tick or undeclared crisis "
                   f"day) {new_ext[:5]}")
    if gone_ext:
        bad.append(f"prices: declared extreme bars no longer extreme "
                   f"{gone_ext} — update the declaration")

    # internal: volume dropouts against the ticker's OWN rhythm — the only
    # test that can see a bad day both vendors share (2008-09-19)
    ts = t_.sort_values(["ticker", "date"])
    med = ts.groupby("ticker")["volume"].transform(
        lambda s: s.rolling(63, min_periods=20).median())
    drop = ts[(ts["volume"] < 0.02 * med) & med.notna()]
    new_drop = [(r.ticker, str(r.date.date())) for r in drop.itertuples()
                if not _declared_volume(r.ticker, r.date)]
    if new_drop:
        bad.append(f"prices: NEW volume dropouts {new_drop[:5]}")

    # internal: the stored return reconciles with its own ingredients, in
    # value AND in NaN pattern — max(0, nan) is 0.0 in Python, so a blanked
    # return that only a pattern test can see must be tested for
    q = p.sort_values(["ticker", "date"])
    prev = q.groupby("ticker")["close"].shift(1)
    recomputed = (q["close"] + q["dividend"] + q["spinoff_value"]) / prev - 1
    n_num = int(((recomputed - q["ret"]).abs() > 1e-12).sum())
    n_pat = int((recomputed.isna() != q["ret"].isna()).sum())
    if n_num or n_pat:
        bad.append(f"prices: ret broken ({n_num} numeric, {n_pat} NaN-pattern)")
    print(f"  prices     : grid {n_tk} x {n_days:,} "
          f"{'complete' if not short and not n_dupk else 'BROKEN'}, "
          f"ret mismatches {n_num}+{n_pat}, bad bars {n_ohlc}, "
          f"volume dropouts {len(drop)} ({len(drop) - len(new_drop)} declared)")

    # cross-source: our cumulative return vs Sharadar's own closeadj series.
    # An independent construction path — a systematic error in ours (a missed
    # dividend basis, the AT&T class) shows as persistent per-ticker drift.
    worst, over = [], 0
    for t in sharadar_tickers():
        g = p[(p["ticker"] == t) & p["tradeable"]].sort_values("date")
        raw = sharadar_prices.load(t)
        if raw is None or not len(raw) or not len(g):
            bad.append(f"prices: no data to compare for {t}")
            continue
        ours = float((1 + g["ret"].fillna(0)).prod())
        adj = raw.sort_values("date")["closeadj"].astype(float)
        theirs = float(adj.iloc[-1] / adj.iloc[0])
        rel = abs(ours / theirs - 1)
        worst.append((rel, t))
        if rel > 0.02:
            over += 1
    worst.sort(reverse=True)
    if worst:
        print(f"  prices     : cumulative TR vs vendor closeadj — "
              f"worst {worst[0][1]} {worst[0][0]:.2%}, >2% on {over} of "
              f"{len(worst)}")
    if over:
        bad.append(f"prices: {over} tickers drift >2% from the vendor TR series "
                   f"({[t for r, t in worst[:4]]})")

    # cross-vendor: daily returns and volume vs yfinance, PER TICKER — a
    # pooled rate would let a short-history ticker be entirely wrong. Single
    # days differ at a low rate (yfinance has its own defects); only a high
    # rate for one ticker points at us. Volume is judged against the ticker's
    # own vendor ratio, so whole-series conventions pass; days on or near a
    # typed split/spin-off are vendor adjustment boundaries and are skipped.
    n_cmp = n_off = 0
    rates, vol_new, n_volflag = [], [], 0
    for t in sharadar_tickers():
        y = yfinance_prices.load(t)
        if y is None or not len(y):
            continue
        idx = pd.to_datetime(y["Date"]).dt.tz_localize(None).dt.normalize()
        yret = ((y["Close"] + y["Dividends"]) / y["Close"].shift(1) - 1)
        yret.index = idx.astype("datetime64[us]")
        yvol = pd.Series(y["Volume"].values, index=yret.index, dtype=float)
        g = p[(p["ticker"] == t) & p["tradeable"]].set_index("date")
        j = pd.concat([g["ret"], yret], axis=1, keys=["ours", "yf"],
                      sort=True).dropna()
        k = int(((j["ours"] - j["yf"]).abs() > 0.005).sum())
        n_cmp += len(j)
        n_off += k
        if len(j):
            rates.append((k / len(j), t))
        jv = pd.concat([g["volume"], yvol], axis=1, keys=["ours", "yf"],
                       sort=True).dropna()
        jv = jv[jv["yf"] > 0]
        if len(jv):
            r = jv["ours"] / jv["yf"]
            dev = r / r.median()
            flagged = dev[(dev < 0.25) | (dev > 4.0)]
            if len(flagged):
                bounds = {pd.Timestamp(a["date"])
                          for a in sharadar_actions.load(t)
                          if a["action"] in ("split", "spinoff")}
                for day in flagged.index:
                    if any(abs((day - s).days) <= 5 for s in bounds):
                        continue
                    if (t, str(day.date())) in KNOWN_YF_VOLUME_DAYS:
                        continue
                    n_volflag += 1
                    if not _declared_volume(t, day):
                        vol_new.append((t, str(day.date())))
    if n_cmp == 0:
        bad.append("prices: no yfinance days compared")
    rates.sort(reverse=True)
    top = rates[0] if rates else (0.0, "-")
    print(f"  prices     : vs yfinance daily returns — {n_cmp:,} compared, "
          f"{n_off:,} differ >0.5%, worst ticker {top[1]} {top[0]:.3%}")
    if top[0] > 0.006:
        bad.append(f"prices: {top[1]} disagrees with yfinance on {top[0]:.2%} "
                   f"of days (ceiling 0.6%)")
    print(f"  prices     : vs yfinance volume — {n_volflag} days out of line "
          f"({n_volflag - len(vol_new)} declared), new: {vol_new[:4] or 'none'}")
    if vol_new:
        bad.append(f"prices: NEW volume disagreements {vol_new[:5]}")

    # cross-source: the split column against the typed actions table. Funds
    # carry splits from the same yfinance response, so only equities have an
    # independent record to compare.
    split_bad = []
    for t in sharadar_tickers():
        g = p[(p["ticker"] == t) & p["tradeable"]].sort_values("date")
        days = pd.DatetimeIndex(g["date"])
        expected = {}
        for a in sharadar_actions.load(t):
            if a["action"] != "split" or a.get("value") is None:
                continue
            d = pd.Timestamp(a["date"])
            if not len(days) or d < days[0] or d > days[-1]:
                continue
            d = days[days.searchsorted(d)]
            expected[d] = expected.get(d, 1.0) * float(a["value"])
        panel_ev = {r.date: r.split for r in
                    g[g["split"] != 1.0].itertuples()}
        for d, v in expected.items():
            if abs(panel_ev.get(d, 1.0) - v) > 1e-9:
                split_bad.append((t, str(d.date())))
        for d in panel_ev:
            if d not in expected:
                split_bad.append((t, str(d.date())))
    print(f"  prices     : split column vs actions table — "
          f"{'all match' if not split_bad else f'MISMATCH {split_bad[:4]}'}")
    if split_bad:
        bad.append(f"prices: split column disagrees with actions {split_bad[:5]}")

    # every build-time correction must still be NEEDED: if the vendor repairs
    # a cell upstream, the stale patch must be removed, not silently reapplied
    stale = []
    for t, a, b in price_dataset.VOLUME_FIXES:
        raw = sharadar_prices.load(t)
        rv = pd.Series(raw["volume"].astype(float).values,
                       index=pd.to_datetime(raw["date"]).astype("datetime64[us]"))
        g = p[p["ticker"] == t].set_index("date")["volume"]
        win = rv.index[(rv.index >= pd.Timestamp(a)) & (rv.index <= pd.Timestamp(b))]
        if not len(win) or float(((g.reindex(win) - rv.loc[win]).abs()
                                  / rv.loc[win].clip(lower=1.0)).max()) < 0.01:
            stale.append(("volume", t, a))
    for t, dstr in price_dataset.BAR_FIXES:
        raw = sharadar_prices.load(t)
        raw = raw.set_index(pd.to_datetime(raw["date"]).astype("datetime64[us]"))
        d = pd.Timestamp(dstr)
        pr = p[(p["ticker"] == t) & (p["date"] == d)]
        if d not in raw.index or not len(pr):
            stale.append(("bar", t, dstr))
            continue
        # still needed iff ANY of the four price fields still disagrees
        rel = max(abs(float(pr[c].iloc[0]) - float(raw.loc[d, c]))
                  / max(float(raw.loc[d, c]), 1e-9)
                  for c in ("open", "high", "low", "close"))
        if rel < 0.001:
            stale.append(("bar", t, dstr))
    for t, dstr, v in price_dataset.VOLUME_RESTORED:
        raw = sharadar_prices.load(t)
        rv = pd.Series(raw["volume"].astype(float).values,
                       index=pd.to_datetime(raw["date"]).astype("datetime64[us]"))
        d = pd.Timestamp(dstr)
        if d not in rv.index or float(rv[d]) > 0.5 * v:
            stale.append(("volume-restored", t, dstr))
    print(f"  prices     : build corrections — "
          f"{len(price_dataset.VOLUME_FIXES)} volume + "
          f"{len(price_dataset.VOLUME_RESTORED)} restored + "
          f"{len(price_dataset.BAR_FIXES)} bars, "
          f"stale: {stale or 'none'}")
    if stale:
        bad.append(f"prices: stale corrections no longer needed {stale} — "
                   f"remove them from price_dataset")

    # conservation: every cash event inside a ticker's traded span must reach
    # the panel — a silently dropped dividend understates returns plausibly.
    lost = []
    for t in sharadar_tickers():
        g = p[(p["ticker"] == t) & p["tradeable"]]
        lo, hi = g["date"].min(), g["date"].max()
        expected = sum(float(a["value"]) for a in sharadar_actions.load(t)
                       if a["action"] in ("dividend", "spinoffdividend")
                       and a.get("value") is not None
                       and lo <= pd.Timestamp(a["date"]) <= hi)
        got = float(g["dividend"].sum() + g["spinoff_value"].sum())
        if abs(got - expected) > 1e-6 * max(expected, 1.0):
            lost.append((t, round(expected - got, 4)))
    print(f"  prices     : cash conservation vs the actions table — "
          f"{'exact for all 108' if not lost else f'LOST {lost[:5]}'}")
    if lost:
        bad.append(f"prices: cash lost between actions and panel: {lost[:5]}")

    # funds: our TR vs Yahoo's own Adj Close ratio
    off = []
    for t in fund_tickers():
        g = p[(p["ticker"] == t) & p["tradeable"]].sort_values("date")
        raw = yfinance_prices.load(t)
        if raw is None or not len(raw) or not len(g):
            bad.append(f"prices: no fund data to compare for {t}")
            continue
        ours = float((1 + g["ret"].fillna(0)).prod())
        theirs = float(raw["Adj Close"].iloc[-1] / raw["Adj Close"].iloc[0])
        if abs(ours / theirs - 1) > 0.005:
            off.append(t)
    print(f"  prices     : fund TR vs yahoo Adj Close — "
          f"{len(fund_tickers()) - len(off)}/{len(fund_tickers())} within 0.5%")
    if off:
        bad.append(f"prices: fund TR off vs yahoo: {off}")
    return bad


# ── macro ────────────────────────────────────────────────────────────────

def check_macro() -> list:
    bad = []
    m = macro_dataset.load().set_index("date")
    cal = calendar.load()["date"]
    if len(m) != len(cal):
        bad.append(f"macro: {len(m)} rows != calendar {len(cal)}")

    # lags tested in BOTH directions on days the series moved: the value must
    # come from t-lag, and must NOT equal today's not-yet-published figure.
    for sid, lag in DAILY_LAGS.items():
        raw = _daily(sid).set_index("date")["value"]
        col = m[sid.lower()]
        wrong = leak = tested = 0
        idx = m.index
        for i in range(300, len(idx)):
            t, src = idx[i], idx[i - lag]
            if t not in raw.index or src not in raw.index:
                continue
            if abs(raw[t] - raw[src]) <= 1e-12:
                continue                    # candidates equal: cannot discriminate
            v = col.iloc[i]
            if pd.isna(v):
                continue
            tested += 1
            if abs(v - raw[src]) > 1e-12:
                wrong += 1
            if abs(v - raw[t]) <= 1e-12:
                leak += 1
        print(f"  macro      : {sid} lag {lag} — {tested:,} moving days, "
              f"wrong-source {wrong}, same-day leak {leak}")
        if wrong or leak:
            bad.append(f"macro: {sid} lag broken (wrong {wrong}, leak {leak})")

    # the revised series rebuilt from the raw ALFRED vintages by an
    # independent walk: a release is served from its realtime_start day
    # (8:30am, public before the close) only while its observation period is
    # the newest known; the value carries forward until the next accepted
    # release. Every day must match — this covers both directions at once,
    # since the reconstruction IS the point-in-time truth.
    for sid in ("CPIAUCSL", "UNRATE", "GDP", "GDPC1"):
        obs = sorted((pd.Timestamp(o["realtime_start"]), pd.Timestamp(o["date"]),
                      float(o["value"]))
                     for o in fred_macro.load(sid) if o["value"] != ".")
        newest = pd.Timestamp.min
        ev_dates, ev_vals = [], []
        for rs, period, val in obs:
            if period >= newest:
                newest = period
                ev_dates.append(rs)
                ev_vals.append(val)
        pos = pd.DatetimeIndex(ev_dates).searchsorted(m.index, side="right") - 1
        exp = pd.Series([ev_vals[i] if i >= 0 else np.nan for i in pos],
                        index=m.index)
        col = m[sid.lower()]
        mism = int(((exp - col).abs() > 1e-9).sum()
                   + (exp.isna() != col.isna()).sum())
        print(f"  macro      : {sid} vs vintage walk — "
              f"{len(ev_dates)} releases, mismatching days {mism}")
        if mism:
            bad.append(f"macro: {sid} disagrees with the vintage walk on "
                       f"{mism} days")

    # the derived year-over-year columns, rebuilt independently: at every
    # accepted release, the headline against the SAME vintage's year-ago
    # value. Must match the stored column on every day.
    for col, (sid, months, diff) in DERIVED.items():
        obs = [(pd.Timestamp(o["realtime_start"]), pd.Timestamp(o["realtime_end"]),
                pd.Timestamp(o["date"]), float(o["value"]))
               for o in fred_macro.load(sid) if o["value"] != "."]
        obs.sort()
        newest = pd.Timestamp.min
        ev_dates, ev_vals = [], []
        for rs, until, period, v in obs:
            if period < newest:
                continue
            newest = period
            target = period - pd.DateOffset(months=months)
            prior = next((pv for prs, pu, pp, pv in obs
                          if pp == target and prs <= rs <= pu), None)
            val = (np.nan if prior is None
                   else v - prior if diff else v / prior - 1)
            ev_dates.append(rs)
            ev_vals.append(val)
        pos = pd.DatetimeIndex(ev_dates).searchsorted(m.index, side="right") - 1
        exp = pd.Series([ev_vals[i] if i >= 0 else np.nan for i in pos],
                        index=m.index)
        colv = m[col]
        mism = int(((exp - colv).abs() > 1e-9).sum()
                   + (exp.isna() != colv.isna()).sum())
        print(f"  macro      : {col} vs independent vintage walk — "
              f"mismatching days {mism}")
        if mism:
            bad.append(f"macro: {col} disagrees with the vintage walk on "
                       f"{mism} days")

    # derived anchors against the outside world. The June-2022 CPI print was
    # the 9.0% peak, released 2022-07-13; real yoy growth outside the COVID
    # comparison window never left [-4.5%, +6%] (the +5.6% edge is the
    # 2021-Q4 recovery figure, served through 2022).
    if pd.Timestamp("2022-07-13") in m.index:
        v = float(m.loc["2022-07-13", "cpi_yoy"])
        eve = float(m.loc["2022-07-12", "cpi_yoy"])
        print(f"  macro      : cpi_yoy peak anchor {v:.4f} (0.0900), "
              f"eve {eve:.4f}")
        if abs(v - 0.0900) > 0.003:
            bad.append(f"macro: cpi_yoy on 2022-07-13 is {v:.4f}, not the peak print")
        if abs(eve - 0.0900) < 0.001:
            bad.append("macro: the 9.0% CPI print visible the day BEFORE release")
    g = m["gdpc1_yoy"]
    ex = g[(m.index < "2020-03-01") | (m.index > "2022-12-31")].dropna()
    print(f"  macro      : gdpc1_yoy ex-COVID range [{ex.min():+.4f}, "
          f"{ex.max():+.4f}] (a base-year re-basing would read >+0.06)")
    if ex.max() > 0.06 or ex.min() < -0.05:
        bad.append(f"macro: gdpc1_yoy out of the honest range "
                   f"[{ex.min():+.4f}, {ex.max():+.4f}] — a re-based vintage "
                   f"is leaking through as growth")

    # first-release anchor: real GDP 2015-Q2 advance estimate, published
    # 2015-07-30, was 16,270.4; the revised figure today is ~18,700
    if pd.Timestamp("2015-07-30") in m.index:
        v = float(m.loc["2015-07-30", "gdpc1"])
        eve = float(m.loc["2015-07-29", "gdpc1"])
        print(f"  macro      : gdpc1 first-release anchor {v:,.1f} "
              f"(16,270.4), eve {eve:,.1f}")
        if abs(v - 16270.4) > 0.1:
            bad.append(f"macro: gdpc1 on 2015-07-30 is {v}, not the first release")
        if abs(eve - 16270.4) < 0.1:
            bad.append("macro: 2015-Q2 GDP visible the day BEFORE release")
    else:
        print("  macro      : gdpc1 2015 anchor outside the window — skipped, "
              "the vintage walk above still covers the series")

    holes = [c for c in m.columns
             if m[c].loc[m[c].first_valid_index():].isna().any()]
    if holes:
        bad.append(f"macro: interior NaN in {holes}")

    # cross-source: FRED's VIXCLS vs the independent yfinance ^VIX close.
    # A NEW disagreement means a source changed history; a declared one
    # disappearing means the declaration is stale.
    y = yfinance_prices.load("^VIX")
    yy = pd.Series(y["Close"].values,
                   index=pd.to_datetime(y["Date"]).dt.tz_localize(None)
                   .dt.normalize().astype("datetime64[us]"))
    fr = _daily("VIXCLS").set_index("date")["value"]
    j = pd.concat([fr, yy], axis=1, keys=["fred", "yahoo"],
                  sort=True).dropna()
    d = (j["fred"] - j["yahoo"]).abs()
    seen = {str(x.date()) for x in d[d > 0.5].index}
    new = sorted(seen - KNOWN_VIX_DAYS)
    gone = sorted(KNOWN_VIX_DAYS - seen)
    print(f"  macro      : VIX vs yfinance ^VIX — {len(j):,} days, "
          f"{len(seen)} differ >0.5 (declared {len(KNOWN_VIX_DAYS)}), "
          f"new: {new or 'none'}")
    if new:
        bad.append(f"macro: NEW VIX cross-source disagreement on {new}")
    if gone:
        bad.append(f"macro: declared VIX days no longer disagree {gone} — "
                   f"update the declaration")
    return bad


# ── fundamentals ─────────────────────────────────────────────────────────

def check_fundamentals() -> list:
    bad = []
    f = fundamentals_dataset.load()
    n_exp = len(sharadar_tickers())
    n_co = f["ticker"].nunique()
    if n_co != n_exp:
        bad.append(f"fundamentals: {n_co}/{n_exp} companies")
    if (n := int((f["published"] < f["period_end"]).sum())):
        bad.append(f"fundamentals: {n} rows published before their period ended")
    if (n := int(f.duplicated(["ticker", "quarter", "published"]).sum())):
        bad.append(f"fundamentals: {n} duplicate (ticker,quarter,published) rows")

    for item in ("revenue", "assets", "netinc", "equity", "marketcap",
                 "sharesbas"):
        cov = int(f.groupby("ticker")[item].apply(lambda s: s.notna().any()).sum())
        if cov != n_exp:
            bad.append(f"fundamentals: {item} covers {cov}/{n_exp}")

    a = f[(f["ticker"] == "AAPL") & (f["quarter"] == "2024-09-30")]
    if len(a):
        v = float(a["revenue"].iloc[0])
        print(f"  fundamentals: {n_co}/{n_exp} companies, AAPL 2024-Q4 revenue "
              f"{v:,.0f} (10-K 94,930,000,000)")
        if abs(v - 94.93e9) > 1e6:
            bad.append(f"fundamentals: AAPL anchor {v:,.0f} != 94,930,000,000")
    else:
        bad.append("fundamentals: the AAPL 2024-Q4 anchor row is missing")

    # cross-table: every fundamentals ticker must exist in the price panel
    # under the same spelling, or joins drop that company silently (the
    # vendor answers for BRK.B while every other table says BRK-B).
    p = price_dataset.load()
    orphans = sorted(set(f["ticker"]) - set(p["ticker"]))
    print(f"  fundamentals: tickers missing from the price panel: "
          f"{orphans or 'none'}")
    if orphans:
        bad.append(f"fundamentals: tickers not in price panel: {orphans}")

    # the vendor strikes marketcap/price on the FILING DATE's close. Anchored
    # one day later, every valuation ratio absorbs the earnings reaction —
    # signed against the surprise, so the error looks like a value signal.
    # Gated on the MAX: today every one of ~9.3k filings anchors exactly, so
    # a single slipped filing must fail, not drown in a median.
    # The vendor's price field is 3-decimal rounded while the panel close is
    # full precision, so the per-filing test is absolute with a rounding
    # allowance; the median (in relative terms) catches a systematic slip.
    px = p.set_index(["date", "ticker"])["close"]
    s = f[f["price"].notna()]
    anchor = pd.Series([px.get((d, t)) for d, t in
                        zip(s["published"], s["ticker"])],
                       index=s.index, dtype=float)
    ab = (s["price"] - anchor).abs().dropna()
    rel = (s["price"] / anchor - 1).abs().dropna()
    if not len(rel):
        bad.append("fundamentals: the price-anchor comparison is empty")
    else:
        n_off = int((ab > 0.0006 + 0.002 * anchor.reindex(ab.index)).sum())
        print(f"  fundamentals: price vs close(published) — {len(rel):,} "
              f"filings, median rel {rel.median():.6f}, "
              f"off-anchor {n_off}")
        if rel.median() > 0.001:
            bad.append(f"fundamentals: anchor systematically off (median "
                       f"{rel.median():.4f}; next-day anchoring reads ~0.01)")
        if n_off:
            bad.append(f"fundamentals: {n_off} filings anchored off the "
                       f"filing-date close")

    # every share correction must still be NEEDED (the raw feed still 14x)
    # and must have LANDED (the served count matches the true neighbours)
    stale_c = []
    for t, q, cols, k in fundamentals_dataset.SHARE_CORRECTIONS:
        raw = [r for r in sharadar_fundamentals.load(t)
               if r.get("calendardate") == q]
        served = f[(f["ticker"] == t) & (f["quarter"] == pd.Timestamp(q))]
        if not raw or not len(served):
            stale_c.append((t, q, "row missing"))
            continue
        rawv = float(raw[-1][cols[0]])
        gotv = float(served[cols[0]].iloc[-1])
        if abs(gotv - rawv * k) > 1:
            stale_c.append((t, q, "correction not applied"))
        if abs(rawv * k - gotv) < 1 and abs(rawv - gotv) < 1:
            stale_c.append((t, q, "vendor repaired — remove the correction"))
    print(f"  fundamentals: share corrections — "
          f"{len(fundamentals_dataset.SHARE_CORRECTIONS)} declared, "
          f"problems: {stale_c or 'none'}")
    if stale_c:
        bad.append(f"fundamentals: share corrections broken {stale_c}")

    # quarter continuity in each ticker's TRADEABLE era: a multi-quarter hole
    # there carries stale values into live features. Pre-listing gaps cannot
    # reach a feature and are ignored. The known gaps are DECLARED BY NAME:
    # a count would let a new hole replace a known one unnoticed.
    first_day = p[p["tradeable"]].groupby("ticker")["date"].min()
    gaps = set()
    for t, g in f[f["quarter"] >= "2004-01-01"].groupby("ticker"):
        q = g["quarter"].drop_duplicates().sort_values().reset_index(drop=True)
        d = q.diff().dt.days
        for i in d[d > 135].index:
            if t in first_day.index and q[i] >= first_day[t]:
                gaps.add((t, str(q[i - 1].date()), str(q[i].date())))
    new_g = sorted(gaps - KNOWN_QUARTER_GAPS)
    gone = sorted(KNOWN_QUARTER_GAPS - gaps)
    print(f"  fundamentals: tradeable-era quarter gaps >135d: {len(gaps)} "
          f"({len(gaps) - len(new_g)} declared), new: {new_g or 'none'}")
    if new_g:
        bad.append(f"fundamentals: NEW quarter gaps {new_g}")
    if gone:
        bad.append(f"fundamentals: declared quarter gaps no longer present "
                   f"{gone} — update the declaration")
    return bad


FAMILIES = {"calendar": check_calendar, "prices": check_prices,
            "macro": check_macro, "fundamentals": check_fundamentals}


if __name__ == "__main__":
    console_utf8()
    ap = argparse.ArgumentParser(description="Verify the processed tables.")
    for name in FAMILIES:
        ap.add_argument(f"--{name}", action="store_true",
                        help=f"check {name} only")
    args = ap.parse_args()
    chosen = [n for n in FAMILIES if getattr(args, n)] or list(FAMILIES)

    failures = []
    for name in chosen:
        print(f"{name.upper()}")
        failures += FAMILIES[name]()
    print()
    if failures:
        print(f"FAILED - {len(failures)} check(s):")
        for b in failures:
            print(f"  {b}")
    else:
        print(f"all checks passed ({', '.join(chosen)})")
    raise SystemExit(1 if failures else 0)
