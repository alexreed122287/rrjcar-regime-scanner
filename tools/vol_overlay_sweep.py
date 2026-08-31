#!/usr/bin/env python3
"""
Is the volatility overlay robust, or did it win test 12 on one lucky setting?

Test 12 found that sizing positions inversely to a trailing volatility estimate beat the
shipped regime filter by +0.52 Sharpe. That is the largest improvement anything in this repo
has produced -- which is exactly why it should be distrusted. It was measured at ONE target
vol (15%), ONE exposure cap (1.0), ONE cost assumption (5 bps) and no rebalancing band. A
result that only exists at one point in a four-dimensional parameter space is not a finding,
it is a coincidence with good manners.

So: score the same overlay across the whole grid.

    target vol      10%, 12%, 15%, 20%, 25%
    exposure cap    1.0 (no leverage), 1.5 (levered -- reported separately, see below)
    costs           0, 5, 10, 20 bps per unit of exposure change
    deadband        0, 5, 10 pp -- do not rebalance until exposure drifts this far
    forecast        trail20 (rolling std), ewma94 -- neither uses the HMM at all

240 configurations. The questions, in order of how much they matter:

    1. Does the overlay beat the shipped regime filter across the grid, or only in patches?
    2. Does it ever beat buy-and-hold on Sharpe? (Test 12 said no at one setting.)
    3. How fast does it decay in transaction costs? A 2.0-turnover strategy should be robust,
       but that must be shown, not assumed.
    4. Does a rebalancing deadband buy anything?

**On leverage:** a cap above 1.0 lets a strategy improve its return by taking more risk, which
is not the same as being better. Levered configurations are scored but held apart from the
headline count, and their exposure is reported alongside, so nothing can win quietly.

Method: one walk-forward pass fits the HMM once per (ticker, window) and saves the forward
returns, both calibrated log-vol forecasts, and the filter's 0/1 exposure path. Every
configuration is then re-scored from those saved paths, so the grid costs one HMM pass rather
than 240. Forecasts are calibrated on training bars only, as in tests 11-13.

Usage:
    python tools/vol_overlay_sweep.py --tickers SPY --save-obs sw/SPY.csv
    python tools/vol_overlay_sweep.py --from-obs 'sw/*.csv' --json docs/vol_overlay_sweep.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from data_loader import fetch_data, engineer_features
from hmm_engine import RegimeDetector, regime_sets
from tools.drawdown_benchmark import _metrics, _paired_bootstrap
from tools.vol_forecast_shootout import (
    EPS, _apply_linear, _ewma_vol, _fit_linear, _fwd_vol, _trail_vol,
)
from tools.vol_sizing import sized_exposure

DEFAULT_TICKERS = ["SPY", "QQQ", "NVDA", "AAPL", "XLF"]
HORIZON = 5

TARGET_VOLS = [0.10, 0.12, 0.15, 0.20, 0.25]
CAPS = [1.0, 1.5]
COSTS = [0.0, 5.0, 10.0, 20.0]
DEADBANDS = [0.0, 0.05, 0.10]
FORECASTS = ["trail20", "ewma94"]


def apply_deadband(target: np.ndarray, deadband: float) -> np.ndarray:
    """Hold the current position until desired exposure drifts more than `deadband`.

    This is the standard practitioner fix for turnover in a vol-targeting overlay. With
    deadband=0 it is the identity, so the no-band column of the sweep reproduces test 12.
    """
    if deadband <= 0:
        return np.asarray(target, dtype=float)
    out = np.empty(len(target), dtype=float)
    held = 0.0
    for i, want in enumerate(np.asarray(target, dtype=float)):
        if not np.isfinite(want):
            want = 0.0
        if abs(want - held) > deadband:
            held = want
        out[i] = held
    return out


def collect(tickers, period_days, interval, n_regimes, train_bars, oos_bars, step_bars,
            n_iter):
    """One HMM pass. Saves forward returns and forecasts, NOT metrics."""
    rows = []
    for tkr in tickers:
        raw = fetch_data(tkr, period_days=period_days, interval=interval)
        if raw is None or raw.empty:
            print(f"  {tkr}: no data, skipped", file=sys.stderr)
            continue
        feats = engineer_features(raw)
        rets = np.asarray(feats["returns"].values, dtype=float)
        fwd_ret = np.full(len(rets), np.nan)
        fwd_ret[:-1] = rets[1:]

        fwd = _fwd_vol(rets, HORIZON)
        log_fwd = np.log(np.clip(fwd, EPS, None))
        log_fwd[~np.isfinite(fwd)] = np.nan
        preds = {"trail20": np.log(np.clip(_trail_vol(rets, 20), EPS, None)),
                 "ewma94": np.log(np.clip(_ewma_vol(rets), EPS, None))}
        for k in preds:
            preds[k][~np.isfinite(preds[k])] = np.nan

        n = len(feats)
        starts = list(range(train_bars, n - oos_bars + 1, step_bars))
        if not starts:
            print(f"  {tkr}: only {n} bars, too short", file=sys.stderr)
            continue
        print(f"  {tkr}: {n} bars, {len(starts)} windows", file=sys.stderr)

        for w, oos_start in enumerate(starts):
            oos_end = oos_start + oos_bars
            tr, oos = slice(0, oos_start), slice(oos_start, oos_end)
            f_ret = fwd_ret[oos]
            if np.isfinite(f_ret).sum() < oos_bars * 0.9:
                continue
            f_ret = np.nan_to_num(f_ret, nan=0.0)

            det = RegimeDetector(n_regimes=n_regimes, n_iter=n_iter)
            try:
                det.train(feats.iloc[:oos_start])
                labeled = det.filtered_regimes(feats.iloc[:oos_end])
            except Exception as exc:
                print(f"    {tkr} w{w}: fit failed ({type(exc).__name__}), skipped",
                      file=sys.stderr)
                continue
            oos_ids = np.asarray(labeled["regime_id"].values[oos], dtype=float)
            bull = set(regime_sets(int(det.n_regimes))["bullish"])
            y_tr = log_fwd[tr]

            fc = {}
            for name in FORECASTS:
                coef = _fit_linear(preds[name][tr].reshape(-1, 1), y_tr)
                fc[name] = _apply_linear(coef, preds[name][oos].reshape(-1, 1))
            if any(not np.isfinite(v).any() for v in fc.values()):
                continue

            for j in range(len(f_ret)):
                rows.append({
                    "ticker": tkr, "window": w, "bar": int(oos_start + j),
                    "fwd_ret": float(f_ret[j]),
                    "f_trail20": float(fc["trail20"][j]),
                    "f_ewma94": float(fc["ewma94"][j]),
                    "filter_expo": 1.0 if (np.isfinite(oos_ids[j])
                                           and int(oos_ids[j]) in bull) else 0.0,
                })
    return pd.DataFrame(rows)


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Re-score every configuration from the saved paths."""
    rows = []
    for (tkr, w), g in df.groupby(["ticker", "window"], sort=True):
        g = g.sort_values("bar")
        f_ret = g["fwd_ret"].values.astype(float)
        for cost in COSTS:
            m = _metrics(np.ones(len(f_ret)), f_ret, cost)
            m.update({"ticker": tkr, "window": w, "strategy": "buy_hold",
                      "cost_bps": cost, "target_vol": np.nan, "cap": np.nan,
                      "deadband": np.nan, "forecast": "-"})
            rows.append(m)
            m = _metrics(g["filter_expo"].values.astype(float), f_ret, cost)
            m.update({"ticker": tkr, "window": w, "strategy": "hmm_filter",
                      "cost_bps": cost, "target_vol": np.nan, "cap": np.nan,
                      "deadband": np.nan, "forecast": "-"})
            rows.append(m)
        for fname in FORECASTS:
            fc = g[f"f_{fname}"].values.astype(float)
            for tv in TARGET_VOLS:
                for cap in CAPS:
                    raw_e = sized_exposure(fc, tv, cap)
                    for db in DEADBANDS:
                        e = apply_deadband(raw_e, db)
                        # THE control. Test 10's lesson: a strategy that reduces drawdown by
                        # holding less must be compared against simply holding less, at the
                        # same average exposure, with no timing at all. Without this, any
                        # de-risking rule looks skilful.
                        const = np.full(len(e), float(np.mean(e)))
                        for cost in COSTS:
                            for label, path in (("overlay", e), ("const_matched", const)):
                                m = _metrics(path, f_ret, cost)
                                m.update({"ticker": tkr, "window": w, "strategy": label,
                                          "cost_bps": cost, "target_vol": tv, "cap": cap,
                                          "deadband": db, "forecast": fname})
                                rows.append(m)
    return pd.DataFrame(rows)


def _config_key(r) -> str:
    return f"{r.forecast}|tv{r.target_vol:.2f}|cap{r.cap:.1f}|db{r.deadband:.2f}"


def evaluate(scored: pd.DataFrame) -> dict:
    ov = scored[scored.strategy == "overlay"].copy()
    ov["config"] = [_config_key(r) for r in ov.itertuples()]
    cm = scored[scored.strategy == "const_matched"].copy()
    cm["config"] = [_config_key(r) for r in cm.itertuples()]
    out = {"configs": [], "n_windows": int(scored.groupby(["ticker", "window"]).ngroups)}

    for (cfg, cost), g in ov.groupby(["config", "cost_bps"], sort=True):
        fc, tv, cap, db = cfg.split("|")
        ref = scored[(scored.cost_bps == cost) & (scored.strategy.isin(
            ["hmm_filter", "buy_hold"]))]
        mine = cm[(cm.config == cfg) & (cm.cost_bps == cost)]
        joined = pd.concat([g.assign(strategy="overlay"), mine, ref], ignore_index=True)
        rec = {
            "config": cfg, "forecast": fc, "target_vol": float(tv[2:]),
            "cap": float(cap[3:]), "deadband": float(db[2:]), "cost_bps": float(cost),
            "return": round(float(g.total_return.mean()), 5),
            "max_drawdown": round(float(g.max_drawdown.mean()), 5),
            "sharpe": round(float(g.sharpe.mean()), 4),
            "vol": round(float(g.vol_ann.mean()), 4),
            "exposure": round(float(g.exposure.mean()), 4),
            "turnover": round(float(g.turnover.mean()), 2),
        }
        rec["const_matched_sharpe"] = round(float(mine.sharpe.mean()), 4)
        rec["const_matched_maxdd"] = round(float(mine.max_drawdown.mean()), 5)
        for opp in ("hmm_filter", "buy_hold", "const_matched"):
            for metric in ("sharpe", "max_drawdown", "total_return"):
                t = _paired_bootstrap(joined, metric, "overlay", opp)
                if not t.get("usable"):
                    rec[f"vs_{opp}_{metric}"] = {"diff": None, "ci": None, "sig": None}
                    continue
                lo, hi = float(t["ci_low"]), float(t["ci_high"])
                # _paired_bootstrap reports no significance flag of its own; a difference
                # counts as significant when the bootstrap interval excludes zero, which is
                # the same rule tests 9-13 use.
                rec[f"vs_{opp}_{metric}"] = {
                    "diff": round(float(t["mean_diff"]), 5),
                    "ci": [round(lo, 5), round(hi, 5)],
                    "sig": bool(lo > 0 or hi < 0),
                    "frac_better": t.get("frac_a_better"),
                }
        out["configs"].append(rec)

    out["reference_by_cost"] = {
        str(int(c)): {s: round(float(d.sharpe.mean()), 4)
                      for s, d in gg[gg.strategy.isin(["hmm_filter", "buy_hold"])
                                     ].groupby("strategy")}
        for c, gg in scored.groupby("cost_bps")}
    ref5 = scored[(scored.cost_bps == 5.0)]
    out["reference_5bps"] = {
        s: {"return": round(float(d.total_return.mean()), 5),
            "max_drawdown": round(float(d.max_drawdown.mean()), 5),
            "sharpe": round(float(d.sharpe.mean()), 4),
            "exposure": round(float(d.exposure.mean()), 4),
            "turnover": round(float(d.turnover.mean()), 2)}
        for s, d in ref5[ref5.strategy != "overlay"].groupby("strategy")}
    return out


def report(res: dict) -> None:
    cfgs = pd.DataFrame(res["configs"])
    unlev = cfgs[cfgs.cap == 1.0]
    lev = cfgs[cfgs.cap > 1.0]

    print("=" * 100)
    print("IS THE VOLATILITY OVERLAY ROBUST, OR DID IT WIN ON ONE SETTING?")
    print(f"{len(cfgs)} configurations, {res['n_windows']} walk-forward windows each")
    print("=" * 100)

    ref = res["reference_5bps"]
    print("\nReference strategies at 5 bps:")
    for s in ("hmm_filter", "buy_hold"):
        if s in ref:
            r = ref[s]
            print(f"  {s:<12} return {r['return']:>7.2%}  maxDD {r['max_drawdown']:>7.2%}  "
                  f"Sharpe {r['sharpe']:>5.2f}  expo {r['exposure']:>5.0%}  "
                  f"turn {r['turnover']:>5.1f}")

    def frac(d, col):
        beat = d[d[col].apply(lambda x: bool(x.get("sig")) and (x.get("diff") or 0) > 0)]
        return len(beat), len(d)

    print("\n" + "-" * 100)
    print("Q1: DOES IT BEAT THE SHIPPED REGIME FILTER ACROSS THE GRID?")
    print("-" * 100)
    for label, d in (("unlevered (cap 1.0)", unlev), ("levered (cap 1.5)", lev)):
        if d.empty:
            continue
        b, n = frac(d, "vs_hmm_filter_sharpe")
        worse = d[d["vs_hmm_filter_sharpe"].apply(
            lambda x: bool(x.get("sig")) and (x.get("diff") or 0) < 0)]
        diffs = d["vs_hmm_filter_sharpe"].apply(lambda x: x.get("diff"))
        print(f"  {label}: beats filter on Sharpe in {b}/{n} configs "
              f"({b / n:.0%}), significantly WORSE in {len(worse)}/{n}")
        print(f"    Sharpe range {d.sharpe.min():.2f} to {d.sharpe.max():.2f}  "
              f"(filter: {ref.get('hmm_filter', {}).get('sharpe', float('nan')):.2f})")
        print(f"    Sharpe advantage over filter: min {diffs.min():+.3f}, "
              f"median {diffs.median():+.3f}, max {diffs.max():+.3f}")

    print("\n" + "-" * 100)
    print("Q2: DOES IT EVER BEAT BUY-AND-HOLD ON SHARPE?")
    print("-" * 100)
    b, n = frac(unlev, "vs_buy_hold_sharpe")
    worse = unlev[unlev["vs_buy_hold_sharpe"].apply(
        lambda x: bool(x.get("sig")) and (x.get("diff") or 0) < 0)]
    print(f"  unlevered: significantly better in {b}/{n}, significantly WORSE in "
          f"{len(worse)}/{n}, indistinguishable in {n - b - len(worse)}/{n}")
    sh = unlev["vs_buy_hold_sharpe"].apply(lambda x: x.get("diff"))
    print(f"  Sharpe vs buy-and-hold: min {sh.min():+.3f}, median {sh.median():+.3f}, "
          f"max {sh.max():+.3f}")
    dd, _ = frac(unlev, "vs_buy_hold_max_drawdown")
    ddd = unlev["vs_buy_hold_max_drawdown"].apply(lambda x: x.get("diff"))
    print(f"  drawdown significantly shallower than buy-and-hold in {dd}/{n} configs")
    print(f"  drawdown saved vs buy-and-hold: min {ddd.min():+.2%}, "
          f"median {ddd.median():+.2%}, max {ddd.max():+.2%}")

    print("\n" + "-" * 100)
    print("Q2b: THE REAL CONTROL -- vs a CONSTANT position at the same average exposure")
    print("-" * 100)
    b, n = frac(unlev, "vs_const_matched_sharpe")
    worse = unlev[unlev["vs_const_matched_sharpe"].apply(
        lambda x: bool(x.get("sig")) and (x.get("diff") or 0) < 0)]
    sh = unlev["vs_const_matched_sharpe"].apply(lambda x: x.get("diff"))
    print(f"  Sharpe: significantly better in {b}/{n}, significantly WORSE in "
          f"{len(worse)}/{n}")
    print(f"    diff min {sh.min():+.3f}, median {sh.median():+.3f}, max {sh.max():+.3f}")
    bd, _ = frac(unlev, "vs_const_matched_max_drawdown")
    dw = unlev[unlev["vs_const_matched_max_drawdown"].apply(
        lambda x: bool(x.get("sig")) and (x.get("diff") or 0) < 0)]
    dd = unlev["vs_const_matched_max_drawdown"].apply(lambda x: x.get("diff"))
    print(f"  max drawdown: significantly SHALLOWER in {bd}/{n}, significantly deeper in "
          f"{len(dw)}/{n}")
    print(f"    diff min {dd.min():+.2%}, median {dd.median():+.2%}, max {dd.max():+.2%}")
    print("\n  NOTE on the Sharpe row: Sharpe is scale-invariant, so a constant-exposure")
    print("  control has the same Sharpe as buy-and-hold by construction (up to cost drag).")
    print("  That row is therefore identical to Q2 and carries no extra information. The")
    print("  informative comparison against this control is DRAWDOWN, which is not")
    print("  scale-invariant: if the overlay cannot produce a shallower drawdown than a")
    print("  constant position holding the same average exposure, then its drawdown")
    print("  reduction is bought by holding less, not by timing volatility at all.")

    print("\nHow the filter's own Sharpe depends on what you charge it:")
    print(f"    {'cost bps':>10}{'hmm_filter':>13}{'buy_hold':>11}")
    for c, v in sorted(res["reference_by_cost"].items(), key=lambda kv: int(kv[0])):
        print(f"    {c:>10}{v.get('hmm_filter', float('nan')):>13.2f}"
              f"{v.get('buy_hold', float('nan')):>11.2f}")

    print("\n" + "-" * 100)
    print("Q3: SENSITIVITY -- mean Sharpe of unlevered configs")
    print("-" * 100)
    for dim, name in (("target_vol", "target vol"), ("cost_bps", "cost (bps)"),
                      ("deadband", "deadband"), ("forecast", "forecast")):
        piv = unlev.groupby(dim).agg(sharpe=("sharpe", "mean"),
                                     ret=("return", "mean"),
                                     dd=("max_drawdown", "mean"),
                                     turn=("turnover", "mean"))
        print(f"\n  by {name}:")
        print(f"    {'value':>10}{'Sharpe':>9}{'return':>9}{'maxDD':>9}{'turnover':>10}")
        for k, r in piv.iterrows():
            kk = f"{k:.2f}" if isinstance(k, float) else str(k)
            print(f"    {kk:>10}{r.sharpe:>9.2f}{r.ret:>9.2%}{r.dd:>9.2%}{r.turn:>10.1f}")

    print("\n" + "=" * 100)
    b1, n1 = frac(unlev, "vs_hmm_filter_sharpe")
    b2, _ = frac(unlev, "vs_buy_hold_sharpe")
    b3, _ = frac(unlev, "vs_const_matched_sharpe")
    d1 = unlev["vs_hmm_filter_sharpe"].apply(lambda x: x.get("diff"))
    pos = int((d1 > 0).sum())
    if pos == n1 and b1 < n1:
        print(f"Q1 ANSWER: the SIGN is robust -- the overlay's Sharpe exceeds the filter's in "
              f"{pos}/{n1}")
        print(f"configurations, is significantly better in {b1}/{n1}, and is never "
              f"significantly worse.")
    elif b1 == n1:
        print(f"Q1 ANSWER: robust. The overlay beats the shipped filter on Sharpe in ALL "
              f"{n1} unlevered")
        print("configurations -- every target vol, every cost level, with and without a "
              "rebalancing band.")
    elif b1 >= 0.8 * n1:
        print(f"Q1 ANSWER: largely robust -- beats the filter in {b1}/{n1} unlevered configs.")
    else:
        print(f"Q1 ANSWER: NOT robust -- beats the filter in only {b1}/{n1} configs. "
              "Test 12's result")
        print("was setting-dependent and should not be relied on.")
    if b2 == 0:
        print("Q2 ANSWER: no. It never beats buy-and-hold on Sharpe. This is a "
              "drawdown-reduction")
        print("overlay, not an edge, and it must not be described as one.")
    else:
        print(f"Q2 ANSWER: beats buy-and-hold on Sharpe in {b2}/{n1} configs -- inspect these.")
    print("=" * 100)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--period-days", type=int, default=3000)
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--regimes", type=int, default=7)
    ap.add_argument("--train-bars", type=int, default=756)
    ap.add_argument("--oos-bars", type=int, default=126)
    ap.add_argument("--step-bars", type=int, default=126)
    ap.add_argument("--n-iter", type=int, default=50)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--save-obs", default=None)
    ap.add_argument("--from-obs", nargs="+", default=None)
    args = ap.parse_args()

    if args.from_obs:
        files = sorted({f for pat in args.from_obs for f in glob.glob(pat)} or set(args.from_obs))
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        print(f"Pooled {len(df)} bars from {len(files)} file(s): "
              f"{sorted(df.ticker.unique())}", file=sys.stderr)
    else:
        df = collect(args.tickers, args.period_days, args.interval, args.regimes,
                     args.train_bars, args.oos_bars, args.step_bars, args.n_iter)
        if args.save_obs and not df.empty:
            Path(args.save_obs).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.save_obs, index=False)
            print(f"Wrote {len(df)} bars to {args.save_obs}", file=sys.stderr)

    if df.empty:
        print("No observations collected.", file=sys.stderr)
        sys.exit(1)

    res = evaluate(score(df))
    report(res)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(res, indent=2, default=str))
        print(f"\nSaved: {args.json_out}")


if __name__ == "__main__":
    main()
