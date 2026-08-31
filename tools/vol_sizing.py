#!/usr/bin/env python3
"""
Test 10 killed the in/out filter. Is SIZING off predicted volatility any better?

The logic of the last three tests points here. Test 9 found the model's only real output is a
weak forward-volatility signal. Test 10 found the binary bullish/bearish filter is worthless
-- indistinguishable from a coin flip at the same exposure, and beaten by a 200-day moving
average. Test 11 found the regime label is no better than a free EWMA as a vol forecast, and
adds nothing detectable on top of one. Sizing is the one remaining use of the model that
nobody has measured: instead of switching between fully invested and flat, hold a continuous
position inversely proportional to predicted volatility.

Two separate questions again, and only the second one is about the HMM:

    1. Is volatility-targeted SIZING better than the regime filter's in/out switching?
       (test 10 already hinted yes, using naive trailing vol -- this asks it properly with
       calibrated forecasts)
    2. Does using the REGIME-informed forecast beat using the free EWMA forecast?
       If not, the HMM contributes nothing here either, and the project is out of ideas.

Strategies, all targeting the same annualized volatility so they are comparable:

    size_ewma       exposure = target_vol / EWMA forecast
    size_ewma_hmm   exposure = target_vol / (EWMA + per-regime offset) forecast
    size_hmm        exposure = target_vol / per-regime mean forecast
    size_trail20    exposure = target_vol / calibrated trailing-20 forecast
    hmm_filter      the shipped behaviour: fully long in bullish regimes, else flat
    buy_hold        always fully long

Usage:
    python tools/vol_sizing.py --tickers SPY --save-obs vs/SPY.csv
    python tools/vol_sizing.py --from-obs 'vs/*.csv' --json docs/vol_sizing.json

Method notes
------------
* Forecast construction is imported wholesale from tools/vol_forecast_shootout.py so the two
  experiments cannot drift apart. Every calibration is fitted on training bars only.
* Exposure chosen at bar t earns bar t+1's return; 5 bps charged on every change in exposure.
  Costs matter more here than anywhere else in this repo, because continuous sizing adjusts
  every single bar while the filter only trades on regime flips. A cost-free comparison would
  badly flatter the sizers.
* Exposure is capped at ``--max-exposure`` (default 1.0, i.e. no leverage) so that any
  improvement cannot come from quietly taking more risk than buy-and-hold.
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

DEFAULT_TICKERS = ["SPY", "QQQ", "NVDA", "AAPL", "XLF"]
STRATEGIES = ["size_ewma", "size_ewma_hmm", "size_hmm", "size_trail20", "hmm_filter",
              "buy_hold"]
HORIZON = 5
ANNUALIZE = np.sqrt(252.0)


def sized_exposure(forecast: np.ndarray, target_vol: float, max_exposure: float) -> np.ndarray:
    """Exposure inversely proportional to predicted vol, capped.

    `forecast` is LOG annualized vol, matching what the forecast models emit. A non-finite
    forecast yields zero exposure rather than NaN, so a failed forecast sits in cash instead
    of silently propagating into the equity curve.
    """
    pv = np.exp(np.asarray(forecast, dtype=float))
    e = np.where(np.isfinite(pv) & (pv > EPS), target_vol / np.where(pv > EPS, pv, 1.0), 0.0)
    return np.clip(np.nan_to_num(e, nan=0.0), 0.0, max_exposure)


def collect(tickers, period_days, interval, n_regimes, train_bars, oos_bars, step_bars,
            n_iter, cost_bps, target_vol, max_exposure):
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
                trained = det.train(feats.iloc[:oos_start])
            except Exception as exc:
                print(f"    {tkr} w{w}: fit failed ({type(exc).__name__}), skipped",
                      file=sys.stderr)
                continue
            train_ids = np.asarray(trained["regime_id"].values, dtype=float)
            labeled = det.filtered_regimes(feats.iloc[:oos_end])
            oos_ids = np.asarray(labeled["regime_id"].values[oos], dtype=float)
            y_tr = log_fwd[tr]

            fc = {}
            for name in ("trail20", "ewma94"):
                coef = _fit_linear(preds[name][tr].reshape(-1, 1), y_tr)
                fc[name] = _apply_linear(coef, preds[name][oos].reshape(-1, 1))

            gm = float(np.nanmean(y_tr)) if np.isfinite(y_tr).any() else np.nan
            reg_mean = {}
            for rid in np.unique(train_ids[np.isfinite(train_ids)]):
                m = (train_ids == rid) & np.isfinite(y_tr)
                if m.sum() >= 20:
                    reg_mean[int(rid)] = float(np.mean(y_tr[m]))
            fc["hmm"] = np.array([reg_mean.get(int(r), gm) if np.isfinite(r) else gm
                                  for r in oos_ids])

            base_tr = _apply_linear(_fit_linear(preds["ewma94"][tr].reshape(-1, 1), y_tr),
                                    preds["ewma94"][tr].reshape(-1, 1))
            resid_tr = y_tr - base_tr
            off = {}
            for rid in np.unique(train_ids[np.isfinite(train_ids)]):
                m = (train_ids == rid) & np.isfinite(resid_tr)
                if m.sum() >= 20:
                    off[int(rid)] = float(np.mean(resid_tr[m]))
            fc["ewma_hmm"] = fc["ewma94"] + np.array(
                [off.get(int(r), 0.0) if np.isfinite(r) else 0.0 for r in oos_ids])

            def sized(forecast: np.ndarray) -> np.ndarray:
                return sized_exposure(forecast, target_vol, max_exposure)

            bull = set(regime_sets(int(det.n_regimes))["bullish"])
            paths = {
                "size_ewma": sized(fc["ewma94"]),
                "size_ewma_hmm": sized(fc["ewma_hmm"]),
                "size_hmm": sized(fc["hmm"]),
                "size_trail20": sized(fc["trail20"]),
                "hmm_filter": np.array([1.0 if (np.isfinite(r) and int(r) in bull) else 0.0
                                        for r in oos_ids]),
                "buy_hold": np.ones(len(f_ret)),
            }
            for name, e in paths.items():
                m = _metrics(e, f_ret, cost_bps)
                m.update({"ticker": tkr, "window": w, "strategy": name})
                rows.append(m)
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame) -> dict:
    pooled = {}
    for s, g in df.groupby("strategy"):
        pooled[s] = {k: round(float(g[k].mean()), 5) for k in
                     ["total_return", "max_drawdown", "vol_ann", "sharpe", "exposure",
                      "turnover"]}
        # Return per unit of realized vol, the metric sizing is supposed to improve.
        pooled[s]["return_per_vol"] = (
            round(pooled[s]["total_return"] / pooled[s]["vol_ann"], 3)
            if pooled[s]["vol_ann"] > 1e-9 else None)
    comps = {}
    for a, b in [("size_ewma_hmm", "size_ewma"), ("size_ewma", "hmm_filter"),
                 ("size_ewma", "buy_hold"), ("size_hmm", "size_ewma"),
                 ("size_ewma", "size_trail20")]:
        comps[f"{a}_vs_{b}"] = {m: _paired_bootstrap(df, m, a, b)
                                for m in ("sharpe", "max_drawdown", "total_return")}
    return {"n_rows": int(len(df)), "pooled": pooled, "comparisons": comps}


def report(res: dict) -> None:
    print("=" * 104)
    print("DOES VOLATILITY-TARGETED SIZING BEAT THE IN/OUT FILTER -- AND DOES THE HMM HELP?")
    print("=" * 104)
    print(f"\nPooled means across all windows and tickers ({res['n_rows']} strategy-windows):\n")
    print(f"  {'strategy':<16}{'return':>9}{'maxDD':>9}{'vol':>8}{'sharpe':>8}"
          f"{'expo':>7}{'turn':>8}{'ret/vol':>9}")
    for s in STRATEGIES:
        p = res["pooled"].get(s)
        if not p:
            continue
        rpv = p.get("return_per_vol")
        print(f"  {s:<16}{p['total_return']*100:>8.2f}%{p['max_drawdown']*100:>8.2f}%"
              f"{p['vol_ann']*100:>7.1f}%{p['sharpe']:>8.2f}{p['exposure']*100:>6.0f}%"
              f"{p['turnover']:>8.1f}{(f'{rpv:.2f}' if rpv is not None else '--'):>9}")

    print("\nPaired per-window differences, bootstrapped over windows.")
    print("Positive favours the FIRST strategy on every metric shown"
          " (maxDD is signed, so higher = shallower).\n")
    for name, block in res["comparisons"].items():
        print(f"  {name}")
        for metric, t in block.items():
            if not t.get("usable"):
                print(f"    {metric:<14} unusable")
                continue
            sig = ("significant" if (t["ci_low"] > 0 or t["ci_high"] < 0)
                   else "not significant")
            print(f"    {metric:<14} diff={t['mean_diff']:+.4f}  "
                  f"95% CI [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}]  "
                  f"first better in {t['frac_a_better']*100:.0f}% of {t['n_windows']}  ({sig})")
        print()

    add = res["comparisons"]["size_ewma_hmm_vs_size_ewma"]["sharpe"]
    beat = res["comparisons"]["size_ewma_vs_hmm_filter"]["sharpe"]
    print("=" * 104)
    print("Q1: is sizing better than the shipped in/out filter?")
    if beat.get("usable") and beat["ci_low"] > 0:
        print("   YES -- vol-targeted sizing beats the regime filter on Sharpe, significantly.")
    elif beat.get("usable") and beat["ci_high"] < 0:
        print("   NO -- the filter wins, which would be a genuine surprise.")
    else:
        print("   Not significantly, on this sample.")
    print("\nQ2: does the REGIME-informed forecast beat the free EWMA one?")
    if add.get("usable") and add["ci_low"] > 0:
        print("   YES -- the regime offset earns its keep in sizing. Worth building on.")
    elif add.get("usable") and add["ci_high"] < 0:
        print("   NO -- the regime offset makes sizing WORSE.")
    else:
        print("   NO -- no significant difference. Consistent with test 11: the HMM adds")
        print("   nothing to a free EWMA, so sizing does not rescue it either.")
    print("=" * 104)


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
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--target-vol", type=float, default=0.15)
    ap.add_argument("--max-exposure", type=float, default=1.0)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--save-obs", default=None)
    ap.add_argument("--from-obs", nargs="+", default=None)
    args = ap.parse_args()

    if args.from_obs:
        files = sorted({f for pat in args.from_obs for f in glob.glob(pat)} or set(args.from_obs))
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        print(f"Pooled {len(df)} rows from {len(files)} file(s): "
              f"{sorted(df.ticker.unique())}", file=sys.stderr)
    else:
        df = collect(args.tickers, args.period_days, args.interval, args.regimes,
                     args.train_bars, args.oos_bars, args.step_bars, args.n_iter,
                     args.cost_bps, args.target_vol, args.max_exposure)
        if args.save_obs and not df.empty:
            Path(args.save_obs).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.save_obs, index=False)
            print(f"Wrote {len(df)} rows to {args.save_obs}", file=sys.stderr)

    if df.empty:
        print("No results collected.", file=sys.stderr)
        sys.exit(1)

    res = evaluate(df)
    report(res)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(res, indent=2, default=str))
        print(f"\nSaved: {args.json_out}")


if __name__ == "__main__":
    main()
