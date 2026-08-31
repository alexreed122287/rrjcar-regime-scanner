#!/usr/bin/env python3
"""
Would purpose-built VOLATILITY features make the HMM add anything to a free vol forecast?

The last open question. Test 9 found the regimes carry a little forward-volatility
information. Test 11 found that information is redundant against a free EWMA. Test 12 found
that trying to size positions with it makes results significantly worse. But all three used
the shipped feature set -- ``returns``, ``range``, ``volume_change`` -- which are only
incidentally volatility proxies. The obvious objection is that the HMM was never given
volatility features to work with.

So: give it some, and re-run the only test that matters. Three feature sets, same walk-forward
harness, same question each time -- does adding a per-regime offset improve an EWMA forecast of
forward 5-bar volatility?

    baseline    returns, range, volume_change                        (shipped)
    vol         log EWMA vol, log 60-bar vol, vol-of-vol,
                downside-vol share, normalized range                 (purpose-built)
    vol_ret     the vol set plus returns                             (vol state + direction)

If the purpose-built sets do not beat the free EWMA either, the conclusion is not "wrong
features" but that a 7-state discrete latent variable is simply a poor container for a
quantity that is continuous, highly persistent and already well estimated by an exponential
average. That is a real finding about the architecture rather than the inputs.

Usage:
    python tools/vol_features.py --tickers SPY --save-obs fs/SPY.csv
    python tools/vol_features.py --from-obs 'fs/*.csv' --json docs/vol_features.json

Method notes
------------
* ``RegimeDetector`` already accepts ``feature_columns``, so no engine change is needed; this
  measures the existing model on different inputs.
* All features are strictly trailing and are computed on the full series before slicing, but
  every one uses only backward-looking windows, so bar t's value never depends on t+1.
* Fit failures are counted and reported rather than silently skipped -- richer feature sets
  make degenerate covariances more likely, and hiding that would flatter them.
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
from hmm_engine import RegimeDetector
from tools.vol_forecast_shootout import (
    EPS, _apply_linear, _dm_test, _decimate, _ewma_vol, _fit_linear, _fwd_vol, _losses,
    _trail_vol,
)

DEFAULT_TICKERS = ["SPY", "QQQ", "NVDA", "AAPL", "XLF"]
HORIZON = 5
ANNUALIZE = np.sqrt(252.0)

FEATURE_SETS = {
    "baseline": ["returns", "range", "volume_change"],
    "vol": ["log_ewma_vol", "log_vol_60", "vol_of_vol", "downside_share", "range_norm"],
    "vol_ret": ["log_ewma_vol", "log_vol_60", "vol_of_vol", "downside_share", "range_norm",
                "returns"],
}


def add_vol_features(feats: pd.DataFrame) -> pd.DataFrame:
    """Purpose-built, strictly trailing volatility features."""
    df = feats.copy()
    r = pd.Series(np.asarray(df["returns"].values, dtype=float), index=df.index)

    ewma = pd.Series(_ewma_vol(r.values), index=df.index)
    v20 = pd.Series(_trail_vol(r.values, 20), index=df.index)
    v60 = pd.Series(_trail_vol(r.values, 60), index=df.index)

    df["log_ewma_vol"] = np.log(ewma.clip(lower=EPS))
    df["log_vol_60"] = np.log(v60.clip(lower=EPS))
    # Vol-of-vol: how unstable the vol estimate itself has been. A real regime marker.
    df["vol_of_vol"] = v20.rolling(20).std(ddof=1) / v20.clip(lower=EPS)
    # Downside share: fraction of trailing variance contributed by negative returns.
    neg = (r.clip(upper=0.0) ** 2).rolling(20).sum()
    tot = (r ** 2).rolling(20).sum()
    df["downside_share"] = neg / tot.clip(lower=EPS)
    # Range normalized by its own trailing average, so it is a vol *state*, not a level.
    rng = pd.Series(np.asarray(df["range"].values, dtype=float), index=df.index)
    df["range_norm"] = rng / rng.rolling(60).mean().clip(lower=EPS)

    for c in ("log_ewma_vol", "log_vol_60", "vol_of_vol", "downside_share", "range_norm"):
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return df


def collect(tickers, period_days, interval, n_regimes, train_bars, oos_bars, step_bars,
            n_iter):
    rows, fails = [], {k: 0 for k in FEATURE_SETS}
    attempts = {k: 0 for k in FEATURE_SETS}
    for tkr in tickers:
        raw = fetch_data(tkr, period_days=period_days, interval=interval)
        if raw is None or raw.empty:
            print(f"  {tkr}: no data, skipped", file=sys.stderr)
            continue
        feats = add_vol_features(engineer_features(raw))
        rets = np.asarray(feats["returns"].values, dtype=float)

        fwd = _fwd_vol(rets, HORIZON)
        log_fwd = np.log(np.clip(fwd, EPS, None))
        log_fwd[~np.isfinite(fwd)] = np.nan
        log_ewma = np.log(np.clip(_ewma_vol(rets), EPS, None))
        log_ewma[~np.isfinite(log_ewma)] = np.nan

        # The purpose-built features need 60+ bars of history before they exist at all.
        first_valid = int(max(
            feats[c].first_valid_index() is not None
            and feats.index.get_loc(feats[c].first_valid_index()) or 0
            for c in FEATURE_SETS["vol"]))
        start = max(train_bars, first_valid + 120)

        n = len(feats)
        starts = list(range(start, n - oos_bars + 1, step_bars))
        if not starts:
            print(f"  {tkr}: only {n} bars, too short", file=sys.stderr)
            continue
        print(f"  {tkr}: {n} bars, {len(starts)} windows", file=sys.stderr)

        for w, oos_start in enumerate(starts):
            oos_end = oos_start + oos_bars
            tr, oos = slice(0, oos_start), slice(oos_start, oos_end)
            y_tr = log_fwd[tr]

            base_coef = _fit_linear(log_ewma[tr].reshape(-1, 1), y_tr)
            if base_coef is None:
                continue
            base_tr = _apply_linear(base_coef, log_ewma[tr].reshape(-1, 1))
            base_oos = _apply_linear(base_coef, log_ewma[oos].reshape(-1, 1))
            resid_tr = y_tr - base_tr

            per_set = {}
            for sname, cols in FEATURE_SETS.items():
                attempts[sname] += 1
                sub = feats.iloc[:oos_end].copy()
                if sub[cols].isna().any().any():
                    sub[cols] = sub[cols].ffill().bfill()
                det = RegimeDetector(n_regimes=n_regimes, n_iter=n_iter,
                                     feature_columns=cols)
                try:
                    trained = det.train(sub.iloc[:oos_start])
                    labeled = det.filtered_regimes(sub)
                except Exception as exc:
                    fails[sname] += 1
                    print(f"    {tkr} w{w} [{sname}]: fit failed ({type(exc).__name__})",
                          file=sys.stderr)
                    continue
                tids = np.asarray(trained["regime_id"].values, dtype=float)
                oids = np.asarray(labeled["regime_id"].values[oos], dtype=float)
                off = {}
                for rid in np.unique(tids[np.isfinite(tids)]):
                    m = (tids == rid) & np.isfinite(resid_tr)
                    if m.sum() >= 20:
                        off[int(rid)] = float(np.mean(resid_tr[m]))
                per_set[sname] = base_oos + np.array(
                    [off.get(int(r), 0.0) if np.isfinite(r) else 0.0 for r in oids])

            if not per_set:
                continue
            for j in range(oos_end - oos_start):
                i = oos_start + j
                if not np.isfinite(log_fwd[i]) or not np.isfinite(base_oos[j]):
                    continue
                row = {"ticker": tkr, "window": w, "bar": int(i),
                       "y": float(log_fwd[i]), "ewma94": float(base_oos[j])}
                ok = True
                for sname in FEATURE_SETS:
                    v = per_set.get(sname, np.array([np.nan]))[j] if sname in per_set else np.nan
                    row[sname] = float(v) if np.isfinite(v) else np.nan
                    ok &= np.isfinite(row[sname])
                if ok:
                    rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.attrs["fits"] = {k: f"{attempts[k]-fails[k]}/{attempts[k]}" for k in FEATURE_SETS}
    print("  fit success: " + ", ".join(
        f"{k} {attempts[k]-fails[k]}/{attempts[k]}" for k in FEATURE_SETS), file=sys.stderr)
    return df


def evaluate(df: pd.DataFrame) -> dict:
    nov = _decimate(df, HORIZON).reset_index(drop=True)
    table = {}
    for m in ["ewma94"] + list(FEATURE_SETS):
        L = _losses(nov, m)
        y = nov["y"].values
        table[m] = {"qlike": round(float(np.mean(L["qlike"])), 6),
                    "mse": round(float(np.mean(L["mse"])), 6),
                    "r2": round(float(1 - np.mean(L["mse"]) / np.var(y)), 4)}
    comps = {f"{s}_vs_ewma94": {l: _dm_test(nov, s, "ewma94", l)
                                for l in ("mse", "qlike")}
             for s in FEATURE_SETS}
    return {"n_obs_all": int(len(df)), "n_obs_nonoverlapping": int(len(nov)),
            "table": table, "comparisons": comps}


def report(res: dict) -> None:
    print("=" * 96)
    print("DO PURPOSE-BUILT VOLATILITY FEATURES MAKE THE HMM USEFUL?")
    print(f"Forward {HORIZON}-bar vol, scored on {res['n_obs_nonoverlapping']} "
          f"non-overlapping OOS bars")
    print("=" * 96)
    print(f"\n  {'forecast':<12}{'QLIKE':>10}{'MSE':>10}{'R2':>8}   note")
    notes = {"ewma94": "free EWMA, no HMM at all",
             "baseline": "EWMA + offset from SHIPPED features",
             "vol": "EWMA + offset from purpose-built vol features",
             "vol_ret": "EWMA + offset from vol features plus returns"}
    for m in ["ewma94"] + list(FEATURE_SETS):
        t = res["table"][m]
        print(f"  {m:<12}{t['qlike']:>10.5f}{t['mse']:>10.5f}{t['r2']:>8.3f}   {notes[m]}")
    print("\n  (lower is better; the HMM only earns its place by beating the first row)")

    print("\nPaired loss differentials vs the free EWMA, bootstrapped over windows.")
    print("A NEGATIVE difference means the feature set IMPROVES on EWMA.\n")
    for name, block in res["comparisons"].items():
        print(f"  {name}")
        for loss, t in block.items():
            if not t.get("usable"):
                print(f"    {loss:<7} unusable")
                continue
            verdict = {"a_worse": "WORSE than EWMA", "a_better": "BETTER than EWMA",
                       "tie": "no significant difference"}[t["verdict"]]
            print(f"    {loss:<7} diff={t['mean_diff']:+.5f}  "
                  f"95% CI [{t['ci_low']:+.5f}, {t['ci_high']:+.5f}]  -> {verdict}")
        print()

    print("=" * 96)
    wins = [s for s in FEATURE_SETS
            if res["comparisons"][f"{s}_vs_ewma94"]["qlike"].get("verdict") == "a_better"]
    if wins:
        print(f"Feature sets that beat a free EWMA: {', '.join(wins)}.")
        print("The earlier nulls were an input problem after all -- worth pursuing.")
    else:
        print("No feature set beats a free EWMA forecast, including ones built specifically")
        print("to describe volatility. The limitation is not the inputs: a 7-state discrete")
        print("latent variable is a poor container for a quantity that is continuous, highly")
        print("persistent, and already well estimated by an exponential average.")
    print("=" * 96)


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
        print(f"Pooled {len(df)} rows from {len(files)} file(s): "
              f"{sorted(df.ticker.unique())}", file=sys.stderr)
    else:
        df = collect(args.tickers, args.period_days, args.interval, args.regimes,
                     args.train_bars, args.oos_bars, args.step_bars, args.n_iter)
        if args.save_obs and not df.empty:
            Path(args.save_obs).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.save_obs, index=False)
            print(f"Wrote {len(df)} rows to {args.save_obs}", file=sys.stderr)

    if df.empty:
        print("No observations collected.", file=sys.stderr)
        sys.exit(1)

    res = evaluate(df)
    report(res)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(res, indent=2, default=str))
        print(f"\nSaved: {args.json_out}")


if __name__ == "__main__":
    main()
