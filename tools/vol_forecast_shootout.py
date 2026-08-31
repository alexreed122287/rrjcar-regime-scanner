#!/usr/bin/env python3
"""
Test 9 said the regimes carry forward-volatility information. Does an EWMA do it better?

Test 9 (`tools/regime_volatility.py`) is the only positive result in this repo's validation
history: the bearish regime set runs about 14.5% higher forward 5-day volatility than the
bullish set, beyond what a trailing-vol control implies. But "beyond a control" is a weak
standard. The control there was a *nuisance regressor* whose job was to absorb the obvious;
it was never asked to compete. A signal can be incrementally informative and still be
strictly worse than the free alternative sitting next to it.

So this is the head-to-head. Six forecasts of forward 5-bar realized volatility, all with
parameters estimated on TRAINING bars only, all scored on identical out-of-sample bars:

    trail20     log trailing 20-bar vol, calibrated
    trail5      log trailing 5-bar vol, calibrated
    ewma94      RiskMetrics EWMA, lambda = 0.94, calibrated
    hmm         per-regime mean forward log vol, estimated on train bars
    ewma_hmm    ewma94 plus a per-regime residual offset  <- does the regime ADD anything?
    combo_vol   trail20 + trail5 + ewma94 together, no regime information

Two questions, and they are different:

    1. Does `hmm` beat `ewma94` outright? If not, the regime label is a worse vol forecast
       than two lines of pandas, and test 9's positive result is redundant.
    2. Does `ewma_hmm` beat `ewma94`? This is the question test 9 was really groping toward.
       Even a poor standalone signal can add value on top of a good one -- and if it cannot,
       there is nothing left to build on.

Scoring: MSE on log vol (symmetric, what the models are fitted for) and QLIKE on variance
(the standard volatility-forecasting loss, which punishes under-prediction asymmetrically as
a risk manager would). Significance by Diebold-Mariano-style paired tests on loss
differentials, bootstrapped over whole (ticker, window) blocks, on DECIMATED non-overlapping
bars -- for the same reason as test 9, overlapping forward windows make naive p-values
meaningless.

Usage:
    python tools/vol_forecast_shootout.py --tickers SPY --save-obs vf/SPY.csv
    python tools/vol_forecast_shootout.py --from-obs 'vf/*.csv' --json docs/vol_forecast_shootout.json

Method notes
------------
* Same walk-forward and leakage discipline as tests 9 and 10: expanding train to oos_start,
  OOS regime labels from ``filtered_regimes`` so bar t's label uses only bars 0..t.
* Every calibration -- the log-log regressions, the per-regime means, the residual offsets --
  is fitted on the training slice of each window and then applied unchanged to OOS bars.
* A regime never seen in training falls back to the training global mean rather than being
  dropped, which is what a live system would have to do.
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
from scipy import stats

from data_loader import fetch_data, engineer_features
from hmm_engine import RegimeDetector

MODELS = ["trail20", "trail5", "ewma94", "hmm", "ewma_hmm", "combo_vol"]
DEFAULT_TICKERS = ["SPY", "QQQ", "NVDA", "AAPL", "XLF"]
HORIZON = 5
ANNUALIZE = np.sqrt(252.0)
EPS = 1e-12


def _fwd_vol(rets: np.ndarray, h: int) -> np.ndarray:
    n = len(rets)
    out = np.full(n, np.nan)
    for i in range(n - h):
        w = rets[i + 1: i + 1 + h]
        if np.isfinite(w).all():
            out[i] = np.std(w, ddof=1) * ANNUALIZE
    return out


def _trail_vol(rets: np.ndarray, k: int) -> np.ndarray:
    return (pd.Series(rets).rolling(k).std(ddof=1) * ANNUALIZE).values


def _ewma_vol(rets: np.ndarray, lam: float = 0.94) -> np.ndarray:
    s = pd.Series(rets).fillna(0.0)
    return (np.sqrt((s ** 2).ewm(alpha=1 - lam, adjust=False).mean()) * ANNUALIZE).values


def _fit_linear(X: np.ndarray, y: np.ndarray):
    """Least squares with an intercept. Returns coefficients or None if underdetermined."""
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if m.sum() < 60:
        return None
    A = np.column_stack([np.ones(m.sum()), X[m]])
    coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    return coef


def _apply_linear(coef, X: np.ndarray) -> np.ndarray:
    if coef is None:
        return np.full(len(X), np.nan)
    return coef[0] + X @ coef[1:]


def collect(tickers, period_days, interval, n_regimes, train_bars, oos_bars, step_bars,
            n_iter):
    """One row per OOS bar, holding the realized target and every model's forecast."""
    rows = []
    for tkr in tickers:
        raw = fetch_data(tkr, period_days=period_days, interval=interval)
        if raw is None or raw.empty:
            print(f"  {tkr}: no data, skipped", file=sys.stderr)
            continue
        feats = engineer_features(raw)
        rets = np.asarray(feats["returns"].values, dtype=float)
        fwd = _fwd_vol(rets, HORIZON)
        log_fwd = np.log(np.clip(fwd, EPS, None))
        log_fwd[~np.isfinite(fwd)] = np.nan

        preds = {
            "trail20": np.log(np.clip(_trail_vol(rets, 20), EPS, None)),
            "trail5": np.log(np.clip(_trail_vol(rets, 5), EPS, None)),
            "ewma94": np.log(np.clip(_ewma_vol(rets), EPS, None)),
        }
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
            tr = slice(0, oos_start)
            oos = slice(oos_start, oos_end)

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

            # --- calibrated single-predictor vol models -------------------------------
            fc = {}
            for name in ("trail20", "trail5", "ewma94"):
                coef = _fit_linear(preds[name][tr].reshape(-1, 1), y_tr)
                fc[name] = _apply_linear(coef, preds[name][oos].reshape(-1, 1))

            # --- combo of vol models, no regime information ---------------------------
            X_tr = np.column_stack([preds[k][tr] for k in ("trail20", "trail5", "ewma94")])
            X_oos = np.column_stack([preds[k][oos] for k in ("trail20", "trail5", "ewma94")])
            fc["combo_vol"] = _apply_linear(_fit_linear(X_tr, y_tr), X_oos)

            # --- the HMM's own best shot: per-regime mean forward log vol -------------
            gm = float(np.nanmean(y_tr)) if np.isfinite(y_tr).any() else np.nan
            reg_mean = {}
            for rid in np.unique(train_ids[np.isfinite(train_ids)]):
                m = (train_ids == rid) & np.isfinite(y_tr)
                if m.sum() >= 20:
                    reg_mean[int(rid)] = float(np.mean(y_tr[m]))
            fc["hmm"] = np.array([reg_mean.get(int(r), gm) if np.isfinite(r) else gm
                                  for r in oos_ids])

            # --- does the regime ADD to EWMA? per-regime offset on EWMA residuals -----
            base_coef = _fit_linear(preds["ewma94"][tr].reshape(-1, 1), y_tr)
            base_tr = _apply_linear(base_coef, preds["ewma94"][tr].reshape(-1, 1))
            resid_tr = y_tr - base_tr
            off = {}
            for rid in np.unique(train_ids[np.isfinite(train_ids)]):
                m = (train_ids == rid) & np.isfinite(resid_tr)
                if m.sum() >= 20:
                    off[int(rid)] = float(np.mean(resid_tr[m]))
            fc["ewma_hmm"] = fc["ewma94"] + np.array(
                [off.get(int(r), 0.0) if np.isfinite(r) else 0.0 for r in oos_ids])

            for j in range(oos_end - oos_start):
                i = oos_start + j
                if not np.isfinite(log_fwd[i]):
                    continue
                row = {"ticker": tkr, "window": w, "bar": int(i),
                       "regime_id": (int(oos_ids[j]) if np.isfinite(oos_ids[j]) else -1),
                       "y": float(log_fwd[i])}
                good = True
                for mname in MODELS:
                    v = fc[mname][j]
                    row[mname] = float(v) if np.isfinite(v) else np.nan
                    good &= np.isfinite(v)
                if good:
                    rows.append(row)
    return pd.DataFrame(rows)


def _losses(df: pd.DataFrame, model: str) -> dict:
    """MSE in log space, plus QLIKE on the variance level."""
    y, f = df["y"].values, df[model].values
    mse = (y - f) ** 2
    # QLIKE on variances: sigma2_true / sigma2_pred - log(...) - 1, >= 0, minimized at equality.
    v_true = np.exp(2 * y)
    v_pred = np.exp(2 * f)
    r = v_true / np.clip(v_pred, EPS, None)
    qlike = r - np.log(np.clip(r, EPS, None)) - 1.0
    return {"mse": mse, "qlike": qlike}


def _dm_test(df: pd.DataFrame, a: str, b: str, loss: str, n_boot: int = 4000,
             seed: int = 0) -> dict:
    """Paired loss-differential test, bootstrapped over (ticker, window) blocks.

    Positive mean_diff means model `a` has HIGHER loss, i.e. `a` is WORSE.
    """
    la = _losses(df, a)[loss]
    lb = _losses(df, b)[loss]
    d = la - lb
    blocks = [g.index.values for _, g in df.groupby(["ticker", "window"])]
    if len(blocks) < 5:
        return {"usable": False}
    dser = pd.Series(d, index=df.index)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        vals = np.concatenate([dser.loc[blocks[i]].values for i in pick])
        boot.append(vals.mean())
    boot = np.array(boot)
    return {"usable": True, "n": int(len(d)), "n_blocks": len(blocks),
            "mean_diff": float(np.mean(d)),
            "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)),
            "frac_a_worse": float((boot > 0).mean()),
            "verdict": ("a_worse" if np.percentile(boot, 2.5) > 0 else
                        "a_better" if np.percentile(boot, 97.5) < 0 else "tie")}


def _decimate(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    return df[df["bar"] % stride == 0].copy()


def evaluate(df: pd.DataFrame) -> dict:
    nov = _decimate(df, HORIZON).reset_index(drop=True)
    table = {}
    for m in MODELS:
        L = _losses(nov, m)
        y, f = nov["y"].values, nov[m].values
        table[m] = {
            "mse": round(float(np.mean(L["mse"])), 6),
            "rmse": round(float(np.sqrt(np.mean(L["mse"]))), 6),
            "qlike": round(float(np.mean(L["qlike"])), 6),
            "corr_with_truth": round(float(np.corrcoef(y, f)[0, 1]), 4),
            "r2": round(float(1 - np.mean(L["mse"]) / np.var(y)), 4),
        }
    ranked = sorted(MODELS, key=lambda m: table[m]["qlike"])
    comps = {
        "hmm_vs_ewma94": {l: _dm_test(nov, "hmm", "ewma94", l) for l in ("mse", "qlike")},
        "ewma_hmm_vs_ewma94": {l: _dm_test(nov, "ewma_hmm", "ewma94", l)
                               for l in ("mse", "qlike")},
        "hmm_vs_trail20": {l: _dm_test(nov, "hmm", "trail20", l) for l in ("mse", "qlike")},
        "ewma_hmm_vs_combo_vol": {l: _dm_test(nov, "ewma_hmm", "combo_vol", l)
                                  for l in ("mse", "qlike")},
    }
    return {"n_obs_all": int(len(df)), "n_obs_nonoverlapping": int(len(nov)),
            "horizon": HORIZON, "table": table, "ranking_by_qlike": ranked,
            "comparisons": comps}


def report(res: dict) -> None:
    print("=" * 96)
    print(f"FORWARD {res['horizon']}-BAR VOLATILITY FORECAST SHOOTOUT")
    print(f"Scored on {res['n_obs_nonoverlapping']} non-overlapping OOS bars "
          f"(of {res['n_obs_all']} total)")
    print("=" * 96)
    print(f"\n  {'model':<12}{'QLIKE':>10}{'MSE(log)':>11}{'RMSE':>9}{'corr':>8}{'R2':>8}   note")
    notes = {
        "trail20": "free, one line of pandas",
        "trail5": "free, noisier",
        "ewma94": "free, the standard benchmark",
        "hmm": "the regime label alone",
        "ewma_hmm": "EWMA + per-regime offset",
        "combo_vol": "three vol models, NO regime info",
    }
    for m in res["ranking_by_qlike"]:
        t = res["table"][m]
        print(f"  {m:<12}{t['qlike']:>10.5f}{t['mse']:>11.5f}{t['rmse']:>9.5f}"
              f"{t['corr_with_truth']:>8.3f}{t['r2']:>8.3f}   {notes[m]}")
    print("\n  (lower QLIKE and MSE are better; ranked by QLIKE)")

    print("\nPaired loss differentials, bootstrapped over (ticker, window) blocks.")
    print("A POSITIVE difference means the FIRST model is WORSE.\n")
    for name, block in res["comparisons"].items():
        print(f"  {name}")
        for loss, t in block.items():
            if not t.get("usable"):
                print(f"    {loss:<7} unusable")
                continue
            print(f"    {loss:<7} diff={t['mean_diff']:+.5f}  "
                  f"95% CI [{t['ci_low']:+.5f}, {t['ci_high']:+.5f}]  -> {t['verdict']}")
        print()

    h_e = res["comparisons"]["hmm_vs_ewma94"]["qlike"]
    add = res["comparisons"]["ewma_hmm_vs_ewma94"]["qlike"]
    print("=" * 96)
    print("Q1: does the regime label beat a free EWMA outright?")
    if h_e.get("verdict") == "a_worse":
        print("   NO -- the HMM regime label is a SIGNIFICANTLY WORSE vol forecast than EWMA.")
    elif h_e.get("verdict") == "a_better":
        print("   YES -- the regime label beats EWMA outright. Unexpected; check it twice.")
    else:
        print("   TIE -- indistinguishable from EWMA, which needs no HMM.")
    print("\nQ2: does the regime label ADD anything on top of EWMA?")
    if add.get("verdict") == "a_better":
        print("   YES -- EWMA + regime offset beats EWMA alone. Small, but real, and it is")
        print("   the only thing in this repo with a measurable forward-looking use.")
    elif add.get("verdict") == "a_worse":
        print("   NO -- adding the regime label makes the forecast WORSE. Test 9's increment")
        print("   does not survive being asked to compete; nothing here is worth building on.")
    else:
        print("   NO -- no significant improvement over EWMA alone. Test 9's positive result")
        print("   is real but redundant: the free forecast already contains it.")
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
