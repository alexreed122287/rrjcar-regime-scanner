#!/usr/bin/env python3
"""Test 15 -- is the volatility model class the binding constraint?

Tests 11-14 kept arriving at the same place: a one-parameter EWMA was never beaten. Test 11
showed the HMM regime label ranks 4th of 6 as a vol forecast and adds nothing on top of EWMA.
Test 13 showed purpose-built vol features are worse than the baseline. Test 14 showed the
sizing overlay built on those forecasts fails its own constant-exposure control.

Every one of those tests held the *model class* fixed. This one asks the question they could
not: **is EWMA's dominance a statement about the HMM, or about the forecasting problem?**

The reference is deliberately the cheapest thing that works:

  ewma94    RiskMetrics EWMA, lambda = 0.94, ONE parameter and it is not even fitted.

Against three model classes that are strictly more expressive:

  ewma_fit  the same EWMA with lambda chosen on TRAIN data only, by grid search. Isolates
            one question: is 0.94 a lucky constant? To be precise about scope -- 0.94 appears
            only in the research tools (tools/regime_volatility.py,
            tools/vol_forecast_shootout.py), never in the serving path, which uses no EWMA at
            all. So this costs production nothing. What it does affect is the *benchmark* that
            tests 9-14 measured the HMM against.
  har       Corsi's HAR: log realized vol at daily, weekly (5) and monthly (22) horizons.
            The standard workhorse of the realized-volatility literature. Three predictors.
            On daily bars the daily component is |r_t|, the usual low-frequency proxy.
  garch11   GARCH(1,1), omega/alpha/beta by Gaussian maximum likelihood on TRAIN returns
            only, then filtered forward through the OOS block and aggregated analytically
            over the horizon. The canonical conditional-variance model.

Scoring is deliberately identical to test 11 -- the loss functions, the block bootstrap and
the non-overlapping decimation are IMPORTED from tools/vol_forecast_shootout rather than
reimplemented, so the numbers are directly comparable to that table instead of merely
similar to it.

Ground rules carried over from test 11:

* Target is log forward 5-bar realized volatility, annualized.
* Every model, including the reference, gets a train-only affine recalibration
  (intercept + slope on its log forecast). Nobody is handicapped by scale or bias.
* Every parameter -- lambda, the HAR coefficients, the GARCH parameters -- is estimated on
  the training slice only and never refitted inside the OOS block.
* Scored on non-overlapping bars only (stride = horizon), because a 5-bar forward target on
  consecutive bars overlaps 4/5 and would inflate significance.
* A positive loss differential means the FIRST model is WORSE.

What a result would mean, stated before running it:

* If HAR or GARCH beats EWMA significantly, the ceiling in tests 11-14 was the model class,
  and the vol work is worth restarting on a better base.
* If they tie, the ceiling is the predictability of 5-bar forward vol from daily bars, and no
  amount of model sophistication moves it. That would make EWMA the right *engineering*
  choice, not merely a convenient baseline.
* If the fitted lambda lands near 0.94, the constant is defensible. If it lands far away and
  still does not beat it, the loss surface is flat and the constant is harmless. If it lands
  far away AND scores better, then tests 11-14 benchmarked the HMM against a needlessly weak
  EWMA -- which would make their negative verdicts stronger, not weaker.

Usage:
    python tools/vol_model_class.py --tickers SPY QQQ NVDA AAPL XLF \\
        --save-obs /path/obs --json docs/vol_model_class.json
    python tools/vol_model_class.py --from-obs '/path/obs/*.csv' --json docs/vol_model_class.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import optimize, stats

from data_loader import fetch_data, engineer_features

# Imported, not reimplemented: identical scoring to test 11 is the whole point.
from tools.vol_forecast_shootout import (
    ANNUALIZE,
    EPS,
    HORIZON,
    _apply_linear,
    _decimate,
    _dm_test,
    _ewma_vol,
    _fit_linear,
    _fwd_vol,
    _losses,
    _trail_vol,
)

MODELS = ["ewma94", "ewma_fit", "har", "garch11"]
DEFAULT_TICKERS = ["SPY", "QQQ", "NVDA", "AAPL", "XLF"]

# Grid for the fitted-lambda variant. Wide enough to embarrass 0.94 if it deserves it.
# Deliberately extends well below the RiskMetrics 0.94: a first pass bounded at 0.80 put SPY's
# median fit at 0.8325 with a window pinned to the boundary, so a narrower grid would have
# reported the bound rather than the optimum. `frac_at_grid_edge` in the diagnostics keeps
# this honest -- if it is materially above zero, the grid is still censoring the answer.
LAMBDA_GRID = np.round(np.arange(0.70, 0.9901, 0.005), 4)

# Fixed lambdas scored as their own models, to map the OOS loss surface directly instead of
# inferring its shape from the fitted-lambda result. Includes the repo's hardcoded 0.94.
LAMBDA_CURVE = [0.70, 0.75, 0.80, 0.84, 0.88, 0.90, 0.94, 0.97, 0.99]


def _lam_col(lam: float) -> str:
    return f"lam{int(round(lam * 1000)):04d}"


NOTES = {
    "ewma94": "1 param, NOT fitted -- the reference",
    "ewma_fit": "1 param, fitted on train",
    "har": "3 params, daily/weekly/monthly",
    "garch11": "3 params, Gaussian MLE",
}


# --------------------------------------------------------------------------- GARCH(1,1)

def _garch_nll(params, r2, backcast):
    """Gaussian negative log-likelihood for GARCH(1,1). Returns on a percent scale."""
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
        return 1e12
    n = len(r2)
    s2 = np.empty(n)
    s2[0] = backcast
    for t in range(1, n):
        s2[t] = omega + alpha * r2[t - 1] + beta * s2[t - 1]
        if s2[t] <= 0 or not np.isfinite(s2[t]):
            return 1e12
    return 0.5 * float(np.sum(np.log(s2) + r2 / s2))


def _fit_garch(rets_train: np.ndarray):
    """MLE on the training returns. Returns (omega, alpha, beta, converged)."""
    r = np.asarray(rets_train, dtype=float)
    r = r[np.isfinite(r)] * 100.0  # percent scale keeps the optimizer well conditioned
    if len(r) < 250:
        return None
    r = r - r.mean()
    r2 = r ** 2
    backcast = float(np.mean(r2))

    best, best_nll = None, np.inf
    # Several starts: GARCH likelihoods are flat in (alpha+beta) and can stall.
    for a0, b0 in ((0.05, 0.90), (0.10, 0.85), (0.02, 0.96), (0.15, 0.75)):
        x0 = np.array([backcast * (1 - a0 - b0), a0, b0])
        try:
            res = optimize.minimize(
                _garch_nll, x0, args=(r2, backcast), method="L-BFGS-B",
                bounds=[(1e-10, None), (0.0, 0.999), (0.0, 0.999)],
            )
        except Exception:
            continue
        if res.fun < best_nll and np.isfinite(res.fun) and res.fun < 1e11:
            o, a, b = res.x
            if a + b < 0.9999 and o > 0:
                best, best_nll = (float(o), float(a), float(b), bool(res.success)), res.fun
    return best


def _garch_forecast_path(rets_full: np.ndarray, omega, alpha, beta, h: int) -> np.ndarray:
    """Filter variance through the whole series with FIXED params, then forecast h bars.

    At bar t the returned value uses returns up to and including t, and predicts the average
    per-bar variance over t+1 .. t+h -- matching the realized target exactly. Annualized vol.
    """
    r = np.where(np.isfinite(rets_full), rets_full, 0.0) * 100.0
    r2 = r ** 2
    n = len(r)
    lr = omega / (1.0 - alpha - beta)  # long-run variance
    s2 = np.empty(n)
    s2[0] = lr
    for t in range(1, n):
        s2[t] = omega + alpha * r2[t - 1] + beta * s2[t - 1]

    persist = alpha + beta
    out = np.full(n, np.nan)
    # Average of the k-step forecasts, k = 1..h, in closed form.
    weights = np.array([persist ** (k - 1) for k in range(1, h + 1)])
    wsum = float(weights.sum())
    for t in range(n - 1):
        s2_next = omega + alpha * r2[t] + beta * s2[t]
        avg = lr + (s2_next - lr) * wsum / h
        if avg > 0 and np.isfinite(avg):
            out[t] = np.sqrt(avg) / 100.0 * ANNUALIZE
    return out


# --------------------------------------------------------------------------- collection

def _log_clip(x: np.ndarray) -> np.ndarray:
    out = np.log(np.clip(x, EPS, None))
    out[~np.isfinite(x)] = np.nan
    return out


def collect(tickers, period_days, interval, train_bars, oos_bars, step_bars):
    """One row per OOS bar: the realized target plus every model's forecast."""
    rows = []
    for tkr in tickers:
        raw = fetch_data(tkr, period_days=period_days, interval=interval)
        if raw is None or raw.empty:
            print(f"  {tkr}: no data, skipped", file=sys.stderr)
            continue
        feats = engineer_features(raw)
        rets = np.asarray(feats["returns"].values, dtype=float)
        fwd = _fwd_vol(rets, HORIZON)
        log_fwd = _log_clip(fwd)
        log_fwd[~np.isfinite(fwd)] = np.nan

        # HAR components on daily bars: |r| stands in for daily realized vol.
        abs_ret = np.abs(rets) * ANNUALIZE
        har_cols = {
            "rv_d": _log_clip(abs_ret),
            "rv_w": _log_clip(_trail_vol(rets, 5)),
            "rv_m": _log_clip(_trail_vol(rets, 22)),
        }
        ewma_cache = {float(lam): _log_clip(_ewma_vol(rets, float(lam)))
                      for lam in LAMBDA_GRID}
        log_ewma94 = _log_clip(_ewma_vol(rets, 0.94))

        n = len(feats)
        starts = list(range(train_bars, n - oos_bars + 1, step_bars))
        if not starts:
            print(f"  {tkr}: only {n} bars, too short", file=sys.stderr)
            continue
        print(f"  {tkr}: {n} bars, {len(starts)} windows", file=sys.stderr)

        for w, oos_start in enumerate(starts):
            oos_end = oos_start + oos_bars
            tr, oos = slice(0, oos_start), slice(oos_start, oos_end)
            y_tr = log_fwd[tr]
            fc = {}

            # --- reference: lambda = 0.94, affine-recalibrated on train ----------------
            coef = _fit_linear(log_ewma94[tr].reshape(-1, 1), y_tr)
            fc["ewma94"] = _apply_linear(coef, log_ewma94[oos].reshape(-1, 1))

            # --- lambda fitted on TRAIN only, same affine treatment -------------------
            best_lam, best_mse = np.nan, np.inf
            for lam, series in ewma_cache.items():
                c = _fit_linear(series[tr].reshape(-1, 1), y_tr)
                if c is None:
                    continue
                pred_tr = _apply_linear(c, series[tr].reshape(-1, 1))
                m = np.isfinite(pred_tr) & np.isfinite(y_tr)
                if m.sum() < 60:
                    continue
                mse = float(np.mean((y_tr[m] - pred_tr[m]) ** 2))
                if mse < best_mse:
                    best_lam, best_mse = lam, mse
            if np.isfinite(best_lam):
                series = ewma_cache[best_lam]
                c = _fit_linear(series[tr].reshape(-1, 1), y_tr)
                fc["ewma_fit"] = _apply_linear(c, series[oos].reshape(-1, 1))
            else:
                fc["ewma_fit"] = np.full(oos_bars, np.nan)

            # --- HAR: three horizons, OLS on train ------------------------------------
            X_tr = np.column_stack([har_cols[k][tr] for k in ("rv_d", "rv_w", "rv_m")])
            X_oos = np.column_stack([har_cols[k][oos] for k in ("rv_d", "rv_w", "rv_m")])
            fc["har"] = _apply_linear(_fit_linear(X_tr, y_tr), X_oos)

            # --- GARCH(1,1): MLE on train, filtered forward ---------------------------
            g = _fit_garch(rets[tr])
            if g is None:
                fc["garch11"] = np.full(oos_bars, np.nan)
                g_omega = g_alpha = g_beta = np.nan
                g_ok = False
            else:
                g_omega, g_alpha, g_beta, g_ok = g
                gvol = _garch_forecast_path(rets, g_omega, g_alpha, g_beta, HORIZON)
                lg = _log_clip(gvol)
                c = _fit_linear(lg[tr].reshape(-1, 1), y_tr)
                fc["garch11"] = _apply_linear(c, lg[oos].reshape(-1, 1))

            # --- fixed-lambda curve: same affine treatment, one column each -----------
            curve = {}
            for lam in LAMBDA_CURVE:
                series = ewma_cache.get(float(lam))
                if series is None:
                    series = _log_clip(_ewma_vol(rets, float(lam)))
                c = _fit_linear(series[tr].reshape(-1, 1), y_tr)
                curve[_lam_col(lam)] = _apply_linear(c, series[oos].reshape(-1, 1))

            for j in range(oos_end - oos_start):
                i = oos_start + j
                if not np.isfinite(log_fwd[i]):
                    continue
                row = {"ticker": tkr, "window": w, "bar": int(i),
                       "y": float(log_fwd[i]),
                       "lam_fit": float(best_lam), "g_omega": float(g_omega),
                       "g_alpha": float(g_alpha), "g_beta": float(g_beta),
                       "g_converged": bool(g_ok)}
                good = True
                for m in MODELS:
                    v = fc[m][j]
                    row[m] = float(v) if np.isfinite(v) else np.nan
                    good &= np.isfinite(v)
                for col, arr in curve.items():
                    v = arr[j]
                    row[col] = float(v) if np.isfinite(v) else np.nan
                    good &= np.isfinite(v)
                if good:
                    rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- evaluation

def _block_win_rate(df: pd.DataFrame, a: str, b: str, loss: str = "qlike") -> dict:
    """Descriptive: in how many (ticker, window) blocks does `a` have the lower mean loss?

    Secondary to the bootstrap CI, never a substitute for it. A model can win most blocks by a
    hair and still be indistinguishable overall -- and a consistent direction across blocks is
    worth reporting even when the magnitude is inside the noise band, because the two answer
    different questions. Significance is the CI excluding zero, and nothing else.
    """
    la = pd.Series(_losses(df, a)[loss], index=df.index)
    lb = pd.Series(_losses(df, b)[loss], index=df.index)
    diffs = []
    for _, g in df.groupby(["ticker", "window"]):
        diffs.append(float(la.loc[g.index].mean() - lb.loc[g.index].mean()))
    diffs = np.asarray(diffs)
    out = {"loss": loss, "n_blocks": int(len(diffs)),
           "a_better_blocks": int((diffs < 0).sum()),
           "a_better_frac": round(float((diffs < 0).mean()), 4),
           "median_block_diff": round(float(np.median(diffs)), 6)}
    if len(diffs) >= 6:
        try:
            out["wilcoxon_p"] = round(float(stats.wilcoxon(diffs).pvalue), 5)
        except Exception:
            out["wilcoxon_p"] = None
    return out


def evaluate(df: pd.DataFrame, n_boot: int = 4000) -> dict:
    """Score every model. ``n_boot`` is exposed so tests can run a cheap bootstrap; the
    default is the value used for every published number."""
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

    comps = {}
    for a, b in (("har", "ewma94"), ("garch11", "ewma94"), ("ewma_fit", "ewma94"),
                 ("garch11", "har"), ("har", "ewma_fit")):
        comps[f"{a}_vs_{b}"] = {l: _dm_test(nov, a, b, l, n_boot=n_boot)
                                for l in ("mse", "qlike")}
        comps[f"{a}_vs_{b}"]["block_win_rate"] = _block_win_rate(nov, a, b)

    per_ticker = {}
    for tkr, g in nov.groupby("ticker"):
        per_ticker[tkr] = {m: round(float(np.mean(_losses(g, m)["qlike"])), 6) for m in MODELS}
        per_ticker[tkr]["best"] = min(MODELS, key=lambda m: per_ticker[tkr][m])

    lam_curve = {}
    for lam in LAMBDA_CURVE:
        col = _lam_col(lam)
        if col in nov.columns:
            lam_curve[f"{lam:.2f}"] = round(float(np.mean(_losses(nov, col)["qlike"])), 6)

    lam = df.groupby(["ticker", "window"])["lam_fit"].first().dropna()
    gp = df.groupby(["ticker", "window"])[["g_omega", "g_alpha", "g_beta"]].first().dropna()
    conv = df.groupby(["ticker", "window"])["g_converged"].first()
    diag = {
        "lambda_fitted": {
            "n_windows": int(len(lam)),
            "min": round(float(lam.min()), 4), "median": round(float(lam.median()), 4),
            "max": round(float(lam.max()), 4), "mean": round(float(lam.mean()), 4),
            "frac_within_0.02_of_0.94": round(float((lam - 0.94).abs().le(0.02).mean()), 4),
            "frac_at_grid_edge": round(
                float(((lam <= LAMBDA_GRID[0] + 1e-9) | (lam >= LAMBDA_GRID[-1] - 1e-9)).mean()), 4),
        },
        "garch_params": {
            "n_windows": int(len(gp)),
            "alpha_median": round(float(gp["g_alpha"].median()), 4),
            "beta_median": round(float(gp["g_beta"].median()), 4),
            "persistence_median": round(float((gp["g_alpha"] + gp["g_beta"]).median()), 4),
            "persistence_max": round(float((gp["g_alpha"] + gp["g_beta"]).max()), 4),
            "frac_converged": round(float(conv.mean()), 4),
        },
    }
    return {"n_obs_all": int(len(df)), "n_obs_nonoverlapping": int(len(nov)),
            "horizon": HORIZON, "table": table, "ranking_by_qlike": ranked,
            "comparisons": comps, "per_ticker_qlike": per_ticker,
            "lambda_loss_curve": lam_curve, "diagnostics": diag}


def report(res: dict) -> None:
    print("=" * 96)
    print(f"TEST 15 -- IS THE VOL MODEL CLASS THE BINDING CONSTRAINT?")
    print(f"Forward {res['horizon']}-bar vol, scored on {res['n_obs_nonoverlapping']} "
          f"non-overlapping OOS bars (of {res['n_obs_all']})")
    print("=" * 96)
    print(f"\n  {'model':<10}{'QLIKE':>10}{'MSE(log)':>11}{'RMSE':>9}{'corr':>8}{'R2':>8}   note")
    for m in res["ranking_by_qlike"]:
        t = res["table"][m]
        print(f"  {m:<10}{t['qlike']:>10.5f}{t['mse']:>11.5f}{t['rmse']:>9.5f}"
              f"{t['corr_with_truth']:>8.3f}{t['r2']:>8.3f}   {NOTES[m]}")
    print("\n  (lower QLIKE and MSE are better; ranked by QLIKE)")

    print("\nPaired loss differentials, bootstrapped over (ticker, window) blocks.")
    print("A POSITIVE difference means the FIRST model is WORSE.\n")
    for name, block in res["comparisons"].items():
        print(f"  {name}")
        for loss, t in block.items():
            if loss == "block_win_rate":
                continue
            if not t.get("usable"):
                print(f"    {loss:<7} unusable")
                continue
            print(f"    {loss:<7} diff={t['mean_diff']:+.5f}  "
                  f"95% CI [{t['ci_low']:+.5f}, {t['ci_high']:+.5f}]  -> {t['verdict']}")
        bw = block.get("block_win_rate")
        if bw:
            print(f"    blocks  first model better in {bw['a_better_blocks']}/{bw['n_blocks']}"
                  f" ({bw['a_better_frac']:.0%}) by QLIKE, median {bw['median_block_diff']:+.5f}"
                  f", wilcoxon p={bw.get('wilcoxon_p')}   [descriptive only]")
        print()

    print("Per-ticker QLIKE (which model wins where):")
    hdr = "".join(f"{m:>11}" for m in MODELS)
    print(f"  {'ticker':<8}{hdr}{'winner':>12}")
    for tkr, row in res["per_ticker_qlike"].items():
        vals = "".join(f"{row[m]:>11.5f}" for m in MODELS)
        print(f"  {tkr:<8}{vals}{row['best']:>12}")

    curve = res.get("lambda_loss_curve") or {}
    if curve:
        print("\nOOS QLIKE as a function of a FIXED lambda (the loss surface itself):")
        best_lam = min(curve, key=curve.get)
        for k, v in curve.items():
            mark = "  <- best on this grid" if k == best_lam else ""
            tag = "  (hardcoded in the repo)" if k == "0.94" else ""
            print(f"    lambda={k}  QLIKE {v:.5f}{tag}{mark}")
        spread = max(curve.values()) - min(curve.values())
        print(f"    spread across the whole grid: {spread:.5f}")

    d = res["diagnostics"]
    lam, gp = d["lambda_fitted"], d["garch_params"]
    print(f"\nFitted lambda over {lam['n_windows']} windows: median {lam['median']}, "
          f"range [{lam['min']}, {lam['max']}]")
    print(f"  within 0.02 of the hardcoded 0.94: {lam['frac_within_0.02_of_0.94']:.0%}"
          f"   pinned at a grid edge: {lam['frac_at_grid_edge']:.0%}")
    print(f"GARCH(1,1) over {gp['n_windows']} windows: alpha {gp['alpha_median']}, "
          f"beta {gp['beta_median']}, persistence {gp['persistence_median']} "
          f"(max {gp['persistence_max']}), converged {gp['frac_converged']:.0%}")

    print("\n" + "=" * 96)
    print("VERDICT")
    ref = res["table"]["ewma94"]["qlike"]
    beat = [m for m in MODELS if m != "ewma94"
            and res["comparisons"].get(f"{m}_vs_ewma94", {}).get("qlike", {}).get("verdict")
            == "a_better"]
    worse = [m for m in MODELS if m != "ewma94"
             and res["comparisons"].get(f"{m}_vs_ewma94", {}).get("qlike", {}).get("verdict")
             == "a_worse"]
    if beat:
        print(f"   The model class WAS a real constraint: {', '.join(beat)} significantly beat")
        print(f"   the one-parameter EWMA reference (QLIKE {ref:.5f}). The vol work in tests")
        print("   11-14 was built on the wrong base and is worth revisiting on this one.")
    else:
        print("   No model class significantly beat a one-parameter, unfitted EWMA.")
        if worse:
            print(f"   {', '.join(worse)} were significantly WORSE -- extra parameters cost")
            print("   accuracy here rather than buying it.")
        print("   The ceiling in tests 11-14 was the predictability of 5-bar forward vol from")
        print("   daily bars, NOT the HMM and NOT the model class. EWMA is the right")
        print("   engineering choice, not merely a convenient straw man.")
    print("=" * 96)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--period-days", type=int, default=3000)
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--train-bars", type=int, default=756)
    ap.add_argument("--oos-bars", type=int, default=126)
    ap.add_argument("--step-bars", type=int, default=126)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--save-obs", default=None)
    ap.add_argument("--from-obs", nargs="+", default=None)
    args = ap.parse_args()

    if args.from_obs:
        paths = []
        for pat in args.from_obs:
            paths.extend(sorted(glob.glob(pat)))
        if not paths:
            print("no observation files matched", file=sys.stderr)
            sys.exit(1)
        print(f"pooling {len(paths)} observation file(s)", file=sys.stderr)
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    else:
        print(f"collecting {len(args.tickers)} ticker(s)", file=sys.stderr)
        df = collect(args.tickers, args.period_days, args.interval,
                     args.train_bars, args.oos_bars, args.step_bars)

    if df.empty:
        print("no usable observations", file=sys.stderr)
        sys.exit(1)

    if args.save_obs:
        os.makedirs(args.save_obs, exist_ok=True)
        for tkr, g in df.groupby("ticker"):
            out = os.path.join(args.save_obs, f"{tkr}.csv")
            g.to_csv(out, index=False)
            print(f"  saved {len(g)} rows -> {out}", file=sys.stderr)

    res = evaluate(df)
    report(res)
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"\nwrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
