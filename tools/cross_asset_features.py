"""
Do cross-asset features give the HMM regimes any forward information?

Tests 4 and 5 (docs/validation-findings.md) showed the three core features --
``returns``, ``range``, ``volume_change`` -- produce regimes with no forward-return
information at any horizon from 1 to 20 bars. All three are short-horizon restatements of
the target's own volatility and momentum, so the natural question is whether regimes built
on *macro* state do better.

Three feature sets are compared on identical windows and identical tests:

  core    returns, range, volume_change                     (the current production set)
  cross   credit, rates, breadth                            (macro state only)
  both    all six

Cross-asset series, all ETF proxies because index tickers (^VIX, ^TNX) are not available
from any configured data source:

  credit   log(HYG/LQD) change   -- high-yield vs investment-grade, i.e. credit appetite
  rates    TLT return            -- long-duration Treasuries, i.e. rate moves
  breadth  log(RSP/SPY) change   -- equal- vs cap-weighted S&P, i.e. participation

Test, unchanged from tools/regime_separability.py so results are comparable: fit on the
in-sample window, label the out-of-sample window causally, then ask whether forward
returns differ across regimes (Kruskal-Wallis) and whether the in-sample bullish ordering
survives (Spearman rho, which must be NEGATIVE to be tradeable).

Caveat stated up front: ``both`` fits 6 features with full covariance on 252 samples,
which is a lot of free parameters. A null there is partly a sample-size statement, so
``cross`` (3 features, same count as ``core``) is the cleaner apples-to-apples comparison.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from data_loader import fetch_data, engineer_features   # noqa: E402
from hmm_engine import RegimeDetector                   # noqa: E402

CORE = ["returns", "range", "volume_change"]
CROSS = ["credit", "rates", "breadth"]
SETS = {"core": CORE, "cross": CROSS, "both": CORE + CROSS}


def load_cross_asset(period_days: int, interval: str) -> pd.DataFrame:
    """Credit, rates and breadth as stationary daily changes."""
    px = {}
    for sym in ("HYG", "LQD", "TLT", "RSP", "SPY"):
        px[sym] = fetch_data(sym, period_days=period_days, interval=interval)["Close"]
    p = pd.DataFrame(px).dropna()
    out = pd.DataFrame(index=p.index)
    out["credit"] = np.log(p["HYG"] / p["LQD"]).diff()
    out["rates"] = p["TLT"].pct_change()
    out["breadth"] = np.log(p["RSP"] / p["SPY"]).diff()
    return out.dropna()


def analyze(ticker: str, cross: pd.DataFrame, period_days: int, interval: str,
            n_regimes, is_bars: int, oos_bars: int, hmm_iter: int = 100) -> List[dict]:
    raw = fetch_data(ticker, period_days=period_days, interval=interval)
    df = engineer_features(raw)
    # Inner join keeps only dates where both the target and every macro series exist.
    df = df.join(cross, how="inner").dropna(subset=CORE + CROSS)

    rows: List[dict] = []
    start, idx = 0, 0
    while start + is_bars + oos_bars <= len(df):
        is_end = start + is_bars
        oos_start, oos_end = is_end, is_end + oos_bars
        idx += 1
        for name, cols in SETS.items():
            try:
                det = RegimeDetector(n_regimes=n_regimes, n_iter=hmm_iter,
                                     random_state=42, feature_columns=cols)
                det.train(df.iloc[start:is_end])
                labeled = det.filtered_regimes(df.iloc[:oos_end])
                oos = labeled.iloc[oos_start:oos_end].copy()
                oos["fwd"] = oos["Close"].pct_change().shift(-1) * 100.0
                oos = oos.dropna(subset=["fwd", "regime_id"])
                if len(oos) < 20:
                    continue

                groups, means = [], {}
                for rid, g in oos.groupby("regime_id"):
                    if len(g) >= 3:
                        groups.append(g["fwd"].values)
                        means[int(rid)] = float(g["fwd"].mean())
                if len(groups) < 2:
                    continue

                rho, _ = stats.spearmanr(oos["regime_id"].values, oos["fwd"].values)
                kw_p = stats.kruskal(*groups).pvalue
                r0 = means.get(0)
                rows.append({
                    "ticker": ticker, "window": idx, "featureset": name,
                    "n_features": len(cols), "n_regimes": int(det.n_regimes),
                    "rho": float(rho), "kruskal_p": float(kw_p),
                    "regime0_mean": r0, "overall_mean": float(oos["fwd"].mean()),
                })
            except Exception as exc:  # noqa: BLE001
                print(f"  [{ticker}] w{idx} {name}: {exc}", file=sys.stderr)
        start += oos_bars
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", default="SPY,QQQ,NVDA,AAPL,XLF")
    ap.add_argument("--period-days", type=int, default=3000)
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--regimes", default="auto")
    ap.add_argument("--is-bars", type=int, default=252)
    ap.add_argument("--oos-bars", type=int, default=126)
    ap.add_argument("--out", default="/home/user/workspace/cross_asset.json")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    n_regimes = args.regimes if args.regimes == "auto" else int(args.regimes)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cross = load_cross_asset(args.period_days, args.interval)
    print(f"  cross-asset panel: {len(cross)} bars "
          f"{cross.index[0].date()} -> {cross.index[-1].date()}", file=sys.stderr)

    rows: List[dict] = []
    for t in tickers:
        b = io.StringIO()
        with contextlib.redirect_stdout(b):
            r = analyze(t, cross, args.period_days, args.interval, n_regimes,
                        args.is_bars, args.oos_bars)
        rows += r
        print(f"  {t}: {len(r)} (window x featureset) results", file=sys.stderr)

    if not rows:
        print("nothing analyzed", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print(f"CROSS-ASSET FEATURES vs CORE FEATURES    regimes={args.regimes}")
    print("Identical windows and tests. rho must be NEGATIVE to be tradeable.")
    print("=" * 80)

    hdr = (f"{'set':<7}{'nfeat':>6}{'wins':>6}{'meanRho':>10}{'rho p':>8}"
           f"{'KW<.05':>8}{'KW p':>8}{'r0>avg':>8}")
    print("\n" + hdr); print("-" * len(hdr))
    summary = {}
    for name in ("core", "cross", "both"):
        g = df[df["featureset"] == name]
        if g.empty:
            continue
        # Ticker-level aggregation: tickers are the independent unit, not windows.
        tk = g.groupby("ticker")["rho"].mean()
        rho_p = stats.ttest_1samp(tk.values, 0.0).pvalue if len(tk) > 2 else float("nan")
        kw_rate = 100.0 * (g["kruskal_p"] < 0.05).mean()
        kw_binom = stats.binomtest(int((g["kruskal_p"] < 0.05).sum()),
                                   len(g), 0.05).pvalue
        beat = g.dropna(subset=["regime0_mean"])
        beat_pct = 100.0 * (beat["regime0_mean"] > beat["overall_mean"]).mean()
        print(f"{name:<7}{g['n_features'].iloc[0]:>6}{len(g):>6}"
              f"{g['rho'].mean():>10.3f}{rho_p:>8.3f}"
              f"{kw_rate:>7.0f}%{kw_binom:>8.3f}{beat_pct:>7.0f}%")
        summary[name] = {"n_windows": int(len(g)),
                         "mean_rho": float(g["rho"].mean()),
                         "rho_p_tickerlevel": float(rho_p),
                         "kw_sig_rate_pct": float(kw_rate),
                         "kw_vs_null_p": float(kw_binom),
                         "regime0_beat_pct": float(beat_pct),
                         "mean_n_regimes": float(g["n_regimes"].mean())}

    print("\nKruskal-Wallis significance rate by ticker "
          "(5% is the null; higher = regimes separate forward returns):")
    piv = df.assign(sig=df["kruskal_p"] < 0.05).pivot_table(
        index="ticker", columns="featureset", values="sig", aggfunc="mean") * 100
    print(piv.round(0).to_string())

    print("\nInterpretation")
    print("-" * 80)
    # Three feature sets are compared, so the per-set threshold is Bonferroni-adjusted.
    # And separation alone is not enough: a set that separates forward returns while the
    # bullish ordering is INVERTED is not tradeable as labelled, so direction is required.
    alpha = 0.05 / max(1, len(summary))
    print(f"  Bonferroni threshold for {len(summary)} feature sets: alpha = {alpha:.3f}")
    for name, sm in summary.items():
        separates = sm["kw_vs_null_p"] < alpha and sm["kw_sig_rate_pct"] > 5
        right_sign = sm["mean_rho"] < 0
        if separates and right_sign:
            verdict = "separates forward returns in the tradeable direction"
        elif separates:
            verdict = "separates, but ordering is INVERTED -> not tradeable as labelled"
        elif sm["kw_vs_null_p"] < 0.05 and sm["kw_sig_rate_pct"] > 5:
            verdict = ("marginal separation, does not survive multiplicity"
                       + ("" if right_sign else ", and wrong sign"))
        else:
            verdict = "indistinguishable from a random partition"
        print(f"  {name:<6} ({sm['mean_n_regimes']:.1f} regimes avg): "
              f"KW sig {sm['kw_sig_rate_pct']:.0f}% (p={sm['kw_vs_null_p']:.3f}), "
              f"rho {sm['mean_rho']:+.3f}")
        print(f"         -> {verdict}")
    print("\n  'cross' is the fair comparison against 'core': same feature count, so a")
    print("  difference cannot be explained by parameter count. 'both' fits 6 features")
    print("  with full covariance on 252 samples, so a null there is partly a")
    print("  sample-size statement rather than a pure statement about the features.")

    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=1)
    print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
