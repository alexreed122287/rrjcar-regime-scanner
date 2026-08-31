#!/usr/bin/env python3
"""
Is the HMM's capital preservation real -- and does it beat a 200-day moving average?

The repo's headline defensive result is SPY -4.01% against buy-and-hold's -18.78% in 2022 at
single-digit exposure. That is ONE ticker in ONE year with NO benchmark, which makes it an
anecdote rather than a finding. It is also, after eight negative return tests, the only claim
the system has left, so it deserves the same treatment the return claims got.

The right question is not "did the HMM reduce drawdown". Any filter that spends most of its
time in cash reduces drawdown; being flat is not a skill. The question is whether it reduces
drawdown MORE than trivial alternatives that cost nothing to implement:

    hmm             long when the regime is in the bullish set, else flat
    sma200          long when close > its 200-day simple moving average, else flat
    vol_target      exposure = min(1, target_vol / trailing_vol), the standard naive recipe
    random_matched  flat/long at random, with the SAME average exposure as the HMM
    buy_hold        always long

random_matched is the load-bearing benchmark. It answers "would a coin that trades this often
have done as well?", and it is averaged over many draws so a single lucky sequence cannot
carry it. If the HMM cannot beat a coin with its own exposure budget, then its drawdown
reduction is a mechanical consequence of being out of the market, not evidence of timing.

Metrics are reported per window and pooled, with a block bootstrap over windows for the
differences that matter. Drawdown reduction is also expressed as reduction PER UNIT OF RETURN
GIVEN UP, since a filter that halves drawdown by halving returns has achieved nothing a
smaller position size could not.

Usage:
    python tools/drawdown_benchmark.py --tickers SPY --save-obs dd/SPY.csv
    python tools/drawdown_benchmark.py --from-obs 'dd/*.csv' --json docs/drawdown_benchmark.json

Method notes
------------
* Same walk-forward discipline as tools/regime_ranking.py and tools/regime_volatility.py:
  expanding train to oos_start, OOS labels from ``filtered_regimes`` so bar t's label uses
  only bars 0..t. The SMA and vol-target rules are likewise strictly trailing.
* Exposure decided at bar t earns bar t+1's return. No same-bar execution.
* Costs are charged on every change in exposure at ``--cost-bps`` per side (default 5), which
  matters because the HMM flips far more often than a 200-day average does. A cost-free
  comparison would flatter the noisiest rule.
* Max drawdown is computed on each window's equity curve, so it is a within-window figure and
  not comparable to a multi-year peak-to-trough number.
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
from hmm_engine import RegimeDetector, regime_sets

DEFAULT_TICKERS = ["SPY", "QQQ", "NVDA", "AAPL", "XLF"]
STRATEGIES = ["hmm", "sma200", "vol_target", "random_matched", "buy_hold"]
ANNUALIZE = np.sqrt(252.0)
N_RANDOM_DRAWS = 200


def _metrics(exposure: np.ndarray, fwd_ret: np.ndarray, cost_bps: float) -> dict:
    """Equity metrics for an exposure path. exposure[i] earns fwd_ret[i] (already t+1)."""
    e = np.nan_to_num(exposure, nan=0.0)
    turn = np.abs(np.diff(np.concatenate([[0.0], e])))
    net = e * fwd_ret - turn * (cost_bps / 10000.0)
    eq = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(eq)
    dd = float(np.min(eq / peak - 1.0)) if len(eq) else 0.0
    sd = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
    return {
        "total_return": float(eq[-1] - 1.0) if len(eq) else 0.0,
        "max_drawdown": dd,
        "vol_ann": sd * ANNUALIZE,
        "sharpe": float(np.mean(net) / sd * ANNUALIZE) if sd > 0 else 0.0,
        "exposure": float(np.mean(e)),
        "turnover": float(np.sum(turn)),
    }


def collect(tickers, period_days, interval, n_regimes, train_bars, oos_bars, step_bars,
            n_iter, cost_bps, target_vol, seed):
    """One row per (ticker, window, strategy)."""
    rows = []
    rng = np.random.default_rng(seed)
    for tkr in tickers:
        raw = fetch_data(tkr, period_days=period_days, interval=interval)
        if raw is None or raw.empty:
            print(f"  {tkr}: no data, skipped", file=sys.stderr)
            continue
        feats = engineer_features(raw)
        rets = np.asarray(feats["returns"].values, dtype=float)
        close = np.asarray(feats["Close"].values, dtype=float)
        fwd = np.full(len(rets), np.nan)
        fwd[:-1] = rets[1:]
        sma = pd.Series(close).rolling(200).mean().values
        trail = (pd.Series(rets).rolling(20).std(ddof=1) * ANNUALIZE).values

        n = len(feats)
        starts = list(range(train_bars, n - oos_bars + 1, step_bars))
        if not starts:
            print(f"  {tkr}: only {n} bars, too short", file=sys.stderr)
            continue
        print(f"  {tkr}: {n} bars, {len(starts)} windows", file=sys.stderr)

        for w, oos_start in enumerate(starts):
            oos_end = oos_start + oos_bars
            sl = slice(oos_start, oos_end)
            f = fwd[sl]
            ok = np.isfinite(f)
            if ok.sum() < oos_bars * 0.9:
                continue
            f = np.nan_to_num(f, nan=0.0)

            det = RegimeDetector(n_regimes=n_regimes, n_iter=n_iter)
            try:
                det.train(feats.iloc[:oos_start])
            except Exception as exc:
                print(f"    {tkr} w{w}: fit failed ({type(exc).__name__}), skipped",
                      file=sys.stderr)
                continue
            labeled = det.filtered_regimes(feats.iloc[:oos_end])
            ids = np.asarray(labeled["regime_id"].values[sl], dtype=float)
            bull = set(regime_sets(int(det.n_regimes))["bullish"])
            e_hmm = np.array([1.0 if (np.isfinite(r) and int(r) in bull) else 0.0
                              for r in ids])

            e_sma = np.where(np.isfinite(sma[sl]) & (close[sl] > sma[sl]), 1.0, 0.0)
            tv = trail[sl]
            e_vt = np.where(np.isfinite(tv) & (tv > 1e-9),
                            np.clip(target_vol / np.where(tv > 1e-9, tv, np.nan), 0.0, 1.0),
                            0.0)
            e_vt = np.nan_to_num(e_vt, nan=0.0)

            paths = {"hmm": e_hmm, "sma200": e_sma, "vol_target": e_vt,
                     "buy_hold": np.ones(len(f))}
            for name, e in paths.items():
                m = _metrics(e, f, cost_bps)
                m.update({"ticker": tkr, "window": w, "strategy": name})
                rows.append(m)

            # Matched random: same average exposure as the HMM, averaged over many draws so
            # one lucky path cannot decide the comparison.
            p = float(np.mean(e_hmm))
            draws = [_metrics((rng.random(len(f)) < p).astype(float), f, cost_bps)
                     for _ in range(N_RANDOM_DRAWS)]
            avg = {k: float(np.mean([d[k] for d in draws])) for k in draws[0]}
            avg.update({"ticker": tkr, "window": w, "strategy": "random_matched"})
            rows.append(avg)
    return pd.DataFrame(rows)


def _paired_bootstrap(df: pd.DataFrame, metric: str, a: str, b: str,
                      n_boot: int = 4000, seed: int = 0) -> dict:
    """Bootstrap the paired per-window difference a - b, resampling whole windows."""
    wide = df.pivot_table(index=["ticker", "window"], columns="strategy", values=metric)
    if a not in wide or b not in wide:
        return {"usable": False}
    d = (wide[a] - wide[b]).dropna().values
    if len(d) < 5:
        return {"usable": False}
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    try:
        w_p = float(stats.wilcoxon(d)[1])
    except Exception:
        w_p = float("nan")
    return {"usable": True, "n_windows": int(len(d)), "mean_diff": round(float(d.mean()), 5),
            "ci_low": round(float(np.percentile(boot, 2.5)), 5),
            "ci_high": round(float(np.percentile(boot, 97.5)), 5),
            "frac_a_better": round(float((d > 0).mean()), 4), "wilcoxon_p": w_p}


def evaluate(df: pd.DataFrame) -> dict:
    pooled = {}
    for s, g in df.groupby("strategy"):
        pooled[s] = {k: round(float(g[k].mean()), 5) for k in
                     ["total_return", "max_drawdown", "vol_ann", "sharpe", "exposure",
                      "turnover"]}
        # Drawdown reduction bought per unit of return surrendered, vs buy and hold.
        bh = df[df.strategy == "buy_hold"]
        if len(bh):
            dd_saved = float(g["max_drawdown"].mean() - bh["max_drawdown"].mean())
            ret_given = float(bh["total_return"].mean() - g["total_return"].mean())
            pooled[s]["dd_saved"] = round(dd_saved, 5)
            pooled[s]["return_given_up"] = round(ret_given, 5)
            pooled[s]["dd_saved_per_return_given_up"] = (
                round(dd_saved / ret_given, 3) if abs(ret_given) > 1e-9 else None)

    comps = {}
    for opp in ["sma200", "vol_target", "random_matched", "buy_hold"]:
        comps[f"hmm_vs_{opp}"] = {
            # Less negative drawdown is better, so hmm - opp > 0 favours the HMM.
            "max_drawdown": _paired_bootstrap(df, "max_drawdown", "hmm", opp),
            "total_return": _paired_bootstrap(df, "total_return", "hmm", opp),
            "sharpe": _paired_bootstrap(df, "sharpe", "hmm", opp),
        }
    return {"n_rows": int(len(df)), "pooled": pooled, "comparisons": comps,
            "per_ticker_drawdown": {
                t: {s: round(float(gg["max_drawdown"].mean()), 4)
                    for s, gg in g.groupby("strategy")}
                for t, g in df.groupby("ticker")}}


def report(res: dict) -> None:
    print("=" * 100)
    print("IS THE CAPITAL PRESERVATION REAL, AND DOES IT BEAT A 200-DAY MOVING AVERAGE?")
    print("=" * 100)
    print(f"\nPooled means across all windows and tickers ({res['n_rows']} strategy-windows):\n")
    print(f"  {'strategy':<16}{'return':>9}{'maxDD':>9}{'vol':>8}{'sharpe':>8}"
          f"{'expo':>7}{'turn':>8}{'DD saved/ret given up':>24}")
    for s in STRATEGIES:
        p = res["pooled"].get(s)
        if not p:
            continue
        ratio = p.get("dd_saved_per_return_given_up")
        print(f"  {s:<16}{p['total_return']*100:>8.2f}%{p['max_drawdown']*100:>8.2f}%"
              f"{p['vol_ann']*100:>7.1f}%{p['sharpe']:>8.2f}{p['exposure']*100:>6.0f}%"
              f"{p['turnover']:>8.1f}{(f'{ratio:.2f}' if ratio is not None else '--'):>24}")

    print("\nPaired per-window differences, bootstrapped over windows.")
    print("For maxDD a POSITIVE difference favours the HMM (shallower drawdown).\n")
    for name, block in res["comparisons"].items():
        print(f"  {name}")
        for metric, t in block.items():
            if not t.get("usable"):
                print(f"    {metric:<14} unusable")
                continue
            sig = "significant" if (t["ci_low"] > 0 or t["ci_high"] < 0) else "NOT significant"
            print(f"    {metric:<14} diff={t['mean_diff']:+.4f}  "
                  f"95% CI [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}]  "
                  f"HMM better in {t['frac_a_better']*100:.0f}% of {t['n_windows']} windows  "
                  f"({sig})")
        print()

    dd_sma = res["comparisons"]["hmm_vs_sma200"]["max_drawdown"]
    dd_rnd = res["comparisons"]["hmm_vs_random_matched"]["max_drawdown"]
    print("=" * 100)
    if dd_rnd.get("usable") and dd_rnd["ci_low"] > 0:
        print("The HMM's drawdown reduction BEATS a coin flip with the same exposure budget,")
        print("so it is doing more than merely sitting in cash.")
    else:
        print("The HMM does NOT beat a random filter holding the same average exposure.")
        print("Its drawdown reduction is therefore consistent with being out of the market")
        print("rather than with timing -- a smaller position size achieves the same thing.")
    if dd_sma.get("usable"):
        if dd_sma["ci_low"] > 0:
            print("It also beats a 200-day moving average on drawdown.")
        elif dd_sma["ci_high"] < 0:
            print("A 200-day moving average is SIGNIFICANTLY BETTER on drawdown. The HMM is")
            print("elaborate machinery losing to one line of pandas.")
        else:
            print("Against a 200-day moving average the difference is not significant: the")
            print("simple filter does the same job, and is far easier to reason about.")
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
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--target-vol", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
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
                     args.cost_bps, args.target_vol, args.seed)
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
