"""
Entry-filter sweep with an honest tune/verify split.

Motivation
----------
Walk-forward validation showed the strategy does not beat exposure-matched random entry
at either 7 regimes or a BIC-selected 3-4. Regime count was ruled out as the binding
constraint, so the next candidate is the entry filter stack: the regime-posterior
confidence gate and the indicator confirmation count.

Method
------
The trap here is sweeping ~30 parameter combinations over 14 windows and reporting the
best one. With a per-window standard deviation around 6%, something always looks good.
So:

  * Windows are split CHRONOLOGICALLY. The earlier half is TUNE, the later half is
    VERIFY. Verify windows are strictly after tune windows, so there is no leakage.
  * The full surface is reported, not just the argmax.
  * Exactly ONE setting is carried to VERIFY, chosen on TUNE alone.
  * The HMM is fitted once per (ticker, window) and the labeled out-of-sample frames are
    cached, then reused across every parameter combination. The fit does not depend on
    backtest parameters, so this is a pure speedup and changes no result.

The scoring metric is mean excess return versus random entry at MATCHED exposure. Raw
return is not usable: tightening the filters cuts exposure, which mechanically cuts
return without saying anything about signal quality.
"""

from __future__ import annotations

import argparse
import io
import contextlib
import itertools
import json
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")

from data_loader import fetch_data, engineer_features          # noqa: E402
from hmm_engine import RegimeDetector                          # noqa: E402
from backtester import run_backtest, compute_confirmations     # noqa: E402
from walk_forward import (                                     # noqa: E402
    INDICATOR_WARMUP_BARS,
    benchmark_random_entry,
    _exposure_fraction,
    periods_per_year,
)


@dataclass
class CachedWindow:
    """One labeled out-of-sample window, reusable across parameter combinations."""
    ticker: str
    index: int
    oos_start: int
    scored: pd.DataFrame
    n_regimes: int
    start_date: str
    end_date: str


def build_windows(
    ticker: str,
    period_days: int = 3000,
    interval: str = "1d",
    is_bars: int = 252,
    oos_bars: int = 126,
    n_regimes=7,
    hmm_iter: int = 100,
    random_state: int = 42,
) -> List[CachedWindow]:
    """Fit the HMM once per window and cache the causally-labeled OOS frame."""
    raw = fetch_data(ticker, period_days=period_days, interval=interval)
    df = engineer_features(raw)

    out: List[CachedWindow] = []
    idx = 0
    start = 0
    while start + is_bars + oos_bars <= len(df):
        is_start, is_end = start, start + is_bars
        oos_start, oos_end = is_end, is_end + oos_bars
        idx += 1
        try:
            det = RegimeDetector(
                n_regimes=n_regimes, n_iter=hmm_iter, random_state=random_state
            )
            det.train(df.iloc[is_start:is_end])

            # Mirrors WalkForwardEngine._label_oos_causally exactly: causal regimes
            # over bar 0..oos_end, then confirmations over a warmup buffer that is
            # trimmed off, so the scored window keeps every one of its bars.
            ctx_start = max(0, oos_start - INDICATOR_WARMUP_BARS)
            labeled = det.filtered_regimes(df.iloc[:oos_end])
            ctx = compute_confirmations(labeled.iloc[ctx_start:oos_end])
            scored = ctx.iloc[oos_start - ctx_start:]
            if len(scored) >= 10:
                out.append(CachedWindow(
                    ticker=ticker, index=idx, oos_start=oos_start, scored=scored,
                    n_regimes=int(det.n_regimes),
                    start_date=str(scored.index[0].date()),
                    end_date=str(scored.index[-1].date()),
                ))
        except Exception as exc:  # noqa: BLE001
            print(f"  [{ticker}] window {idx} failed: {exc}", file=sys.stderr)
        start += oos_bars
    return out


def score_window(
    win: CachedWindow,
    min_confidence: float,
    min_confirmations: int,
    ppy: float,
    initial_capital: float = 100_000.0,
    n_trials: int = 50,
) -> Optional[dict]:
    """Excess return vs exposure-matched random entry for one window and setting."""
    try:
        bt = run_backtest(
            win.scored,
            initial_capital=initial_capital,
            skip_confirmations=True,
            n_regimes=win.n_regimes,
            min_confidence=min_confidence,
            min_confirmations=min_confirmations,
        )
    except Exception:  # noqa: BLE001
        return None

    strat_ret = bt["metrics"].get("total_return_pct")
    if strat_ret is None:
        return None

    expo = _exposure_fraction(bt["trades"], len(bt["df"])) or 0.0
    if expo <= 0:
        # No exposure means no bet was made. Excess is zero by construction, not a win.
        return {"excess": 0.0, "strategy": 0.0, "random": 0.0,
                "exposure": 0.0, "trades": 0, "flat": True}

    rnd = benchmark_random_entry(
        bt["df"], n_trials=n_trials, exposure_target=expo,
        initial_capital=initial_capital, ppy=ppy,
    )
    rnd_ret = rnd.get("total_return_pct", 0.0)
    return {
        "excess": float(strat_ret) - float(rnd_ret),
        "strategy": float(strat_ret),
        "random": float(rnd_ret),
        "exposure": expo * 100.0,
        "trades": bt["metrics"].get("total_trades", 0),
        "flat": False,
    }


def summarize(excesses: List[float]) -> dict:
    arr = np.array([e for e in excesses if e is not None], dtype=float)
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"),
                "t": float("nan"), "p": float("nan"), "wins": 0, "win_pct": 0.0}
    t = p = float("nan")
    if n > 1 and arr.std(ddof=1) > 0:
        t, p = stats.ttest_1samp(arr, 0.0)
    wins = int((arr > 0).sum())
    return {"n": n, "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if n > 1 else 0.0,
            "t": float(t), "p": float(p),
            "wins": wins, "win_pct": 100.0 * wins / n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", default="SPY,QQQ,NVDA,AAPL,XLF")
    ap.add_argument("--period-days", type=int, default=3000)
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--is-bars", type=int, default=252)
    ap.add_argument("--oos-bars", type=int, default=126)
    ap.add_argument("--regimes", default="7")
    ap.add_argument("--confidences", default="0.4,0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--confirmations", default="3,4,5,6,7")
    ap.add_argument("--out", default="/home/user/workspace/sweep_results.json")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    confs = [float(c) for c in args.confidences.split(",")]
    confirms = [int(c) for c in args.confirmations.split(",")]
    n_regimes = args.regimes if args.regimes == "auto" else int(args.regimes)
    ppy = periods_per_year(args.interval)

    # 1. Cache labeled windows (the expensive part, done once).
    print(f"Fitting windows for {len(tickers)} tickers "
          f"(HMM fitted once per window, reused across "
          f"{len(confs) * len(confirms)} settings)...", file=sys.stderr)
    cache: Dict[str, List[CachedWindow]] = {}
    for t in tickers:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cache[t] = build_windows(
                t, period_days=args.period_days, interval=args.interval,
                is_bars=args.is_bars, oos_bars=args.oos_bars,
                n_regimes=n_regimes,
            )
        print(f"  {t}: {len(cache[t])} windows", file=sys.stderr)

    n_win = min((len(v) for v in cache.values() if v), default=0)
    if n_win < 4:
        print("Not enough windows to split into tune/verify.", file=sys.stderr)
        return 1

    # 2. Chronological split. Verify windows come strictly after tune windows.
    split = n_win // 2
    print(f"\nChronological split: windows 1-{split} = TUNE, "
          f"{split + 1}-{n_win} = VERIFY", file=sys.stderr)

    grid = list(itertools.product(confs, confirms))
    tune_rows, verify_cache = [], {}

    for mc, mcf in grid:
        tune_ex, ver_ex, expos, trades = [], [], [], 0
        tune_active, ver_active, flats_tune = [], [], 0
        for t in tickers:
            for w in cache[t][:n_win]:
                r = score_window(w, mc, mcf, ppy)
                if r is None:
                    continue
                is_tune = w.index <= split
                (tune_ex if is_tune else ver_ex).append(r["excess"])
                if r["flat"]:
                    if is_tune:
                        flats_tune += 1
                else:
                    (tune_active if is_tune else ver_active).append(r["excess"])
                expos.append(r["exposure"])
                trades += r["trades"]
        row = summarize(tune_ex)
        act = summarize(tune_active)
        row.update({"min_confidence": mc, "min_confirmations": mcf,
                    "mean_exposure": float(np.mean(expos)) if expos else 0.0,
                    "trades": trades, "flat_windows": flats_tune,
                    "active_n": act["n"], "active_mean": act["mean"],
                    "active_p": act["p"], "active_win_pct": act["win_pct"]})
        tune_rows.append(row)
        verify_cache[(mc, mcf)] = {"all": ver_ex, "active": ver_active}

    # 3. Report the full TUNE surface.
    print("\n" + "=" * 78)
    print("TUNE SURFACE  (windows 1-%d)   metric = mean excess vs matched-exposure random"
          % split)
    print("=" * 78)
    print("'all' includes zero-exposure windows as 0.0 excess; 'active' counts only")
    print("windows where a trade actually happened. Flat windows are not evidence.")
    hdr = (f"{'conf':>5}{'confs':>6}{'n':>4}{'mean':>7}{'sd':>6}{'p':>7}{'win%':>6}"
           f"{'|':>3}{'actN':>6}{'actMean':>9}{'actP':>7}{'actWin%':>8}"
           f"{'|':>3}{'expo%':>7}{'trd':>5}{'flat':>6}")
    print(hdr); print("-" * len(hdr))
    for r in sorted(tune_rows, key=lambda x: -(x["active_mean"]
                    if not np.isnan(x["active_mean"]) else -1e9)):
        print(f"{r['min_confidence']:>5.2f}{r['min_confirmations']:>6}{r['n']:>4}"
              f"{r['mean']:>7.2f}{r['sd']:>6.2f}{r['p']:>7.3f}{r['win_pct']:>6.0f}"
              f"{'|':>3}{r['active_n']:>6}{r['active_mean']:>9.2f}{r['active_p']:>7.3f}"
              f"{r['active_win_pct']:>8.0f}"
              f"{'|':>3}{r['mean_exposure']:>7.1f}{r['trades']:>5}{r['flat_windows']:>6}")

    # 4. Pick ONE setting on TUNE only, then verify it once.
    usable = [r for r in tune_rows
              if r["mean_exposure"] > 1.0 and r["active_n"] >= max(4, r["n"] // 3)
              and not np.isnan(r["active_mean"])]
    if not usable:
        print("\nNo setting kept meaningful exposure and sample size; nothing to verify.")
        return 1
    best = max(usable, key=lambda x: x["active_mean"])
    key = (best["min_confidence"], best["min_confirmations"])
    v = summarize(verify_cache[key]["all"])
    v_act = summarize(verify_cache[key]["active"])
    baseline = next((r for r in tune_rows
                     if r["min_confidence"] == 0.5 and r["min_confirmations"] == 5), None)

    print("\n" + "=" * 78)
    print("VERIFY  (held-out later windows, this setting evaluated exactly once)")
    print("=" * 78)
    print(f"Selected on TUNE (by active mean): min_confidence={key[0]:.2f}, "
          f"min_confirmations={key[1]}")
    print(f"  TUNE   all: n={best['n']:<3} mean={best['mean']:+.2f}%  p={best['p']:.3f}"
          f"   active: n={best['active_n']:<3} mean={best['active_mean']:+.2f}%  "
          f"p={best['active_p']:.3f}")
    print(f"  VERIFY all: n={v['n']:<3} mean={v['mean']:+.2f}%  p={v['p']:.3f}"
          f"   active: n={v_act['n']:<3} mean={v_act['mean']:+.2f}%  "
          f"p={v_act['p']:.3f}")
    if baseline:
        bv = summarize(verify_cache[(0.5, 5)]["all"])
        bva = summarize(verify_cache[(0.5, 5)]["active"])
        print("\nShipped default (0.50 / 5) for reference:")
        print(f"  TUNE   all: n={baseline['n']:<3} mean={baseline['mean']:+.2f}%  "
              f"p={baseline['p']:.3f}   active: n={baseline['active_n']:<3} "
              f"mean={baseline['active_mean']:+.2f}%  p={baseline['active_p']:.3f}")
        print(f"  VERIFY all: n={bv['n']:<3} mean={bv['mean']:+.2f}%  p={bv['p']:.3f}"
              f"   active: n={bva['n']:<3} mean={bva['mean']:+.2f}%  p={bva['p']:.3f}")

    drop = best["active_mean"] - v_act["mean"]
    print(f"\nTune -> verify decay: {drop:+.2f} percentage points.")
    print("A large positive decay means the tuned setting was fitting noise.")

    json.dump({"tune": tune_rows,
               "verify": {f"{k[0]}_{k[1]}": {"all": summarize(x["all"]),
                                             "active": summarize(x["active"])}
                          for k, x in verify_cache.items()},
               "selected": {"min_confidence": key[0], "min_confirmations": key[1]},
               "split": split, "n_windows": n_win, "tickers": tickers},
              open(args.out, "w"), indent=1)
    print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
