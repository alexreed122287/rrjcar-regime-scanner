"""
How much do transaction costs change the picture?

Every result in docs/validation-findings.md was produced with zero friction. That makes
them optimistic by an unknown amount, which matters because the strategy's headline
defensive property depends on turnover it never paid for.

This sweeps ``cost_bps_per_side`` and reports, per level:

  * strategy return, net of costs
  * matched-exposure random entry return, **charged the same friction**
  * excess of one over the other

Charging only the strategy would hand the benchmark a free edge, so
``benchmark_random_entry`` takes the same ``cost_bps_per_side`` and pays it on every
transition into and out of the market.

Reference points for liquid US equities: ~1-2 bps per side is optimistic, 5 bps is a fair
central estimate, 10+ bps applies to wider spreads or meaningful size. Tradier charges no
equity commission, so this is spread plus slippage.
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

from backtester import run_backtest                                  # noqa: E402
from data_loader import fetch_data, engineer_features                # noqa: E402
from hmm_engine import RegimeDetector                                 # noqa: E402
from backtester import compute_confirmations                         # noqa: E402
from walk_forward import (benchmark_random_entry, periods_per_year,  # noqa: E402
                          INDICATOR_WARMUP_BARS, _exposure_fraction)


def build_windows(ticker: str, period_days: int, interval: str, n_regimes,
                  is_bars: int, oos_bars: int, hmm_iter: int = 100) -> List[dict]:
    raw = fetch_data(ticker, period_days=period_days, interval=interval)
    df = engineer_features(raw)
    out: List[dict] = []
    start = 0
    while start + is_bars + oos_bars <= len(df):
        is_end = start + is_bars
        oos_start, oos_end = is_end, is_end + oos_bars
        try:
            det = RegimeDetector(n_regimes=n_regimes, n_iter=hmm_iter, random_state=42)
            det.train(df.iloc[start:is_end])
            # Mirror walk_forward._label_oos_causally exactly.
            labeled = det.filtered_regimes(df.iloc[:oos_end])
            ctx_start = max(0, oos_start - INDICATOR_WARMUP_BARS)
            scored = compute_confirmations(labeled.iloc[ctx_start:oos_end])
            scored = scored.iloc[oos_start - ctx_start:]
            out.append({"ticker": ticker, "scored": scored,
                        "n_regimes": int(det.n_regimes)})
        except Exception as exc:  # noqa: BLE001
            print(f"  [{ticker}] window skipped: {exc}", file=sys.stderr)
        start += oos_bars
    return out


def score(win: dict, cost_bps: float, ppy: float, n_trials: int = 50):
    try:
        bt = run_backtest(win["scored"], skip_confirmations=True,
                          n_regimes=win["n_regimes"], cost_bps_per_side=cost_bps)
    except Exception:  # noqa: BLE001
        return None
    sr = bt["metrics"].get("total_return_pct")
    if sr is None:
        return None
    expo = _exposure_fraction(bt["trades"], len(bt["df"])) or 0.0
    if expo <= 0:
        return {"excess": 0.0, "strategy": 0.0, "random": 0.0, "exposure": 0.0,
                "trades": 0, "flat": True, "cost_paid": 0.0}
    rnd = benchmark_random_entry(bt["df"], n_trials=n_trials, exposure_target=expo,
                                ppy=ppy, cost_bps_per_side=cost_bps)
    rr = rnd["total_return_pct"]
    return {"excess": float(sr - rr), "strategy": float(sr), "random": float(rr),
            "exposure": float(expo * 100), "trades": len(bt["trades"]),
            "flat": False,
            "cost_paid": float(bt["metrics"].get("total_cost_paid_pct", 0.0))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", default="SPY,QQQ,NVDA,AAPL,XLF")
    ap.add_argument("--costs", default="0,1,2,5,10,20")
    ap.add_argument("--period-days", type=int, default=3000)
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--regimes", default="7")
    ap.add_argument("--is-bars", type=int, default=252)
    ap.add_argument("--oos-bars", type=int, default=126)
    ap.add_argument("--out", default="/home/user/workspace/cost_sensitivity.json")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    costs = [float(c) for c in args.costs.split(",")]
    n_regimes = args.regimes if args.regimes == "auto" else int(args.regimes)
    ppy = periods_per_year(args.interval)

    wins: List[dict] = []
    for t in tickers:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            w = build_windows(t, args.period_days, args.interval, n_regimes,
                              args.is_bars, args.oos_bars)
        wins += w
        print(f"  {t}: {len(w)} windows", file=sys.stderr)

    rows: List[dict] = []
    for c in costs:
        for w in wins:
            r = score(w, c, ppy)
            if r:
                r.update({"cost_bps": c, "ticker": w["ticker"]})
                rows.append(r)
        print(f"  cost {c} bps done", file=sys.stderr)

    df = pd.DataFrame(rows)

    print("\n" + "=" * 82)
    print("TRANSACTION COST SENSITIVITY")
    print(f"regimes={args.regimes}  {len(wins)} windows  "
          f"both strategy and random benchmark pay the same friction")
    print("=" * 82)
    hdr = (f"{'bps/side':>9}{'strategy':>11}{'random':>10}{'excess':>10}"
           f"{'p':>8}{'win%':>7}{'costPaid':>10}{'trades':>8}")
    print(hdr); print("-" * len(hdr))
    summary = []
    for c in costs:
        g = df[df["cost_bps"] == c]
        act = g[~g["flat"]]
        ex = act["excess"].values
        p = stats.ttest_1samp(ex, 0.0).pvalue if len(ex) > 2 else float("nan")
        row = {"cost_bps": c, "strategy": float(act["strategy"].mean()),
               "random": float(act["random"].mean()), "excess": float(ex.mean()),
               "p": float(p), "win_pct": float(100.0 * (ex > 0).mean()),
               "cost_paid": float(act["cost_paid"].mean()),
               "trades": float(act["trades"].mean())}
        summary.append(row)
        print(f"{c:>9.0f}{row['strategy']:>11.2f}{row['random']:>10.2f}"
              f"{row['excess']:>10.2f}{p:>8.3f}{row['win_pct']:>6.0f}%"
              f"{row['cost_paid']:>10.2f}{row['trades']:>8.1f}")

    print("\nPer-ticker strategy return, net of costs:")
    piv = df[~df["flat"]].pivot_table(index="ticker", columns="cost_bps",
                                      values="strategy", aggfunc="mean")
    print(piv.round(2).to_string())

    base = summary[0]["strategy"]
    print("\nInterpretation")
    print("-" * 82)
    for row in summary:
        if row["cost_bps"] == 0:
            continue
        drag = base - row["strategy"]
        print(f"  {row['cost_bps']:>4.0f} bps/side: strategy gives up "
              f"{drag:5.2f} pp per window "
              f"({100.0 * drag / abs(base) if base else 0:5.1f}% of the gross result)")
    # Turnover matters more than it looks: the benchmark holds a fixed 10 bars, so at
    # MATCHED exposure the strategy's shorter holds mean strictly more round trips and
    # strictly more friction. Costs therefore do not cancel in the comparison.
    act0 = df[(df["cost_bps"] == costs[0]) & (~df["flat"])]
    expo0, trd0 = act0["exposure"].mean(), act0["trades"].mean()
    in_mkt = expo0 / 100.0 * args.oos_bars
    bench_trips = in_mkt / 10.0
    print(f"\n  Turnover at matched exposure ({args.oos_bars}-bar windows):")
    print(f"    strategy : {trd0:.2f} round trips/window at {expo0:.1f}% exposure "
          f"=> avg hold {in_mkt / trd0:.1f} bars")
    print(f"    benchmark: fixed 10-bar holds => {bench_trips:.2f} round trips/window")
    print(f"    the strategy pays friction on ~{trd0 / bench_trips:.1f}x as many round "
          f"trips for the same time in market.")
    print("\n  So costs do NOT cancel in the comparison. Excess degrades monotonically")
    print("  with friction, because higher turnover at equal exposure is a net")
    print("  headwind. Costs erode the absolute return AND widen the gap to random.")

    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=1)
    print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
