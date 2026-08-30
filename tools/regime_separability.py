"""
Does the regime labeling carry any out-of-sample information about forward returns?

Why this test
-------------
Every result so far has been mediated by the backtest: entry rules, confirmations,
cooldowns, exposure. Those layers can mask or manufacture effects. This script removes
them and asks the question the whole system rests on.

`RegimeDetector` assigns regime ids by ranking states on their **in-sample** mean return,
so id 0 is the most bullish state *on the training window*. `run_backtest` then treats ids
0..k as bullish. That is only meaningful if the ranking persists on unseen data.

So, per window, with regimes assigned causally:

  1. Rank correlation between regime id and realized forward return. If the in-sample
     ordering holds, this should be clearly NEGATIVE (low id -> high return).
  2. Kruskal-Wallis across regimes: do forward-return distributions differ at all?
  3. Does regime 0 actually outperform the window average out-of-sample?

Forward return is the bar-to-bar return realized AFTER the regime is observed
(`Close.pct_change().shift(-1)`), so nothing here is contaminated by hindsight.

A null result means the features cannot separate regimes in a way that predicts returns,
which would explain every downstream negative result and make further filter tuning
pointless.
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


def analyze_ticker(
    ticker: str,
    period_days: int = 3000,
    interval: str = "1d",
    is_bars: int = 252,
    oos_bars: int = 126,
    n_regimes=7,
    hmm_iter: int = 100,
) -> List[dict]:
    raw = fetch_data(ticker, period_days=period_days, interval=interval)
    df = engineer_features(raw)

    rows: List[dict] = []
    start, idx = 0, 0
    while start + is_bars + oos_bars <= len(df):
        is_start, is_end = start, start + is_bars
        oos_start, oos_end = is_end, is_end + oos_bars
        idx += 1
        try:
            det = RegimeDetector(n_regimes=n_regimes, n_iter=hmm_iter, random_state=42)
            det.train(df.iloc[is_start:is_end])

            # Causal labels for the OOS window.
            labeled = det.filtered_regimes(df.iloc[:oos_end])
            oos = labeled.iloc[oos_start:oos_end].copy()

            # Return realized AFTER the regime is observed.
            oos["fwd"] = oos["Close"].pct_change().shift(-1) * 100.0
            oos = oos.dropna(subset=["fwd", "regime_id"])
            if len(oos) < 20:
                start += oos_bars
                continue

            groups, means = [], {}
            for rid, g in oos.groupby("regime_id"):
                if len(g) >= 3:
                    groups.append(g["fwd"].values)
                    means[int(rid)] = float(g["fwd"].mean())
            if len(groups) < 2:
                start += oos_bars
                continue

            # 1. Rank correlation: regime id vs realized forward return, per bar.
            rho, rho_p = stats.spearmanr(oos["regime_id"].values, oos["fwd"].values)

            # 2. Do the distributions differ at all?
            kw_h, kw_p = stats.kruskal(*groups)

            # 3. Did regime 0 beat the window average?
            overall = float(oos["fwd"].mean())
            r0 = means.get(0)

            # Ordering quality: correlation between id and per-regime mean return.
            ids = sorted(means)
            ord_rho = np.nan
            if len(ids) >= 3:
                ord_rho, _ = stats.spearmanr(ids, [means[i] for i in ids])

            rows.append({
                "ticker": ticker, "window": idx,
                "start": str(oos.index[0].date()), "end": str(oos.index[-1].date()),
                "n_bars": int(len(oos)), "n_regimes_present": len(means),
                "spearman_rho": float(rho), "spearman_p": float(rho_p),
                "kruskal_p": float(kw_p),
                "order_rho": float(ord_rho) if ord_rho == ord_rho else None,
                "regime0_mean": r0, "overall_mean": overall,
                "regime0_beat": (None if r0 is None else bool(r0 > overall)),
            })
        except Exception as exc:  # noqa: BLE001
            print(f"  [{ticker}] window {idx}: {exc}", file=sys.stderr)
        start += oos_bars
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", default="SPY,QQQ,NVDA,AAPL,XLF")
    ap.add_argument("--period-days", type=int, default=3000)
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--regimes", default="7")
    ap.add_argument("--out", default="/home/user/workspace/separability.json")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    n_regimes = args.regimes if args.regimes == "auto" else int(args.regimes)

    all_rows: List[dict] = []
    for t in tickers:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rows = analyze_ticker(t, period_days=args.period_days,
                                  interval=args.interval, n_regimes=n_regimes)
        all_rows += rows
        print(f"  {t}: {len(rows)} windows analyzed", file=sys.stderr)

    if not all_rows:
        print("No windows analyzed.", file=sys.stderr)
        return 1

    df = pd.DataFrame(all_rows)

    print("\n" + "=" * 76)
    print("REGIME -> FORWARD RETURN, OUT OF SAMPLE")
    print(f"n_regimes={args.regimes}   {len(df)} windows across {len(tickers)} tickers")
    print("=" * 76)

    print("\nPer ticker:")
    hdr = (f"{'ticker':<8}{'win':>4}{'medRho':>9}{'rho<0':>7}"
           f"{'KW p<.05':>10}{'r0 beat':>9}{'medOrdRho':>11}")
    print(hdr); print("-" * len(hdr))
    for t, g in df.groupby("ticker"):
        neg = 100.0 * (g["spearman_rho"] < 0).mean()
        kw = 100.0 * (g["kruskal_p"] < 0.05).mean()
        beat = 100.0 * g["regime0_beat"].dropna().mean() if g["regime0_beat"].notna().any() else float("nan")
        ordr = g["order_rho"].dropna().median() if g["order_rho"].notna().any() else float("nan")
        print(f"{t:<8}{len(g):>4}{g['spearman_rho'].median():>9.3f}{neg:>6.0f}%"
              f"{kw:>9.0f}%{beat:>8.0f}%{ordr:>11.3f}")

    print("\nPooled:")
    rho = df["spearman_rho"].dropna().values
    ordr = df["order_rho"].dropna().values
    beat = df["regime0_beat"].dropna().astype(bool)

    t_rho, p_rho = stats.ttest_1samp(rho, 0.0)
    print(f"  bar-level rho (id vs fwd return): mean {rho.mean():+.4f}  "
          f"median {np.median(rho):+.4f}  t={t_rho:.2f}  p={p_rho:.3f}")
    print(f"    fraction negative (expected direction): "
          f"{100.0 * (rho < 0).mean():.0f}%  "
          f"sign test p={stats.binomtest(int((rho < 0).sum()), len(rho), 0.5).pvalue:.3f}")

    if len(ordr):
        t_o, p_o = stats.ttest_1samp(ordr, 0.0)
        print(f"  regime-mean ordering rho          : mean {ordr.mean():+.4f}  "
              f"median {np.median(ordr):+.4f}  t={t_o:.2f}  p={p_o:.3f}")
        print(f"    fraction negative: {100.0 * (ordr < 0).mean():.0f}%  "
              f"sign test p={stats.binomtest(int((ordr < 0).sum()), len(ordr), 0.5).pvalue:.3f}")

    kwsig = 100.0 * (df['kruskal_p'] < 0.05).mean()
    print(f"  Kruskal-Wallis p<0.05             : {kwsig:.0f}% of windows "
          f"(5% expected under the null)")
    if len(beat):
        print(f"  regime 0 beat window average      : {100.0 * beat.mean():.0f}% "
              f"of windows  sign test "
              f"p={stats.binomtest(int(beat.sum()), len(beat), 0.5).pvalue:.3f}")

    print("\nInterpretation")
    print("-" * 76)
    print("The labeling asserts id 0 = most bullish. For that to be tradeable, bar-level")
    print("rho and the ordering rho must both be reliably NEGATIVE out of sample, and")
    print("regime 0 should beat the window average well above 50% of the time.")

    json.dump(all_rows, open(args.out, "w"), indent=1)
    print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
