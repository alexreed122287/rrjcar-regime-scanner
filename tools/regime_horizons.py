"""
Do regimes carry forward information at longer holding horizons?

`tools/regime_separability.py` showed regimes carry no information about the NEXT BAR.
That leaves an obvious loophole: a regime could be uninformative at 1 day and still be
informative at 1-4 weeks. Regime models are usually motivated as medium-horizon
descriptions, so testing only h=1 would be an unfair dismissal.

This runs the same causal setup across h in {1, 5, 10, 20} bars.

Statistical care
----------------
Overlapping forward returns are strongly autocorrelated; using every bar at h=20 would
inflate significance badly. So observations are **subsampled with stride h**, making them
non-overlapping within a window.

That leaves ~6 observations per window at h=20, too few for a per-window test. Regime ids
are comparable across windows by construction (0 = most bullish in-sample), so the primary
test pools the non-overlapping observations across windows within each ticker, then
reports tickers as independent units. Per-window tests are still reported where n allows.

Direction convention: a tradeable ordering requires rho(regime_id, forward return) to be
NEGATIVE, because id 0 is supposed to be the most bullish state.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from data_loader import fetch_data, engineer_features   # noqa: E402
from hmm_engine import RegimeDetector                   # noqa: E402


def collect(ticker: str, horizons: List[int], period_days: int, interval: str,
            n_regimes, is_bars: int = 252, oos_bars: int = 126,
            hmm_iter: int = 100) -> Tuple[Dict, List[dict]]:
    """Return (pooled[h] -> (ids, fwds), per_window_rows)."""
    raw = fetch_data(ticker, period_days=period_days, interval=interval)
    df = engineer_features(raw)

    pooled: Dict[int, Tuple[List[float], List[float]]] = {h: ([], []) for h in horizons}
    rows: List[dict] = []

    start, idx = 0, 0
    while start + is_bars + oos_bars <= len(df):
        is_end = start + is_bars
        oos_start, oos_end = is_end, is_end + oos_bars
        idx += 1
        try:
            det = RegimeDetector(n_regimes=n_regimes, n_iter=hmm_iter, random_state=42)
            det.train(df.iloc[start:is_end])
            labeled = det.filtered_regimes(df.iloc[:oos_end])
            oos = labeled.iloc[oos_start:oos_end].copy()

            for h in horizons:
                # Forward h-bar return, realized strictly after the regime is observed.
                fwd = (oos["Close"].shift(-h) / oos["Close"] - 1.0) * 100.0
                sub = pd.DataFrame({"rid": oos["regime_id"], "fwd": fwd})
                # Stride h => non-overlapping observations.
                sub = sub.iloc[::h].dropna()
                if len(sub) < 4:
                    continue
                pooled[h][0].extend(sub["rid"].astype(int).tolist())
                pooled[h][1].extend(sub["fwd"].tolist())

                if len(sub) >= 20 and sub["rid"].nunique() >= 2:
                    rho, _ = stats.spearmanr(sub["rid"].values, sub["fwd"].values)
                    groups = [g["fwd"].values for _, g in sub.groupby("rid") if len(g) >= 3]
                    kw_p = stats.kruskal(*groups).pvalue if len(groups) >= 2 else np.nan
                    rows.append({"ticker": ticker, "window": idx, "horizon": h,
                                 "n": int(len(sub)), "rho": float(rho),
                                 "kruskal_p": float(kw_p) if kw_p == kw_p else None})
        except Exception as exc:  # noqa: BLE001
            print(f"  [{ticker}] window {idx}: {exc}", file=sys.stderr)
        start += oos_bars
    return pooled, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", default="SPY,QQQ,NVDA,AAPL,XLF")
    ap.add_argument("--horizons", default="1,5,10,20")
    ap.add_argument("--period-days", type=int, default=3000)
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--regimes", default="auto")
    ap.add_argument("--out", default="/home/user/workspace/horizons.json")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    horizons = [int(h) for h in args.horizons.split(",")]
    n_regimes = args.regimes if args.regimes == "auto" else int(args.regimes)

    per_ticker: Dict[str, Dict] = {}
    all_rows: List[dict] = []
    for t in tickers:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pooled, rows = collect(t, horizons, args.period_days, args.interval, n_regimes)
        per_ticker[t] = pooled
        all_rows += rows
        print(f"  {t}: done", file=sys.stderr)

    print("\n" + "=" * 78)
    print(f"REGIME -> FORWARD RETURN BY HORIZON   regimes={args.regimes}")
    print("Non-overlapping observations (stride = horizon). Pooled within ticker.")
    print("A tradeable ordering needs rho NEGATIVE.")
    print("=" * 78)

    summary: Dict[str, list] = {}
    for h in horizons:
        print(f"\n--- horizon = {h} bar(s) ---")
        hdr = f"{'ticker':<8}{'n':>6}{'rho':>9}{'rho p':>8}{'KW p':>8}{'r0 mean':>10}{'all mean':>10}"
        print(hdr); print("-" * len(hdr))
        rhos, entries = [], []
        for t in tickers:
            ids, fwds = per_ticker[t][h]
            if len(ids) < 20:
                print(f"{t:<8}{len(ids):>6}   insufficient observations")
                continue
            ids_a, fwd_a = np.array(ids), np.array(fwds)
            rho, rp = stats.spearmanr(ids_a, fwd_a)
            groups = [fwd_a[ids_a == u] for u in np.unique(ids_a)
                      if (ids_a == u).sum() >= 3]
            kwp = stats.kruskal(*groups).pvalue if len(groups) >= 2 else float("nan")
            r0 = fwd_a[ids_a == 0].mean() if (ids_a == 0).any() else float("nan")
            print(f"{t:<8}{len(ids):>6}{rho:>9.3f}{rp:>8.3f}{kwp:>8.3f}"
                  f"{r0:>10.2f}{fwd_a.mean():>10.2f}")
            rhos.append(rho)
            entries.append({"ticker": t, "n": int(len(ids)), "rho": float(rho),
                            "rho_p": float(rp),
                            "kruskal_p": float(kwp) if kwp == kwp else None,
                            "regime0_mean": float(r0) if r0 == r0 else None,
                            "overall_mean": float(fwd_a.mean())})
        summary[str(h)] = entries
        if rhos:
            r = np.array(rhos)
            neg = int((r < 0).sum())
            sp = stats.binomtest(neg, len(r), 0.5).pvalue
            tt = stats.ttest_1samp(r, 0.0)
            print(f"{'POOLED':<8}{'':>6}{r.mean():>9.3f}"
                  f"{tt.pvalue:>8.3f}        "
                  f"  tickers with intended (negative) sign: {neg}/{len(r)} (p={sp:.3f})")

    print("\n" + "=" * 78)
    print("VERDICT BY HORIZON")
    print("=" * 78)
    print(f"{'h':>4}{'mean rho':>11}{'intended sign':>16}{'KW sig tickers':>17}{'read':>10}")
    print("-" * 58)
    for h in horizons:
        e = summary[str(h)]
        if not e:
            continue
        r = np.array([x["rho"] for x in e])
        kws = sum(1 for x in e if x["kruskal_p"] is not None and x["kruskal_p"] < 0.05)
        read = "none" if kws == 0 else ("weak" if kws < 3 else "signal?")
        print(f"{h:>4}{r.mean():>11.3f}{f'{int((r<0).sum())}/{len(r)}':>16}"
              f"{f'{kws}/{len(e)}':>17}{read:>10}")

    json.dump({"summary": summary, "per_window": all_rows}, open(args.out, "w"), indent=1)
    print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
