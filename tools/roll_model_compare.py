#!/usr/bin/env python3
"""
Does pricing the roll legs change the conclusion?

Background: ``strategy_v2`` used to credit a flat +0.5% of stock price per roll up (and
+0.3% per roll out, on losing positions, which was the wrong sign) while continuing to
compound full capital on the underlying. Measured across five tickers, those credits were a
median 41% of the reported return -- see ``docs/validation-findings.md``.

A roll up is not income. You sell a call, buy one struck higher, pocket the difference, and
walk away with less delta. The cash and the lost participation are the same transaction.
``roll_model="priced"`` now models both with Black-Scholes (``options_pricing.py``).

Configurations:

    legacy_flat      old default: flat +0.5 / +0.3 credits, stock accounting
    flat_zero        credits switched off, stock accounting
    priced_norolls   priced legs, rolling disabled  (isolates theta and sizing)
    priced           new default: priced legs, up to 3 rolls

Sizing note: the priced modes size the position so its dollar delta equals capital, holding
the balance in cash. Spending all capital on premium at these strikes is roughly 5x levered,
and comparing a 5x book against buy-and-hold would flatter the strategy for a reason that has
nothing to do with the signal. "Stock replacement" means matching exposure, not multiplying it.

Usage:
    python tools/roll_model_compare.py
    python tools/roll_model_compare.py --tickers SPY QQQ --json out.json

Caveat up front: run_backtest_v2 is not reachable from walk_forward, so these are in-sample
full-history backtests -- the basis the API uses, and therefore the right basis for "what does
the dashboard show", but NOT out-of-sample evidence. Nothing here says the strategy works.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from data_loader import fetch_data, engineer_features
from hmm_engine import RegimeDetector
from strategy_v2 import run_backtest_v2

DEFAULT_TICKERS = ["SPY", "QQQ", "NVDA", "AAPL", "XLF"]

CONFIGS = {
    "legacy_flat":    dict(roll_model="flat", roll_up_credit_pct=0.5,
                           roll_out_credit_pct=0.3),
    "flat_zero":      dict(roll_model="flat", roll_up_credit_pct=0.0,
                           roll_out_credit_pct=0.0),
    "priced_norolls": dict(roll_model="priced", max_rolls=0),
    "priced":         dict(roll_model="priced", max_rolls=3),
}


def run_one(scored, cfg, min_confs, cost_bps, n_regimes) -> dict:
    r = run_backtest_v2(scored, min_confirmations=min_confs, n_regimes=n_regimes,
                        cost_bps_per_side=cost_bps, **cfg)
    m, trades = r["metrics"], r["trades"]
    return {
        "total_return_pct": float(m.get("total_return_pct", 0.0)),
        "sharpe_ratio": float(m.get("sharpe_ratio", 0.0)),
        "max_drawdown_pct": float(m.get("max_drawdown_pct", 0.0)),
        "win_rate": float(m.get("win_rate", 0.0)),
        "n_trades": len(trades),
        "total_rolls": int(m.get("total_rolls", 0)),
        "roll_credits_pct": float(m.get("total_roll_credits_pct", 0.0)),
        "roll_cash_pct": float(m.get("total_roll_cash_pct", 0.0)),
        "roll_cost_pct": float(m.get("total_roll_cost_pct", 0.0)),
        "expiries": sum(1 for t in trades if "expiry" in str(t.get("exit_reason", ""))),
        "buyhold_return_pct": float(m.get("buyhold_return_pct", 0.0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--period-days", type=int, default=3000)
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--regimes", type=int, default=7)
    ap.add_argument("--min-confs", type=int, default=6)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    n_regimes = int(args.regimes)
    rows = []
    for tkr in args.tickers:
        raw = fetch_data(tkr, period_days=args.period_days, interval=args.interval)
        if raw is None or raw.empty:
            print(f"  {tkr}: no data, skipped", file=sys.stderr)
            continue
        scored = RegimeDetector(n_regimes=n_regimes, n_iter=60).train(engineer_features(raw))
        for name, cfg in CONFIGS.items():
            res = run_one(scored, cfg, args.min_confs, args.cost_bps, n_regimes)
            res.update(ticker=tkr, config=name)
            rows.append(res)

    if not rows:
        print("No results.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    views = {n: df[df.config == n].set_index("ticker") for n in CONFIGS}

    print()
    print("=" * 88)
    print(f"ROLL MODEL COMPARISON   (cost {args.cost_bps:g} bps/side, {n_regimes} regimes)")
    print("In-sample full-history backtests -- the same basis the API uses.")
    print("=" * 88)

    print("\nTotal return by configuration (%):")
    print(f"{'ticker':<8}{'legacy':>10}{'flat0':>10}{'priced_nr':>11}{'priced':>10}"
          f"{'rolls':>7}{'cash%':>8}{'B&H':>10}")
    print("-" * 74)
    for t in views["priced"].index:
        print(f"{t:<8}{views['legacy_flat'].loc[t,'total_return_pct']:>10.2f}"
              f"{views['flat_zero'].loc[t,'total_return_pct']:>10.2f}"
              f"{views['priced_norolls'].loc[t,'total_return_pct']:>11.2f}"
              f"{views['priced'].loc[t,'total_return_pct']:>10.2f}"
              f"{views['priced'].loc[t,'total_rolls']:>7}"
              f"{views['priced'].loc[t,'roll_cash_pct']:>8.1f}"
              f"{views['priced'].loc[t,'buyhold_return_pct']:>10.2f}")

    summary = {}
    print(f"\n{'config':<16}{'mean ret':>10}{'mean Sh':>9}{'mean DD':>9}"
          f"{'>0':>5}{'>B&H':>6}")
    print("-" * 55)
    for n, v in views.items():
        beat = int((v["total_return_pct"] > v["buyhold_return_pct"]).sum())
        pos = int((v["total_return_pct"] > 0).sum())
        summary[n] = {
            "mean_return_pct": float(v["total_return_pct"].mean()),
            "mean_sharpe": float(v["sharpe_ratio"].mean()),
            "mean_max_drawdown_pct": float(v["max_drawdown_pct"].mean()),
            "n_positive": pos, "n_beat_buyhold": beat, "n_tickers": len(v),
        }
        print(f"{n:<16}{v['total_return_pct'].mean():>10.2f}{v['sharpe_ratio'].mean():>9.2f}"
              f"{v['max_drawdown_pct'].mean():>9.2f}{pos:>5}{beat:>6}")

    legacy = summary["legacy_flat"]["mean_return_pct"]
    pr = summary["priced"]["mean_return_pct"]
    nr = summary["priced_norolls"]["mean_return_pct"]

    print("\nInterpretation")
    print("-" * 88)
    print(f"  Legacy flat credit : {legacy:+.2f}% mean")
    print(f"  Priced legs        : {pr:+.2f}% mean   "
          f"({pr - legacy:+.2f} pp vs legacy)")
    print(f"  Priced, no rolling : {nr:+.2f}% mean   "
          f"(rolling is worth {pr - nr:+.2f} pp once priced)")
    if pr < nr:
        print("  Rolling now SUBTRACTS value, which is the expected sign: taking cash off")
        print("  the table caps participation in the move that triggered the roll.")
    elif pr > nr:
        print("  Rolling still adds value once priced -- worth a closer look, since the")
        print("  de-risking should cost return in a rising market.")
    beat = summary["priced"]["n_beat_buyhold"]
    print(f"\n  Buy-and-hold beaten on {beat}/{summary['priced']['n_tickers']} tickers "
          f"under the priced model.")
    print("  In-sample. Says nothing about whether the strategy works.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"rows": rows, "summary": summary,
             "params": {"tickers": args.tickers, "period_days": args.period_days,
                        "interval": args.interval, "n_regimes": n_regimes,
                        "min_confs": args.min_confs, "cost_bps": args.cost_bps},
             "configs": {k: {kk: str(vv) for kk, vv in c.items()}
                         for k, c in CONFIGS.items()}}, indent=2))
        print(f"\nSaved: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
