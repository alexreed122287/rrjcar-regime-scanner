#!/usr/bin/env python3
"""
How much of strategy_v2's reported return is the unconditional roll credit?

strategy_v2 is the default strategy behind ``GET /backtest/{symbol}``, so its numbers are
what the dashboard shows. It hands out two credits that are not derived from option pricing:

    ROLL_UP    +0.5% of stock price, when price >= effective_entry + 1 ATR
    ROLL_OUT   +0.3% of stock price, when a position is past its time stop AT A LOSS

up to 3 per trade. Neither depends on strike, moneyness, implied vol, or time to expiry,
and nothing is surrendered in return.

ROLL_UP at least has the right sign: closing a long call and reopening at a higher strike
is genuinely a net credit. But the model keeps compounding full capital on the underlying
afterwards, so it never pays for the delta reduction a real roll up costs.

ROLL_OUT has the WRONG sign. Rolling a long call to a later expiry buys time value, which
is a debit. And because it fires only on a losing position at the time stop, it pays cash
*and* defers realizing the loss -- converting a would-be loss exit into a credit plus
continued exposure.

Four configurations on identical data:

    baseline    as shipped                    (up=+0.5, out=+0.3)
    no_rolls    credits switched off          (up=0,    out=0)
    up_only     the defensible credit only    (up=+0.5, out=0)
    sign_fixed  roll-out charged as a debit   (up=+0.5, out=-0.3)

Usage:
    python tools/roll_credit_sensitivity.py
    python tools/roll_credit_sensitivity.py --tickers SPY QQQ --json out.json

Caveat stated up front: run_backtest_v2 is not reachable from walk_forward, which uses the
v1 backtester. So these are single full-history in-sample backtests -- exactly the basis the
API uses, and therefore the right basis for "how much of the dashboard number is this", but
NOT an out-of-sample result. Nothing here is evidence the strategy works.
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

# Every config pins roll_model="flat". The flat credit is no longer the default --
# run_backtest_v2 now prices the roll legs -- but this tool exists to reproduce the
# historical numbers, so it must ask for the historical model explicitly. Without the pin
# these four configs would silently return identical results.
CONFIGS = {
    "baseline":   dict(roll_model="flat", roll_up_credit_pct=0.5, roll_out_credit_pct=0.3),
    "no_rolls":   dict(roll_model="flat", roll_up_credit_pct=0.0, roll_out_credit_pct=0.0),
    "up_only":    dict(roll_model="flat", roll_up_credit_pct=0.5, roll_out_credit_pct=0.0),
    "sign_fixed": dict(roll_model="flat", roll_up_credit_pct=0.5, roll_out_credit_pct=-0.3),
}


def run_one(scored: pd.DataFrame, cfg: dict, min_confs: int,
            cost_bps: float, n_regimes: int) -> dict:
    r = run_backtest_v2(
        scored,
        min_confirmations=min_confs,
        n_regimes=n_regimes,
        cost_bps_per_side=cost_bps,
        **cfg,
    )
    m, trades = r["metrics"], r["trades"]
    return {
        "total_return_pct": float(m.get("total_return_pct", 0.0)),
        "sharpe_ratio": float(m.get("sharpe_ratio", 0.0)),
        "max_drawdown_pct": float(m.get("max_drawdown_pct", 0.0)),
        "win_rate": float(m.get("win_rate", 0.0)),
        "n_trades": len(trades),
        "total_rolls": int(m.get("total_rolls", 0)),
        "roll_credits_pct": float(m.get("total_roll_credits_pct", 0.0)),
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
    ap.add_argument("--cost-bps", type=float, default=5.0,
                    help="per-side friction; defaults to 5 so this is not a cost-free read")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    n_regimes = int(args.regimes)
    rows = []

    for tkr in args.tickers:
        raw = fetch_data(tkr, period_days=args.period_days, interval=args.interval)
        if raw is None or raw.empty:
            print(f"  {tkr}: no data, skipped", file=sys.stderr)
            continue
        scored = RegimeDetector(n_regimes=n_regimes, n_iter=60).train(
            engineer_features(raw))
        for name, cfg in CONFIGS.items():
            res = run_one(scored, cfg, args.min_confs, args.cost_bps, n_regimes)
            res.update(ticker=tkr, config=name)
            rows.append(res)

    if not rows:
        print("No results.", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)

    print()
    print("=" * 84)
    print("ROLL CREDIT CONTRIBUTION IN strategy_v2   "
          f"(cost {args.cost_bps:g} bps/side, {n_regimes} regimes)")
    print("In-sample full-history backtests -- the same basis the API uses.")
    print("=" * 84)

    views = {name: df[df.config == name].set_index("ticker") for name in CONFIGS}
    base = views["baseline"]

    print("\nPer-ticker total return by configuration (%):")
    print(f"{'ticker':<8}{'baseline':>10}{'no_rolls':>10}{'up_only':>10}"
          f"{'sign_fix':>10}{'credit':>9}{'rolls':>7}{'trades':>8}{'B&H':>9}")
    print("-" * 81)
    for t in base.index:
        print(f"{t:<8}{base.loc[t,'total_return_pct']:>10.2f}"
              f"{views['no_rolls'].loc[t,'total_return_pct']:>10.2f}"
              f"{views['up_only'].loc[t,'total_return_pct']:>10.2f}"
              f"{views['sign_fixed'].loc[t,'total_return_pct']:>10.2f}"
              f"{base.loc[t,'roll_credits_pct']:>9.2f}"
              f"{base.loc[t,'total_rolls']:>7.0f}"
              f"{base.loc[t,'n_trades']:>8.0f}"
              f"{base.loc[t,'buyhold_return_pct']:>9.2f}")

    summary = {
        name: {
            "mean_return_pct": float(sub["total_return_pct"].mean()),
            "mean_sharpe": float(sub["sharpe_ratio"].mean()),
            "mean_max_dd_pct": float(sub["max_drawdown_pct"].mean()),
            "mean_win_rate": float(sub["win_rate"].mean()),
            "n_positive": int((sub["total_return_pct"] > 0).sum()),
            "n_beat_buyhold": int((sub["total_return_pct"]
                                   > sub["buyhold_return_pct"]).sum()),
            "n_tickers": int(len(sub)),
        }
        for name, sub in views.items()
    }

    print("\nAcross tickers:")
    print(f"{'config':<12}{'mean ret':>10}{'mean Sharpe':>13}{'mean maxDD':>12}"
          f"{'#positive':>11}{'#beat B&H':>11}")
    print("-" * 69)
    for name, sm in summary.items():
        print(f"{name:<12}{sm['mean_return_pct']:>10.2f}{sm['mean_sharpe']:>13.2f}"
              f"{sm['mean_max_dd_pct']:>12.2f}"
              f"{sm['n_positive']:>8}/{sm['n_tickers']}"
              f"{sm['n_beat_buyhold']:>8}/{sm['n_tickers']}")

    b = summary["baseline"]["mean_return_pct"]
    n = summary["no_rolls"]["mean_return_pct"]
    delta = b - n
    share = (delta / b * 100.0) if b else float("nan")

    # Per-ticker shares matter more than the mean, which a single huge name can dominate.
    per_ticker = {}
    for t in base.index:
        bt = base.loc[t, "total_return_pct"]
        nt = views["no_rolls"].loc[t, "total_return_pct"]
        per_ticker[t] = ((bt - nt) / bt * 100.0) if bt else float("nan")
    shares = sorted(v for v in per_ticker.values() if v == v)

    print("\nCredit share of reported return, per ticker (%):")
    for t, v in per_ticker.items():
        print(f"  {t:<6}{v:>6.1f}")
    if shares:
        print(f"  min {shares[0]:.0f}% | median {shares[len(shares)//2]:.0f}% "
              f"| max {shares[-1]:.0f}%")

    # If roll-out never fires, up_only/sign_fixed are identical to baseline. Say so rather
    # than letting a reader assume the sign error was measured and found harmless.
    roll_out_inert = (
        abs(summary["up_only"]["mean_return_pct"] - b) < 1e-9
        and abs(summary["sign_fixed"]["mean_return_pct"] - b) < 1e-9
    )

    print("\nInterpretation")
    print("-" * 84)
    print(f"  Removing both credits: mean return {b:+.2f}% -> {n:+.2f}% "
          f"({delta:+.2f} pp).")
    if b > 0 and delta > 0:
        print(f"  {share:.0f}% of the mean, but the mean is dominated by the largest")
        print(f"  compounded name; per ticker the median is "
              f"{shares[len(shares)//2]:.0f}%, ranging {shares[0]:.0f}-{shares[-1]:.0f}%.")
        print("  None of it is derived from option pricing.")
    if roll_out_inert:
        print()
        print("  ROLL_OUT IS UNREACHABLE at these settings: up_only and sign_fixed are")
        print("  identical to baseline to machine precision, so every credit came from")
        print("  ROLL_UP. Its wrong sign is therefore inert in practice, not merely small.")
        print("  Verify by re-running with an absurd roll_out_credit_pct and seeing no")
        print("  change. The reason is the exit ordering: the regime-flip check precedes")
        print("  the time-stop roll attempt, so positions are closed before they can")
        print("  qualify as a losing position still in a bullish regime.")
    nb = summary["baseline"]["n_beat_buyhold"]
    nn = summary["no_rolls"]["n_beat_buyhold"]
    print()
    print(f"  Separately, and larger than the credit itself: buy-and-hold is beaten on")
    print(f"  {nb}/{summary['baseline']['n_tickers']} tickers with the credits and "
          f"{nn}/{summary['no_rolls']['n_tickers']} without them.")
    print(f"  Mean {base['total_rolls'].mean():.0f} rolls/ticker, "
          f"{base['roll_credits_pct'].mean():.2f}% of credits booked.")
    print(f"  Charging roll-out as the debit it really is: "
          f"{summary['sign_fixed']['mean_return_pct']:+.2f}% mean.")
    print(f"  Positive tickers: baseline {summary['baseline']['n_positive']}"
          f"/{summary['baseline']['n_tickers']} -> no_rolls "
          f"{summary['no_rolls']['n_positive']}/{summary['no_rolls']['n_tickers']}.")
    print()
    print("  In-sample, so this says nothing about whether the strategy works. It bounds")
    print("  how much of the DISPLAYED number is an artifact of the roll model rather")
    print("  than of any signal.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "config": {
                "tickers": args.tickers, "period_days": args.period_days,
                "interval": args.interval, "n_regimes": n_regimes,
                "min_confs": args.min_confs, "cost_bps_per_side": args.cost_bps,
                "roll_configs": CONFIGS,
            },
            "rows": rows,
            "summary": summary,
            "credit_share_of_return_pct": share,
            "credit_share_per_ticker_pct": per_ticker,
            "roll_out_unreachable": bool(roll_out_inert),
        }, indent=2))
        print(f"\nSaved: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
