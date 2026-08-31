#!/usr/bin/env python3
"""
Does ranking regimes by their own realized return carry any forward information?

This is the last idea in the repo with an actual mechanism behind it rather than a hunch.

The rule under test: ``RegimeDetector`` ranks raw HMM states by the mean return of the bars
DURING each state, and calls rank 0 "most bullish". Since the HMM is fitted on returns and
momentum features, that is close to tautological -- a state scores highest largely because
it is *defined* by prices having risen. Whether the bar AFTER such a state is also good is a
separate question, and three earlier experiments (tests 4, 5 and 7 in
``docs/validation-findings.md``) each found the labelled-bullish regime mildly
*under*performing, which is what you would see if rank 0 marks local tops.

So: five ranking rules, all estimated on training data only, evaluated on causally-labelled
out-of-sample windows.

    return          mean return during the state          (the shipped rule)
    inverted        same, ranked ascending                 (is the rule anti-predictive?)
    forward_return  mean return of the NEXT bar            (what a trader actually needs)
    forward_sharpe  next-bar mean / next-bar sd
    persistence     self-transition probability            (sticky, direction-agnostic)

A rule only "works" if it clears all four bars:

    1. right direction  -- rank 0's forward return beats the unconditional mean
    2. monotonic-ish    -- Spearman rho between regime_id and forward return is NEGATIVE
    3. significant      -- Kruskal-Wallis p below a Bonferroni-corrected alpha
    4. consistent       -- rank 0 beats the window mean in a majority of windows,
                           and the effect is not concentrated in one ticker

Bar 4 exists because a bare p-value already fooled this repo once: tools/cross_asset_features.py
originally printed "separates forward returns" for a config whose edge was the wrong sign and
lived in 2 of 5 tickers.

Usage:
    python tools/regime_ranking.py
    python tools/regime_ranking.py --tickers SPY QQQ --json out.json

Method notes
------------
* Training is expanding, from bar 0 to the start of each OOS window; the HMM never sees an
  OOS bar. The rank rule is scored on the training slice only.
* OOS labels come from ``filtered_regimes``, the forward algorithm, so bar t's label uses
  only bars 0..t. Appending bars cannot rewrite an earlier label.
* The rule changes only the state -> regime_id mapping, never the fit, so each window is
  fitted ONCE and re-ranked five times via ``RegimeDetector.apply_rank_rule``. Same code
  path the engine uses.
* Forward return is the next bar's simple return. No strategy, no costs, no confirmations --
  this measures information content, which is upstream of whether any strategy can harvest it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from data_loader import fetch_data, engineer_features
from hmm_engine import RegimeDetector, RANK_RULES, regime_sets

DEFAULT_TICKERS = ["SPY", "QQQ", "NVDA", "AAPL", "XLF"]


def collect(tickers, period_days, interval, n_regimes, train_bars, oos_bars, step_bars,
            n_iter):
    """Fit each window once, re-rank under every rule, return tidy OOS observations."""
    obs = []
    for tkr in tickers:
        raw = fetch_data(tkr, period_days=period_days, interval=interval)
        if raw is None or raw.empty:
            print(f"  {tkr}: no data, skipped", file=sys.stderr)
            continue
        feats = engineer_features(raw)
        rets = np.asarray(feats["returns"].values, dtype=float)
        fwd = np.full(len(rets), np.nan)
        fwd[:-1] = rets[1:]

        n = len(feats)
        starts = list(range(train_bars, n - oos_bars + 1, step_bars))
        if not starts:
            print(f"  {tkr}: only {n} bars, too short", file=sys.stderr)
            continue
        print(f"  {tkr}: {n} bars, {len(starts)} windows", file=sys.stderr)

        for w, oos_start in enumerate(starts):
            oos_end = oos_start + oos_bars
            train_slice = feats.iloc[:oos_start]
            det = RegimeDetector(n_regimes=n_regimes, n_iter=n_iter)
            try:
                trained = det.train(train_slice)
            except Exception as exc:  # hmmlearn raises on degenerate covariances
                print(f"    {tkr} w{w}: fit failed ({type(exc).__name__}), skipped",
                      file=sys.stderr)
                continue
            train_states = np.asarray(trained["raw_state"].values)

            for rule in RANK_RULES:
                det.apply_rank_rule(train_slice, train_states, rank_by=rule)
                labeled = det.filtered_regimes(feats.iloc[:oos_end])
                ids = np.asarray(labeled["regime_id"].values[oos_start:oos_end])
                f = fwd[oos_start:oos_end]
                ok = np.isfinite(f) & np.isfinite(ids.astype(float))
                for rid, fr in zip(ids[ok], f[ok]):
                    obs.append({"ticker": tkr, "window": w, "rule": rule,
                                "regime_id": int(rid), "fwd_return": float(fr)})
    return pd.DataFrame(obs)


def check_partitions(df: pd.DataFrame) -> tuple:
    """Validity check: a rank rule must relabel states, never repartition the bars.

    Within one window the multiset of group sizes must be identical under every rule, since
    all five rules reorder the same Viterbi/forward states. Note this holds PER WINDOW only
    -- pooled group sizes legitimately differ, because each rule promotes a different raw
    state to id 0 in each window and pooling mixes them. Expecting pooled invariance is a
    trap worth naming: it looks like a bug in the tool when it is not.
    """
    mismatches = []
    for (tkr, w), g in df.groupby(["ticker", "window"]):
        sizes = {r: sorted(sub.groupby("regime_id").size().tolist())
                 for r, sub in g.groupby("rule")}
        ref = sizes.get("return")
        if ref is not None and any(v != ref for v in sizes.values()):
            mismatches.append((tkr, int(w)))
    return (len(mismatches) == 0, mismatches)


def evaluate(df: pd.DataFrame, alpha: float) -> dict:
    """Score one rule against all four bars."""
    out = {}
    overall = float(df["fwd_return"].mean())
    for rule, g in df.groupby("rule"):
        by_id = g.groupby("regime_id")["fwd_return"]
        means = {int(k): float(v) for k, v in by_id.mean().items()}
        counts = {int(k): int(v) for k, v in by_id.count().items()}
        rule_mean = float(g["fwd_return"].mean())

        bull = regime_sets(max(means) + 1)["bullish"] if means else []
        rank0 = g[g.regime_id == 0]["fwd_return"]
        rest = g[g.regime_id != 0]["fwd_return"]
        bull_ret = g[g.regime_id.isin(bull)]["fwd_return"]

        rho, rho_p = (stats.spearmanr(g["regime_id"], g["fwd_return"])
                      if g["regime_id"].nunique() > 1 else (np.nan, np.nan))
        groups = [v.values for _, v in by_id if len(v) > 1]
        kw_p = stats.kruskal(*groups)[1] if len(groups) > 1 else np.nan
        mw_p = (stats.mannwhitneyu(rank0, rest, alternative="two-sided")[1]
                if len(rank0) > 1 and len(rest) > 1 else np.nan)

        # Consistency: per window and per ticker, did rank 0 beat that slice's own mean?
        wins, total = 0, 0
        for (_, _), sub in g.groupby(["ticker", "window"]):
            r0 = sub[sub.regime_id == 0]["fwd_return"]
            if len(r0) == 0:
                continue
            total += 1
            wins += int(r0.mean() > sub["fwd_return"].mean())
        per_ticker = {}
        for tkr, sub in g.groupby("ticker"):
            r0 = sub[sub.regime_id == 0]["fwd_return"]
            per_ticker[tkr] = (float(r0.mean() - sub["fwd_return"].mean()) * 1e4
                              if len(r0) else float("nan"))

        edge_bps = (float(rank0.mean()) - rule_mean) * 1e4 if len(rank0) else float("nan")
        bull_edge_bps = ((float(bull_ret.mean()) - rule_mean) * 1e4
                         if len(bull_ret) else float("nan"))
        pos_tickers = sum(1 for v in per_ticker.values() if v == v and v > 0)

        checks = {
            "direction": bool(edge_bps == edge_bps and edge_bps > 0),
            "monotonic": bool(rho == rho and rho < 0),
            "significant": bool(kw_p == kw_p and kw_p < alpha),
            "consistent": bool(total and wins / total > 0.5
                               and pos_tickers > len(per_ticker) / 2),
        }
        out[rule] = {
            "n_obs": int(len(g)), "rule_mean_bps": rule_mean * 1e4,
            "rank0_edge_bps": edge_bps, "bullish_set_edge_bps": bull_edge_bps,
            "spearman_rho": float(rho) if rho == rho else None,
            "spearman_p": float(rho_p) if rho_p == rho_p else None,
            "kruskal_p": float(kw_p) if kw_p == kw_p else None,
            "mannwhitney_p": float(mw_p) if mw_p == mw_p else None,
            "rank0_beat_window_mean": f"{wins}/{total}",
            "rank0_win_frac": (wins / total) if total else None,
            "per_ticker_edge_bps": per_ticker,
            "mean_fwd_by_regime_bps": {k: v * 1e4 for k, v in means.items()},
            "n_by_regime": counts,
            "checks": checks,
            "verdict": "WORKS" if all(checks.values()) else "no",
        }
    out["_overall_mean_bps"] = overall * 1e4
    return out


def main() -> int:
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
    # Fitting all tickers in one process can outrun a shell timeout, so observations can be
    # collected per ticker and pooled afterwards. Pooling raw observations (not per-ticker
    # summaries) keeps the statistics identical to a single run.
    ap.add_argument("--save-obs", default=None, help="write tidy observations to CSV")
    ap.add_argument("--from-obs", nargs="+", default=None,
                    help="skip fitting; pool these observation CSVs and evaluate")
    args = ap.parse_args()

    if args.from_obs:
        df = pd.concat([pd.read_csv(f) for f in args.from_obs], ignore_index=True)
        print(f"Pooled {len(df)} observations from {len(args.from_obs)} file(s): "
              f"{sorted(df.ticker.unique())}", file=sys.stderr)
    else:
        df = collect(args.tickers, args.period_days, args.interval, args.regimes,
                     args.train_bars, args.oos_bars, args.step_bars, args.n_iter)
        if args.save_obs and not df.empty:
            Path(args.save_obs).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.save_obs, index=False)
            print(f"Wrote {len(df)} observations to {args.save_obs}", file=sys.stderr)
    if df.empty:
        print("No observations.", file=sys.stderr)
        return 1

    ok, mismatches = check_partitions(df)
    if not ok:
        print(f"VALIDITY CHECK FAILED: {len(mismatches)} window(s) repartitioned by the "
              f"rank rule, e.g. {mismatches[:3]}. A rule must relabel, not repartition; "
              f"results below are not trustworthy.", file=sys.stderr)
    else:
        print(f"Validity check: rank rules relabel without repartitioning in all "
              f"{df.groupby(['ticker', 'window']).ngroups} windows.", file=sys.stderr)

    alpha = 0.05 / len(RANK_RULES)
    res = evaluate(df, alpha)
    res["_partition_check_passed"] = bool(ok)

    print()
    print("=" * 92)
    print(f"REGIME RANKING RULES vs FORWARD RETURN   ({args.regimes} regimes, "
          f"{args.interval}, train>={args.train_bars} / oos {args.oos_bars} bars)")
    print(f"Causal OOS labels. Bonferroni alpha = 0.05/{len(RANK_RULES)} = {alpha:.4f}")
    print("=" * 92)
    print(f"\nUnconditional mean forward return: {res['_overall_mean_bps']:+.2f} bps")

    print(f"\n{'rule':<16}{'rank0 edge':>12}{'bull edge':>11}{'rho':>8}{'KW p':>9}"
          f"{'MW p':>9}{'win':>8}{'verdict':>9}")
    print("-" * 84)
    for rule in RANK_RULES:
        r = res[rule]
        rho = r["spearman_rho"]
        print(f"{rule:<16}{r['rank0_edge_bps']:>+12.2f}{r['bullish_set_edge_bps']:>+11.2f}"
              f"{(rho if rho is not None else float('nan')):>8.3f}"
              f"{r['kruskal_p']:>9.4f}{r['mannwhitney_p']:>9.4f}"
              f"{r['rank0_beat_window_mean']:>8}{r['verdict']:>9}")

    print("\nPer-rule detail")
    print("-" * 92)
    for rule in RANK_RULES:
        r = res[rule]
        failed = [k for k, v in r["checks"].items() if not v]
        print(f"\n  {rule}  ({r['n_obs']} obs)  -> {r['verdict']}")
        if failed:
            print(f"    fails: {', '.join(failed)}")
        fwd = r["mean_fwd_by_regime_bps"]
        print("    forward return by regime id (bps): "
              + "  ".join(f"{k}:{fwd[k]:+.1f}" for k in sorted(fwd)))
        print("    per-ticker rank0 edge (bps): "
              + "  ".join(f"{t}:{v:+.1f}" for t, v in r["per_ticker_edge_bps"].items()))

    print("\n" + "=" * 92)
    winners = [r for r in RANK_RULES if res[r]["verdict"] == "WORKS"]
    if winners:
        print(f"PASSES ALL FOUR BARS: {', '.join(winners)}")
        print("Worth a strategy-level walk-forward run before believing it.")
    else:
        print("NO RULE PASSES. The ranking carries no usable forward information, and")
        print("neither does inverting it -- so the labelled-bullish underperformance seen")
        print("in tests 4, 5 and 7 is not a sign-flip waiting to be harvested.")
    print("=" * 92)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"results": res, "alpha": alpha,
             "params": vars(args)}, indent=2, default=str))
        print(f"\nSaved: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
