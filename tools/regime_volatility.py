#!/usr/bin/env python3
"""
Do the regimes separate forward VOLATILITY -- and do they add anything over trailing vol?

Eight experiments in ``docs/validation-findings.md`` all asked whether the regimes predict
forward *returns*. They do not. This asks a different question, and it is the first one in
this repo with a strong prior reason to expect a positive answer: volatility clustering is
one of the most robust empirical regularities in finance, whereas return predictability is
fragile and mostly absent. The features the HMM is fitted on (returns, high-low range, volume
change) are largely vol proxies, so the model plausibly encodes a vol state even though it
demonstrably fails to encode a return state.

This also matters practically. The repo's one apparent virtue is capital preservation -- SPY
-4.01% against buy-and-hold's -18.78% in 2022 at single-digit exposure. If regimes separate
forward vol, that behaviour has a mechanism. If they do not, it was luck in one window.

THE TRAP THIS TOOL IS BUILT AROUND
----------------------------------
A naive version of this test passes trivially and means nothing. Realized volatility is
strongly autocorrelated, so *any* label correlated with current vol will look like it
"predicts" forward vol. The regime label is built from features that include current vol, so
a raw Kruskal-Wallis across regimes is close to guaranteed to be significant -- and would
tell us only that vol is persistent, which we already knew without an HMM.

So every test here is run twice:

    raw       forward log vol grouped by regime
    residual  forward log vol with trailing log vol REGRESSED OUT first

Only the residual result can support a claim that the HMM adds information. The regression
is fitted on the training slice of each window (never on OOS bars) so the control itself
cannot leak. A cheap benchmark is reported alongside: the Spearman rho of trailing vol
against forward vol, which is what you get for free with no model at all.

Bars to clear (all of them, on the RESIDUAL test):

    1. separation   -- Kruskal-Wallis p below a Bonferroni-corrected alpha
    2. effect size  -- epsilon-squared above 0.01 (a floor, not a triumph)
    3. direction    -- the bearish regime set has HIGHER forward vol than the bullish set
    4. consistency  -- that direction holds in a majority of tickers and windows

Usage:
    python tools/regime_volatility.py --tickers SPY --save-obs obs/vol_SPY.csv
    python tools/regime_volatility.py --from-obs 'obs/vol_*.csv' --json docs/regime_volatility.json

Method notes
------------
* Windows, labelling and leakage discipline are identical to tools/regime_ranking.py:
  expanding train from bar 0 to oos_start, OOS labels from ``filtered_regimes`` (the forward
  algorithm, so bar t uses only bars 0..t), state ordering fitted on train only.
* Forward vol at bar t is the standard deviation of returns over bars t+1..t+h, annualized.
  It uses only bars strictly after t. Trailing vol is the standard deviation over t-19..t.
* Horizons 5 and 20 bars are both reported; the Bonferroni alpha accounts for 2 horizons x
  2 tests = 4 comparisons.
* Log vol is used throughout because vol is right-skewed; the residual control is a linear
  regression in log space.
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
HORIZONS = (5, 20)
TRAIL_BARS = 20
ANNUALIZE = np.sqrt(252.0)
EPS = 1e-12


def _fwd_vol(rets: np.ndarray, h: int) -> np.ndarray:
    """Annualized stdev of returns over the NEXT h bars. Strictly forward-looking."""
    n = len(rets)
    out = np.full(n, np.nan)
    for i in range(n - h):
        w = rets[i + 1: i + 1 + h]
        if np.isfinite(w).all():
            out[i] = np.std(w, ddof=1) * ANNUALIZE
    return out


def _trail_vol(rets: np.ndarray, k: int = TRAIL_BARS) -> np.ndarray:
    """Annualized stdev of the trailing k bars, inclusive of bar i. The free benchmark."""
    s = pd.Series(rets)
    return (s.rolling(k).std(ddof=1) * ANNUALIZE).values


def _ewma_vol(rets: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics-style EWMA vol. A stronger free benchmark than a flat window."""
    s = pd.Series(rets).fillna(0.0)
    return (np.sqrt((s ** 2).ewm(alpha=1 - lam, adjust=False).mean()) * ANNUALIZE).values


def collect(tickers, period_days, interval, n_regimes, train_bars, oos_bars, step_bars,
            n_iter):
    """Fit each window once; emit tidy causally-labelled OOS observations."""
    obs = []
    for tkr in tickers:
        raw = fetch_data(tkr, period_days=period_days, interval=interval)
        if raw is None or raw.empty:
            print(f"  {tkr}: no data, skipped", file=sys.stderr)
            continue
        feats = engineer_features(raw)
        rets = np.asarray(feats["returns"].values, dtype=float)
        trail = _trail_vol(rets)
        trail5 = _trail_vol(rets, 5)
        ewma = _ewma_vol(rets)
        fwd = {h: _fwd_vol(rets, h) for h in HORIZONS}

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
                det.train(train_slice)
            except Exception as exc:  # hmmlearn raises on degenerate covariances
                print(f"    {tkr} w{w}: fit failed ({type(exc).__name__}), skipped",
                      file=sys.stderr)
                continue

            # Control fitted on TRAIN ONLY: log fwd vol ~ a + b * log trailing vol.
            # Fitting it on OOS bars would leak the very thing we are controlling for.
            ctrl = {}
            for h in HORIZONS:
                X_tr = np.column_stack([np.log(np.clip(trail[:oos_start], EPS, None)),
                                        np.log(np.clip(trail5[:oos_start], EPS, None)),
                                        np.log(np.clip(ewma[:oos_start], EPS, None))])
                fv = fwd[h][:oos_start]
                m = np.isfinite(fv) & (fv > EPS) & np.isfinite(X_tr).all(axis=1)
                if m.sum() >= 50:
                    A = np.column_stack([np.ones(m.sum()), X_tr[m]])
                    coef, *_ = np.linalg.lstsq(A, np.log(fv[m]), rcond=None)
                    ctrl[h] = coef
                else:
                    ctrl[h] = np.zeros(4)

            labeled = det.filtered_regimes(feats.iloc[:oos_end])
            ids = np.asarray(labeled["regime_id"].values[oos_start:oos_end])
            sets = regime_sets(int(det.n_regimes))
            bull, bear = set(sets["bullish"]), set(sets["bearish"])

            for j, rid in enumerate(ids):
                i = oos_start + j
                tv, tv5, ev = trail[i], trail5[i], ewma[i]
                if not all(np.isfinite([tv, tv5, ev])) or min(tv, tv5, ev) <= EPS \
                        or not np.isfinite(float(rid)):
                    continue
                x = np.array([1.0, np.log(tv), np.log(tv5), np.log(ev)])
                row = {"ticker": tkr, "window": w, "regime_id": int(rid),
                       "trail_vol": float(tv), "trail_vol_5": float(tv5),
                       "ewma_vol": float(ev), "bar": int(i),
                       "is_bull": int(int(rid) in bull), "is_bear": int(int(rid) in bear)}
                keep = False
                for h in HORIZONS:
                    fv = fwd[h][i]
                    if np.isfinite(fv) and fv > EPS:
                        row[f"fwd_vol_{h}"] = float(fv)
                        # Residual = actual log vol minus what the trailing-vol family
                        # already implied. Coefficients came from TRAIN bars only.
                        row[f"resid_{h}"] = float(np.log(fv) - float(x @ ctrl[h]))
                        keep = True
                    else:
                        row[f"fwd_vol_{h}"] = np.nan
                        row[f"resid_{h}"] = np.nan
                if keep:
                    obs.append(row)
    return pd.DataFrame(obs)


def _epsilon_squared(groups: list) -> float:
    """Effect size for Kruskal-Wallis. 0 = no separation, 1 = total separation."""
    n = sum(len(g) for g in groups)
    k = len(groups)
    if n <= k or k < 2:
        return 0.0
    h = stats.kruskal(*groups)[0]
    return float(max(0.0, (h - k + 1) / (n - k)))


def _decimate(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Keep every stride-th bar so forward windows no longer overlap.

    This matters more than any other control here. Forward vol at bar t and bar t+1 share
    h-1 of their h returns, so consecutive observations are almost the same number. Feeding
    6300 of them to Kruskal-Wallis produces p-values that are arithmetically correct and
    scientifically meaningless, because the test believes it has 6300 independent draws when
    it has roughly 6300/h. Decimating costs power and buys honesty.
    """
    return df[df["bar"] % stride == 0]


def _bootstrap_gap(df: pd.DataFrame, col: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Block bootstrap over (ticker, window) blocks for the bear-minus-bull gap.

    Resampling whole windows respects serial dependence within a window, which a naive
    per-observation bootstrap would destroy -- and it also lets window-to-window variation
    into the interval, which is where this repo's earlier false positives hid.
    """
    d = df[np.isfinite(df[col])]
    blocks = [g for _, g in d.groupby(["ticker", "window"])]
    if len(blocks) < 5:
        return {"usable": False}
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        sub = pd.concat([blocks[i] for i in pick], ignore_index=True)
        bear, bull = sub[sub.is_bear == 1][col], sub[sub.is_bull == 1][col]
        if len(bear) > 1 and len(bull) > 1:
            gaps.append(bear.mean() - bull.mean())
    if len(gaps) < 100:
        return {"usable": False}
    g = np.array(gaps)
    return {"usable": True, "n_blocks": len(blocks),
            "gap_mean": round(float(g.mean()), 5),
            "ci_low": round(float(np.percentile(g, 2.5)), 5),
            "ci_high": round(float(np.percentile(g, 97.5)), 5),
            "frac_positive": round(float((g > 0).mean()), 4)}


def _quintile_check(df: pd.DataFrame, col: str) -> dict:
    """Distribution-free control: compare bear vs bull WITHIN trailing-vol quintiles.

    Makes no functional-form assumption at all, unlike the log-linear residual. If the
    regime label only restates trailing vol, the gap should vanish once trailing vol is
    held roughly constant.
    """
    rows, pos, tot = {}, 0, 0
    for tkr, g in df.groupby("ticker"):
        try:
            q = pd.qcut(g["trail_vol"], 5, labels=False, duplicates="drop")
        except ValueError:
            continue
        for qi in sorted(pd.unique(q.dropna())):
            sub = g[q == qi]
            bear, bull = sub[sub.is_bear == 1][col], sub[sub.is_bull == 1][col]
            if len(bear) > 1 and len(bull) > 1:
                tot += 1
                d = float(bear.mean() - bull.mean())
                pos += int(d > 0)
                rows.setdefault(tkr, []).append(round(d, 4))
    return {"quintiles_positive": f"{pos}/{tot}",
            "frac": round(pos / tot, 3) if tot else None, "per_ticker": rows}


def _test_one(df: pd.DataFrame, col: str, alpha: float) -> dict:
    """Kruskal-Wallis across regimes plus the bearish-vs-bullish direction check."""
    d = df[np.isfinite(df[col])]
    by_id = [g[col].values for _, g in d.groupby("regime_id") if len(g) > 1]
    if len(by_id) < 2:
        return {"n": int(len(d)), "usable": False}

    kw_p = float(stats.kruskal(*by_id)[1])
    eps2 = _epsilon_squared(by_id)

    bull = d[d.is_bull == 1][col].values
    bear = d[d.is_bear == 1][col].values
    gap = float(np.mean(bear) - np.mean(bull)) if len(bull) and len(bear) else float("nan")
    mw_p = (float(stats.mannwhitneyu(bear, bull, alternative="greater")[1])
            if len(bull) > 1 and len(bear) > 1 else float("nan"))

    # Consistency: the bear > bull direction must not live in one ticker or one window.
    tick_ok, tick_tot = 0, 0
    per_ticker = {}
    for tkr, g in d.groupby("ticker"):
        b1, b0 = g[g.is_bear == 1][col], g[g.is_bull == 1][col]
        if len(b1) > 1 and len(b0) > 1:
            tick_tot += 1
            diff = float(b1.mean() - b0.mean())
            per_ticker[tkr] = round(diff, 4)
            tick_ok += int(diff > 0)
    win_ok, win_tot = 0, 0
    for _, g in d.groupby(["ticker", "window"]):
        b1, b0 = g[g.is_bear == 1][col], g[g.is_bull == 1][col]
        if len(b1) > 1 and len(b0) > 1:
            win_tot += 1
            win_ok += int(float(b1.mean() - b0.mean()) > 0)

    bars = {
        "separation": kw_p < alpha,
        "effect_size": eps2 > 0.01,
        "direction": bool(gap > 0),
        "consistency": (tick_tot > 0 and tick_ok > tick_tot / 2
                        and win_tot > 0 and win_ok > win_tot / 2),
    }
    return {
        "n": int(len(d)), "usable": True, "kw_p": kw_p, "epsilon_squared": round(eps2, 5),
        "bear_minus_bull": round(gap, 5), "mw_p": mw_p,
        "tickers_with_direction": f"{tick_ok}/{tick_tot}",
        "windows_with_direction": f"{win_ok}/{win_tot}",
        "per_ticker_gap": per_ticker,
        "bars": bars, "verdict": "yes" if all(bars.values()) else "no",
    }


def evaluate(df: pd.DataFrame, alpha: float) -> dict:
    out = {"alpha": alpha, "n_obs": int(len(df)), "horizons": {}}
    for h in HORIZONS:
        # The free benchmark: how much does trailing vol alone explain?
        d = df[np.isfinite(df[f"fwd_vol_{h}"])]
        rho = (float(stats.spearmanr(d["trail_vol"], d[f"fwd_vol_{h}"])[0])
               if len(d) > 10 else float("nan"))
        nov = _decimate(df, h)
        blk = {
            "trailing_vol_benchmark_rho": round(rho, 4),
            "raw": _test_one(df, f"fwd_vol_{h}", alpha),
            "residual": _test_one(df, f"resid_{h}", alpha),
            "residual_nonoverlapping": _test_one(nov, f"resid_{h}", alpha),
            "n_nonoverlapping": int(len(nov)),
            "bootstrap_residual": _bootstrap_gap(nov, f"resid_{h}"),
            "quintile_control_raw": _quintile_check(df, f"fwd_vol_{h}"),
        }
        bs = blk["bootstrap_residual"]
        nv = blk["residual_nonoverlapping"]
        blk["strict_verdict"] = "yes" if (
            nv.get("verdict") == "yes"
            and bs.get("usable") and bs.get("ci_low", -1) > 0
            and (blk["quintile_control_raw"].get("frac") or 0) > 0.5
        ) else "no"
        out["horizons"][h] = blk
    return out


def report(res: dict) -> None:
    a = res["alpha"]
    print("=" * 92)
    print("DO REGIMES SEPARATE FORWARD VOLATILITY?   (7 regimes, 1d, causal OOS labels)")
    print(f"Bonferroni alpha = 0.05 / 4 = {a:.4f}     observations: {res['n_obs']}")
    print("=" * 92)
    print("\nRAW tests are expected to pass and prove little: vol is autocorrelated and the")
    print("regime label is built from vol-like features. The RESIDUAL tests -- trailing vol")
    print("regressed out on train only -- are the ones that can support a claim.\n")

    for h, blk in res["horizons"].items():
        print(f"── horizon {h} bars ──────────────────────────────────────────────────────")
        print(f"  free benchmark: Spearman(trailing vol, forward vol) = "
              f"{blk['trailing_vol_benchmark_rho']:+.4f}   (no model required)")
        for kind in ("raw", "residual", "residual_nonoverlapping"):
            t = blk[kind]
            if not t.get("usable"):
                print(f"  {kind:<9} unusable (n={t['n']})")
                continue
            failed = [k for k, v in t["bars"].items() if not v]
            print(f"  {kind:<9} KW p={t['kw_p']:.2e}  eps2={t['epsilon_squared']:.5f}  "
                  f"bear-bull={t['bear_minus_bull']:+.4f}  "
                  f"tickers {t['tickers_with_direction']}  "
                  f"windows {t['windows_with_direction']}  -> {t['verdict'].upper()}")
            if failed:
                print(f"            fails: {', '.join(failed)}")
            print(f"            per-ticker bear-bull gap: "
                  + "  ".join(f"{k}:{v:+.3f}" for k, v in t["per_ticker_gap"].items()))
        bs = blk["bootstrap_residual"]
        if bs.get("usable"):
            print(f"  bootstrap  residual gap {bs['gap_mean']:+.4f}  "
                  f"95% CI [{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]  "
                  f"P(>0)={bs['frac_positive']:.3f}  ({bs['n_blocks']} window blocks, "
                  f"n={blk['n_nonoverlapping']} non-overlapping)")
        qc = blk["quintile_control_raw"]
        print(f"  quintile   bear>bull within trailing-vol quintiles: "
              f"{qc['quintiles_positive']}")
        print(f"  STRICT VERDICT (non-overlapping + bootstrap CI + quintile): "
              f"{blk['strict_verdict'].upper()}")
        print()

    verdicts = {h: res["horizons"][h]["strict_verdict"] for h in res["horizons"]}
    print("=" * 92)
    if any(v == "yes" for v in verdicts.values()):
        print("RESIDUAL SEPARATION FOUND. The regime label carries forward-volatility")
        print("information beyond what trailing volatility already gives you for free.")
        print("This does NOT imply tradeable return edge -- eight prior tests say it does not.")
        print("It does mean the defensive/exposure-reduction use has a measurable basis.")
    else:
        print("NO RESIDUAL SEPARATION. Any raw separation above is explained by volatility")
        print("clustering, which trailing vol captures without an HMM. On this evidence the")
        print("regime label adds nothing to forward-vol estimation, and the capital")
        print("preservation seen in 2022 has no demonstrated mechanism behind it.")
    print("=" * 92)


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
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--save-obs", default=None, help="write tidy observations to CSV")
    ap.add_argument("--from-obs", nargs="+", default=None,
                    help="skip fitting; pool these observation CSVs (globs allowed)")
    args = ap.parse_args()

    if args.from_obs:
        files = sorted({f for pat in args.from_obs for f in glob.glob(pat)} or set(args.from_obs))
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        print(f"Pooled {len(df)} observations from {len(files)} file(s): "
              f"{sorted(df.ticker.unique())}", file=sys.stderr)
    else:
        df = collect(args.tickers, args.period_days, args.interval, args.regimes,
                     args.train_bars, args.oos_bars, args.step_bars, args.n_iter)
        if args.save_obs and not df.empty:
            Path(args.save_obs).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.save_obs, index=False)
            print(f"Wrote {len(df)} observations to {args.save_obs}", file=sys.stderr)

    if df.empty:
        print("No observations collected.", file=sys.stderr)
        sys.exit(1)

    res = evaluate(df, 0.05 / 4)
    report(res)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(res, indent=2, default=str))
        print(f"\nSaved: {args.json_out}")


if __name__ == "__main__":
    main()
