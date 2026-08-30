"""
walk_forward.py — Walk-forward validation, benchmarks, and stress tests.

Why this exists
---------------
A single backtest over all history tells you almost nothing: the HMM saw every bar it is
being scored on. This module retrains the regime detector on a rolling in-sample window
and scores it ONLY on the untouched window that follows.

Two rules are load-bearing and must not be relaxed:

1. The detector is refit on the in-sample slice of each window and never on data that
   overlaps that window's out-of-sample slice.
2. Out-of-sample regime labels come from RegimeDetector.filtered_regimes(), which is
   causal (forward algorithm only). Using train()'s Viterbi labels or predict_proba()
   here would let future bars relabel past ones and inflate the results.

See .github/copilot-instructions.md before changing any of this.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from hmm_engine import RegimeDetector
from backtester import compute_confirmations, run_backtest

# Bars of history prepended to each scored window so EMA-50 / MACD are warm and
# dropna() does not eat the start of the window being measured.
INDICATOR_WARMUP_BARS = 60

TRADING_DAYS = 252

# Periods per year, by bar interval. The scanner defaults to HOURLY bars
# (data_loader.fetch_data interval="1h"), so annualizing Sharpe/CAGR with 252 would
# overstate Sharpe by sqrt(6.5) ~= 2.5x. Always pass the interval you actually fetched.
PERIODS_PER_YEAR = {
    "1d": 252, "1day": 252, "daily": 252,
    "1h": 252 * 6.5, "60m": 252 * 6.5, "hourly": 252 * 6.5,
    "30m": 252 * 13, "15m": 252 * 26, "5m": 252 * 78, "1m": 252 * 390,
    "1wk": 52, "weekly": 52,
}


def periods_per_year(interval: str) -> float:
    """Annualization factor for a bar interval. Defaults to hourly, the repo default."""
    return float(PERIODS_PER_YEAR.get(str(interval).lower().strip(), 252 * 6.5))


@dataclass
class WindowResult:
    """Outcome of one in-sample/out-of-sample roll."""
    window: int
    is_start: Any = None
    is_end: Any = None
    oos_start: Any = None
    oos_end: Any = None
    is_bars: int = 0
    oos_bars: int = 0
    n_regimes: int = 0
    strategy: Dict[str, Any] = field(default_factory=dict)
    benchmarks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    confidence_tiers: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "window": self.window,
            "is_start": _iso(self.is_start),
            "is_end": _iso(self.is_end),
            "oos_start": _iso(self.oos_start),
            "oos_end": _iso(self.oos_end),
            "is_bars": self.is_bars,
            "oos_bars": self.oos_bars,
            "n_regimes": self.n_regimes,
            "strategy": self.strategy,
            "benchmarks": self.benchmarks,
            "confidence_tiers": self.confidence_tiers,
        }
        if self.error:
            d["error"] = self.error
        return d


def _iso(v):
    if v is None:
        return None
    try:
        return pd.Timestamp(v).isoformat()
    except Exception:
        return str(v)


def _exposure_fraction(trades: List[dict], n_bars: int) -> Optional[float]:
    """Fraction of window bars spent holding a position, from the trade list."""
    if not trades or n_bars <= 0:
        return None
    held = 0
    for t in trades:
        entry = t.get("entry_bar")
        exit_ = t.get("exit_bar")
        if entry is None or exit_ is None:
            continue
        held += max(0, int(exit_) - int(entry))
    return min(1.0, held / n_bars) if held else None


def _equity_metrics(
    equity_curve: np.ndarray,
    initial_capital: float,
    ppy: float = TRADING_DAYS,
) -> dict:
    """
    Total return / CAGR / Sharpe / max drawdown for an arbitrary equity curve.

    Used for benchmarks, which produce a curve but no discrete trade list.

    ``ppy`` is periods per year, used to annualize CAGR and Sharpe. Pass the value for
    the bar interval actually being tested (see periods_per_year) — using the daily 252
    on hourly bars overstates Sharpe by roughly 2.5x.
    """
    eq = np.asarray(equity_curve, dtype=float)
    if eq.size < 2:
        return {
            "total_return_pct": 0.0, "cagr_pct": 0.0, "sharpe": 0.0,
            "max_drawdown_pct": 0.0, "final_equity": float(initial_capital),
        }

    total_return = (eq[-1] / eq[0] - 1.0) * 100.0
    years = max(len(eq) / ppy, 1e-9)
    cagr = ((eq[-1] / eq[0]) ** (1.0 / years) - 1.0) * 100.0 if eq[0] > 0 else 0.0

    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]
    sharpe = 0.0
    if rets.size > 1 and rets.std() > 0:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(ppy))

    running_peak = np.maximum.accumulate(eq)
    dd = (eq - running_peak) / running_peak
    max_dd = float(dd.min() * 100.0)

    return {
        "total_return_pct": float(total_return),
        "cagr_pct": float(cagr),
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "final_equity": float(eq[-1]),
    }


# ──────────────────────────────── benchmarks ────────────────────────────────

def benchmark_buy_and_hold(df: pd.DataFrame, initial_capital: float = 100_000.0,
                           ppy: float = TRADING_DAYS) -> dict:
    """Fully invested for the whole window. The bar every strategy must clear."""
    close = df["Close"].values.astype(float)
    if close.size < 2 or close[0] <= 0:
        return _equity_metrics(np.array([initial_capital]), initial_capital, ppy)
    equity = initial_capital * (close / close[0])
    out = _equity_metrics(equity, initial_capital, ppy)
    out["exposure_pct"] = 100.0
    out["n_trades"] = 1
    return out


def benchmark_sma_trend(
    df: pd.DataFrame,
    window: int = 200,
    initial_capital: float = 100_000.0,
    ppy: float = TRADING_DAYS,
) -> dict:
    """
    Long only while close > SMA(window), flat otherwise.

    The SMA is computed on the window's own closes and shifted one bar, so the position
    held on bar t is decided by information available at bar t-1.
    """
    close = pd.Series(df["Close"].values.astype(float))
    sma = close.rolling(window, min_periods=window).mean()
    signal = (close > sma).shift(1).fillna(False).astype(float).values

    rets = close.pct_change().fillna(0.0).values
    strat_rets = signal * rets

    equity = initial_capital * np.cumprod(1.0 + strat_rets)
    equity = np.concatenate([[initial_capital], equity])[: len(strat_rets) + 1]

    out = _equity_metrics(equity, initial_capital, ppy)
    out["exposure_pct"] = float(signal.mean() * 100.0)
    # Count entries (flat → long transitions)
    out["n_trades"] = int(((signal == 1) & (np.roll(signal, 1) == 0)).sum())
    return out


def benchmark_random_entry(
    df: pd.DataFrame,
    n_trials: int = 50,
    hold_bars: int = 10,
    exposure_target: Optional[float] = None,
    initial_capital: float = 100_000.0,
    seed: int = 42,
    ppy: float = TRADING_DAYS,
) -> dict:
    """
    Random entries held a fixed number of bars, averaged over n_trials.

    This is the honest control: if the regime model cannot beat coin-flip entries at
    matched exposure, the regime signal is not adding anything. When ``exposure_target``
    (fraction of bars in market, 0-1) is given, the number of random entries is chosen to
    match the strategy's own exposure so the comparison is like-for-like.
    """
    close = pd.Series(df["Close"].values.astype(float))
    rets = close.pct_change().fillna(0.0).values
    n = len(rets)
    if n < hold_bars + 2:
        return _equity_metrics(np.array([initial_capital]), initial_capital, ppy)

    rng = np.random.default_rng(seed)

    if exposure_target is not None and exposure_target > 0:
        n_entries = max(1, int(round(exposure_target * n / hold_bars)))
    else:
        n_entries = max(1, n // (hold_bars * 3))

    trial_metrics = []
    for _ in range(n_trials):
        signal = np.zeros(n)
        starts = rng.integers(0, n - hold_bars, size=n_entries)
        for s in starts:
            signal[s: s + hold_bars] = 1.0

        strat_rets = signal * rets
        equity = initial_capital * np.cumprod(1.0 + strat_rets)
        equity = np.concatenate([[initial_capital], equity])
        m = _equity_metrics(equity, initial_capital, ppy)
        m["exposure_pct"] = float(signal.mean() * 100.0)
        trial_metrics.append(m)

    keys = ["total_return_pct", "cagr_pct", "sharpe", "max_drawdown_pct",
            "final_equity", "exposure_pct"]
    avg = {k: float(np.mean([m[k] for m in trial_metrics])) for k in keys}
    avg["total_return_pct_std"] = float(np.std([m["total_return_pct"] for m in trial_metrics]))
    # Share of random trials the strategy must beat to be interesting.
    avg["p95_total_return_pct"] = float(
        np.percentile([m["total_return_pct"] for m in trial_metrics], 95)
    )
    avg["n_trials"] = n_trials
    avg["n_entries_per_trial"] = n_entries
    return avg


# ──────────────────────────────── stress tests ────────────────────────────────

def inject_shock(
    df: pd.DataFrame,
    shock_pct: float = -12.0,
    bar_index: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Insert a single-day crash of ``shock_pct`` and carry it through later prices.

    All subsequent OHLC values are scaled by the same factor, so the shock is a genuine
    price-level break rather than a one-bar outlier that mean-reverts for free. The
    shocked bar's low is pushed down to the new close and volume is tripled.
    """
    out = df.copy()
    n = len(out)
    if n < 3:
        return out

    if bar_index is None:
        rng = np.random.default_rng(seed)
        bar_index = int(rng.integers(1, n - 1))
    bar_index = int(np.clip(bar_index, 1, n - 1))

    factor = 1.0 + shock_pct / 100.0

    for col in ("Open", "High", "Low", "Close"):
        if col in out.columns:
            vals = out[col].values.astype(float)
            vals[bar_index:] = vals[bar_index:] * factor
            out[col] = vals

    # The shock bar itself gapped down intraday: open stays pre-shock, low = close.
    if "Open" in out.columns:
        opens = out["Open"].values.astype(float)
        opens[bar_index] = opens[bar_index] / factor
        out["Open"] = opens
    if "Low" in out.columns:
        lows = out["Low"].values.astype(float)
        lows[bar_index] = min(lows[bar_index], float(out["Close"].values[bar_index]))
        out["Low"] = lows
    if "High" in out.columns:
        highs = out["High"].values.astype(float)
        highs[bar_index] = max(highs[bar_index], float(out["Open"].values[bar_index]))
        out["High"] = highs
    if "Volume" in out.columns:
        vols = out["Volume"].values.astype(float)
        vols[bar_index] = vols[bar_index] * 3.0
        out["Volume"] = vols

    out.attrs["shock_bar_index"] = bar_index
    out.attrs["shock_pct"] = shock_pct
    return out


class WalkForwardEngine:
    """
    Rolling in-sample / out-of-sample validation for the regime strategy.

    Parameters
    ----------
    is_bars : int
        In-sample (training) window length. Default 252 (~1 trading year).
    oos_bars : int
        Out-of-sample (scoring) window length. Default 126 (~6 months).
    step_bars : int
        How far each roll advances. Defaults to oos_bars, giving non-overlapping
        out-of-sample periods so results can be concatenated without double counting.
    n_regimes : int or "auto"
        Passed to RegimeDetector. "auto" reselects the regime count per window.
    """

    def __init__(
        self,
        is_bars: int = 252,
        oos_bars: int = 126,
        step_bars: Optional[int] = None,
        n_regimes=7,
        initial_capital: float = 100_000.0,
        backtest_kwargs: Optional[dict] = None,
        hmm_iter: int = 100,
        random_state: int = 42,
        interval: str = "1h",
    ):
        if is_bars < 30:
            raise ValueError("is_bars must be at least 30")
        if oos_bars < 5:
            raise ValueError("oos_bars must be at least 5")

        self.is_bars = int(is_bars)
        self.oos_bars = int(oos_bars)
        self.step_bars = int(step_bars) if step_bars else int(oos_bars)
        self.n_regimes = n_regimes
        self.initial_capital = float(initial_capital)
        self.backtest_kwargs = dict(backtest_kwargs or {})
        self.hmm_iter = hmm_iter
        self.random_state = random_state
        self.interval = interval
        # Annualization factor must match the bar interval actually being tested.
        self.ppy = periods_per_year(interval)

        self.windows: List[WindowResult] = []

    # ── internals ──

    def _label_oos_causally(self, detector: RegimeDetector, df: pd.DataFrame,
                            oos_start: int, oos_end: int) -> pd.DataFrame:
        """
        Causal regime labels + warm indicators for df[oos_start:oos_end].

        Regimes are computed from bar 0 through oos_end using the forward algorithm, so
        each bar's label uses only its own past. Confirmations are computed over a
        warmup buffer and then trimmed, so the scored window keeps all its bars.
        """
        ctx_start = max(0, oos_start - INDICATOR_WARMUP_BARS)

        # Causal regimes over everything up to the end of the OOS window.
        labeled = detector.filtered_regimes(df.iloc[:oos_end])

        # Indicators over warmup + OOS, then drop the warmup rows.
        ctx = labeled.iloc[ctx_start:oos_end]
        ctx = compute_confirmations(ctx)
        scored = ctx.iloc[oos_start - ctx_start:]
        return scored

    def _tier_breakdown(self, trades: List[dict], scored: pd.DataFrame) -> Dict[str, int]:
        """Count out-of-sample trades by position-sizer confidence tier."""
        from position_sizer import compute_position_size

        tiers: Dict[str, int] = {}
        if "atr" in scored.columns:
            atrs = scored["atr"].values
        else:
            atrs = None
        confs = scored["regime_confidence"].values if "regime_confidence" in scored else None

        for t in trades:
            bar = t.get("entry_bar")
            if bar is None or confs is None or bar >= len(confs):
                continue
            atr = float(atrs[bar]) if atrs is not None and np.isfinite(atrs[bar]) else \
                float(t.get("entry_price", 0)) * 0.02
            try:
                sizing = compute_position_size(
                    account_equity=self.initial_capital,
                    entry_price=float(t.get("entry_price", 0)) or 1.0,
                    atr=atr or 1.0,
                    regime_confidence=float(confs[bar]),
                    confirmations_met=int(t.get("confirmations_at_entry", 0) or 0),
                )
                tier = sizing.get("confidence_tier", "UNKNOWN")
            except Exception:
                tier = "UNKNOWN"
            tiers[tier] = tiers.get(tier, 0) + 1
        return tiers

    # ── main entry point ──

    def run(self, df: pd.DataFrame, verbose: bool = True) -> dict:
        """
        Run the rolling validation.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with engineered features (returns/range/volume_change), oldest
            bar first. Must be long enough for at least one is_bars + oos_bars window.

        Returns
        -------
        dict with keys ``windows``, ``aggregate``, ``config``.
        """
        n = len(df)
        min_needed = self.is_bars + self.oos_bars
        if n < min_needed:
            raise ValueError(
                f"Need at least {min_needed} bars for is_bars={self.is_bars} + "
                f"oos_bars={self.oos_bars}; got {n}."
            )

        self.windows = []
        w = 0
        start = 0
        while start + self.is_bars + self.oos_bars <= n:
            is_start, is_end = start, start + self.is_bars
            oos_start, oos_end = is_end, min(is_end + self.oos_bars, n)
            w += 1

            result = WindowResult(
                window=w,
                is_start=df.index[is_start], is_end=df.index[is_end - 1],
                oos_start=df.index[oos_start], oos_end=df.index[oos_end - 1],
                is_bars=is_end - is_start, oos_bars=oos_end - oos_start,
            )

            try:
                # 1. Fit ONLY on the in-sample slice. Never touches the OOS window.
                detector = RegimeDetector(
                    n_regimes=self.n_regimes,
                    n_iter=self.hmm_iter,
                    random_state=self.random_state,
                )
                detector.train(df.iloc[is_start:is_end])
                result.n_regimes = int(detector.n_regimes)

                # 2. Label the OOS window causally.
                scored = self._label_oos_causally(detector, df, oos_start, oos_end)
                if len(scored) < 10:
                    raise ValueError(f"only {len(scored)} usable OOS bars after warmup")

                # 3. Score the strategy on untouched data.
                bt = run_backtest(
                    scored,
                    initial_capital=self.initial_capital,
                    skip_confirmations=True,
                    **self.backtest_kwargs,
                )
                result.strategy = dict(bt["metrics"])
                result.trades = bt["trades"]
                result.confidence_tiers = self._tier_breakdown(bt["trades"], bt["df"])

                # The backtester does not report exposure, so derive it from the trade
                # list; the random-entry benchmark needs it to match exposure.
                exposure_frac = _exposure_fraction(bt["trades"], len(bt["df"]))
                result.strategy["exposure_pct"] = round((exposure_frac or 0) * 100.0, 2)

                # 4. Benchmarks over the identical OOS bars.
                result.benchmarks = {
                    "buy_and_hold": benchmark_buy_and_hold(
                        scored, self.initial_capital, ppy=self.ppy),
                    "sma_200_trend": benchmark_sma_trend(
                        scored, window=min(200, max(20, len(scored) // 2)),
                        initial_capital=self.initial_capital, ppy=self.ppy,
                    ),
                    "random_entry": benchmark_random_entry(
                        scored, exposure_target=exposure_frac,
                        initial_capital=self.initial_capital,
                        seed=self.random_state, ppy=self.ppy,
                    ),
                }

                if verbose:
                    print(
                        f"[WF] window {w}: OOS {result.oos_start.date()}→"
                        f"{result.oos_end.date()} | strat "
                        f"{result.strategy.get('total_return_pct', 0):.2f}% vs B&H "
                        f"{result.benchmarks['buy_and_hold']['total_return_pct']:.2f}%"
                    )

            except Exception as exc:
                result.error = str(exc)
                if verbose:
                    print(f"[WF] window {w} failed: {exc}")

            self.windows.append(result)
            start += self.step_bars

        return {
            "windows": [r.to_dict() for r in self.windows],
            "aggregate": self.aggregate(),
            "config": {
                "is_bars": self.is_bars,
                "oos_bars": self.oos_bars,
                "step_bars": self.step_bars,
                "n_regimes": self.n_regimes,
                "initial_capital": self.initial_capital,
                "backtest_kwargs": self.backtest_kwargs,
                "interval": self.interval,
                "periods_per_year": self.ppy,
            },
        }

    def aggregate(self) -> dict:
        """Summarize out-of-sample performance across all successful windows."""
        ok = [w for w in self.windows if not w.error and w.strategy]
        if not ok:
            return {"n_windows": 0, "n_failed": len(self.windows)}

        def col(key):
            return np.array([float(w.strategy.get(key, 0) or 0) for w in ok])

        strat_ret = col("total_return_pct")
        bh_ret = np.array([
            float(w.benchmarks.get("buy_and_hold", {}).get("total_return_pct", 0) or 0)
            for w in ok
        ])
        rand_ret = np.array([
            float(w.benchmarks.get("random_entry", {}).get("total_return_pct", 0) or 0)
            for w in ok
        ])
        sma_ret = np.array([
            float(w.benchmarks.get("sma_200_trend", {}).get("total_return_pct", 0) or 0)
            for w in ok
        ])

        tiers: Dict[str, int] = {}
        for w in ok:
            for k, v in w.confidence_tiers.items():
                tiers[k] = tiers.get(k, 0) + v

        return {
            "n_windows": len(ok),
            "n_failed": len(self.windows) - len(ok),
            "oos_total_trades": int(sum(len(w.trades) for w in ok)),
            "mean_oos_return_pct": float(strat_ret.mean()),
            "median_oos_return_pct": float(np.median(strat_ret)),
            "std_oos_return_pct": float(strat_ret.std()),
            "worst_oos_return_pct": float(strat_ret.min()),
            "best_oos_return_pct": float(strat_ret.max()),
            "pct_windows_profitable": float((strat_ret > 0).mean() * 100.0),
            "mean_oos_sharpe": float(col("sharpe_ratio").mean()),
            "worst_oos_max_drawdown_pct": float(col("max_drawdown_pct").min()),
            "mean_win_rate": float(col("win_rate").mean()),
            "mean_exposure_pct": float(col("exposure_pct").mean()),
            # Head-to-head: the only numbers that justify running this over an index fund.
            "mean_excess_vs_buy_and_hold_pct": float((strat_ret - bh_ret).mean()),
            "pct_windows_beating_buy_and_hold": float((strat_ret > bh_ret).mean() * 100.0),
            "mean_excess_vs_sma_trend_pct": float((strat_ret - sma_ret).mean()),
            "pct_windows_beating_sma_trend": float((strat_ret > sma_ret).mean() * 100.0),
            "mean_excess_vs_random_pct": float((strat_ret - rand_ret).mean()),
            "pct_windows_beating_random": float((strat_ret > rand_ret).mean() * 100.0),
            "confidence_tiers": tiers,
        }

    # ── stress testing ──

    def stress_test(
        self,
        df: pd.DataFrame,
        shocks: tuple = (-10.0, -12.5, -15.0),
        n_positions: int = 3,
        verbose: bool = True,
    ) -> dict:
        """
        Re-run the walk-forward with single-day crashes injected, and report the damage.

        Each shock magnitude is tested at ``n_positions`` evenly spaced points in the
        series. Reports the change in mean out-of-sample return and worst drawdown
        against the unshocked baseline.
        """
        from data_loader import engineer_features

        baseline = self.run(df, verbose=False)
        base_agg = baseline["aggregate"]

        results = []
        n = len(df)
        for shock in shocks:
            for i in range(n_positions):
                # Place shocks inside the region that actually gets scored.
                pos = int(self.is_bars + (i + 1) * (n - self.is_bars) / (n_positions + 1))
                pos = int(np.clip(pos, 1, n - 2))

                shocked = inject_shock(df, shock_pct=shock, bar_index=pos)
                # Features are price-derived, so they must be rebuilt after the shock.
                try:
                    shocked = engineer_features(shocked)
                except Exception:
                    pass

                try:
                    run = self.run(shocked, verbose=False)
                    agg = run["aggregate"]
                    entry = {
                        "shock_pct": shock,
                        "shock_bar_index": pos,
                        "shock_date": _iso(df.index[pos]),
                        "mean_oos_return_pct": agg.get("mean_oos_return_pct"),
                        "worst_oos_max_drawdown_pct": agg.get("worst_oos_max_drawdown_pct"),
                        "pct_windows_profitable": agg.get("pct_windows_profitable"),
                        "n_windows": agg.get("n_windows"),
                    }
                    if base_agg.get("n_windows"):
                        entry["return_delta_vs_baseline_pct"] = (
                            (entry["mean_oos_return_pct"] or 0)
                            - (base_agg.get("mean_oos_return_pct") or 0)
                        )
                        entry["drawdown_delta_vs_baseline_pct"] = (
                            (entry["worst_oos_max_drawdown_pct"] or 0)
                            - (base_agg.get("worst_oos_max_drawdown_pct") or 0)
                        )
                except Exception as exc:
                    entry = {"shock_pct": shock, "shock_bar_index": pos, "error": str(exc)}

                if verbose:
                    print(f"[Stress] shock {shock}% at bar {pos}: "
                          f"{entry.get('mean_oos_return_pct', 'failed')}")
                results.append(entry)

        valid = [r for r in results if "mean_oos_return_pct" in r]
        summary = {}
        if valid:
            deltas = [r.get("return_delta_vs_baseline_pct", 0) or 0 for r in valid]
            dd = [r.get("worst_oos_max_drawdown_pct", 0) or 0 for r in valid]
            summary = {
                "worst_return_delta_pct": float(min(deltas)),
                "mean_return_delta_pct": float(np.mean(deltas)),
                "worst_drawdown_under_shock_pct": float(min(dd)),
                "n_scenarios": len(valid),
                "n_failed_scenarios": len(results) - len(valid),
            }

        return {"baseline": base_agg, "scenarios": results, "summary": summary}


def format_report(result: dict) -> str:
    """Render a walk-forward result dict as a plain-text report."""
    agg = result.get("aggregate", {})
    cfg = result.get("config", {})
    lines = [
        "=" * 68,
        "WALK-FORWARD VALIDATION (out-of-sample only)",
        "=" * 68,
        f"In-sample window : {cfg.get('is_bars')} bars",
        f"Out-of-sample    : {cfg.get('oos_bars')} bars, step {cfg.get('step_bars')}",
        f"Regimes          : {cfg.get('n_regimes')}",
        f"Bar interval     : {cfg.get('interval')} "
        f"({cfg.get('periods_per_year')} periods/yr)",
        "",
        f"Windows scored   : {agg.get('n_windows', 0)} (failed: {agg.get('n_failed', 0)})",
        f"OOS trades       : {agg.get('oos_total_trades', 0)}",
        "",
        f"Mean OOS return  : {agg.get('mean_oos_return_pct', 0):.2f}%",
        f"Median           : {agg.get('median_oos_return_pct', 0):.2f}%",
        f"Std dev          : {agg.get('std_oos_return_pct', 0):.2f}%",
        f"Worst window     : {agg.get('worst_oos_return_pct', 0):.2f}%",
        f"Profitable       : {agg.get('pct_windows_profitable', 0):.1f}% of windows",
        f"Mean Sharpe      : {agg.get('mean_oos_sharpe', 0):.2f}",
        f"Worst drawdown   : {agg.get('worst_oos_max_drawdown_pct', 0):.2f}%",
        "",
        "-- vs benchmarks (mean excess, % of windows won) --",
        f"Buy & hold  : {agg.get('mean_excess_vs_buy_and_hold_pct', 0):+.2f}%  "
        f"({agg.get('pct_windows_beating_buy_and_hold', 0):.1f}%)",
        f"SMA trend   : {agg.get('mean_excess_vs_sma_trend_pct', 0):+.2f}%  "
        f"({agg.get('pct_windows_beating_sma_trend', 0):.1f}%)",
        f"Random entry: {agg.get('mean_excess_vs_random_pct', 0):+.2f}%  "
        f"({agg.get('pct_windows_beating_random', 0):.1f}%)",
    ]
    lines += [
        "",
        "NOTE: strategy Sharpe comes from backtester._compute_metrics, which hardcodes",
        "sqrt(252). On non-daily bars it is not comparable to the benchmark Sharpes",
        "above, which use the interval-correct factor. Pre-existing; see PR notes.",
    ]
    tiers = agg.get("confidence_tiers") or {}
    if tiers:
        lines += ["", "-- OOS trades by confidence tier --"]
        lines += [f"{k:<8}: {v}" for k, v in sorted(tiers.items())]
    lines.append("=" * 68)
    return "\n".join(lines)


def _cli():
    """
    Run a walk-forward validation from the command line.

    Examples
    --------
        python walk_forward.py SPY
        python walk_forward.py NVDA --is-bars 252 --oos-bars 126 --regimes auto
        python walk_forward.py SPY --stress
    """
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("symbol")
    parser.add_argument("--period-days", type=int, default=1500)
    parser.add_argument("--interval", default="1d",
                        help="bar interval; '1d' for daily, '1h' for the scanner default")
    parser.add_argument("--is-bars", type=int, default=252)
    parser.add_argument("--oos-bars", type=int, default=126)
    parser.add_argument("--step-bars", type=int, default=None)
    parser.add_argument("--regimes", default="7",
                        help="regime count, or 'auto' to select per window")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--stress", action="store_true",
                        help="also run shock scenarios")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="write the full result dict to this path")
    args = parser.parse_args()

    from data_loader import fetch_data, engineer_features

    print(f"Fetching {args.symbol}...")
    df = fetch_data(args.symbol, period_days=args.period_days,
                    interval=args.interval)
    if df is None or df.empty:
        raise SystemExit(f"No data returned for {args.symbol}")
    df = engineer_features(df)
    print(f"Loaded {len(df)} bars: {df.index[0].date()} → {df.index[-1].date()}")

    n_regimes = "auto" if str(args.regimes).lower() == "auto" else int(args.regimes)

    engine = WalkForwardEngine(
        is_bars=args.is_bars,
        oos_bars=args.oos_bars,
        step_bars=args.step_bars,
        n_regimes=n_regimes,
        initial_capital=args.capital,
        interval=args.interval,
    )

    result = engine.run(df)
    print()
    print(format_report(result))

    if args.stress:
        print("\nRunning stress scenarios...")
        stress = engine.stress_test(df)
        result["stress"] = stress
        s = stress.get("summary") or {}
        if s:
            print(f"\nWorst return delta under shock : "
                  f"{s['worst_return_delta_pct']:+.2f}%")
            print(f"Worst drawdown under shock    : "
                  f"{s['worst_drawdown_under_shock_pct']:.2f}%")
            print(f"Scenarios run                 : {s['n_scenarios']}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            _json.dump(result, f, indent=2, default=str)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    _cli()
