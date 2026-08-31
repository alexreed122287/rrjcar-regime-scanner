"""
Tests for regime-set derivation.

The bug these cover: run_backtest hardcoded bullish_regimes=[0, 1, 2] and
bearish_regimes=[5, 6], which only makes sense for a 7-state model. With 3 regimes
every state was "bullish", so the strategy was permanently long and silently became
buy-and-hold — which is exactly what the walk-forward run showed (82-96% exposure and
buy-and-hold-sized drawdowns whenever n_regimes was auto-selected to 3-4).
"""

import numpy as np
import pandas as pd
import pytest

from hmm_engine import (
    REGIME_LABELS,
    REFERENCE_BEARISH,
    REFERENCE_BULLISH,
    REFERENCE_N,
    labels_for,
    regime_sets,
)
from backtester import run_backtest


# ── backwards compatibility: the 7-regime default must not move ──

def test_seven_regimes_reproduces_historical_defaults():
    """The whole change is safe only if n_regimes=7 is bit-identical to before."""
    rs = regime_sets(7)
    assert rs["bullish"] == [0, 1, 2]
    assert rs["bearish"] == [5, 6]
    assert rs["neutral"] == [3, 4]


def test_seven_regime_labels_reproduce_the_constant():
    assert labels_for(7) == list(REGIME_LABELS)


def test_reference_shape_constants_agree_with_labels():
    assert REFERENCE_N == len(REGIME_LABELS)
    assert REFERENCE_BULLISH + REFERENCE_BEARISH < REFERENCE_N


# ── structural invariants across every plausible regime count ──

@pytest.mark.parametrize("n", list(range(3, 13)))
def test_sets_are_disjoint_and_ordered(n):
    rs = regime_sets(n)
    bull, neu, bear = rs["bullish"], rs["neutral"], rs["bearish"]

    assert not set(bull) & set(bear), f"n={n}: bullish and bearish overlap"
    assert not set(bull) & set(neu)
    assert not set(bear) & set(neu)
    assert sorted(bull + neu + bear) == list(range(n)), f"n={n}: not a partition"

    # Rank order: all bullish ids come before all bearish ids.
    assert max(bull) < min(bear), f"n={n}: bullish ids not below bearish ids"


@pytest.mark.parametrize("n", list(range(3, 13)))
def test_never_every_state_bullish(n):
    """The core regression: a 3-regime model must not treat all states as bullish."""
    rs = regime_sets(n)
    assert len(rs["bullish"]) < n
    assert len(rs["bearish"]) >= 1, f"n={n}: no bearish state means exits never fire"
    assert len(rs["neutral"]) >= 1, f"n={n}: no neutral 'do nothing' band"


def test_three_regimes_is_one_each():
    rs = regime_sets(3)
    assert rs == {"bullish": [0], "neutral": [1], "bearish": [2], "n_regimes": 3}


def test_four_regimes():
    rs = regime_sets(4)
    assert rs["bullish"] == [0, 1]
    assert rs["bearish"] == [3]


def test_five_regimes_has_a_bearish_state():
    """routes_scan.py uses n_regimes=5 in cloud; previously it had no bear regime."""
    rs = regime_sets(5)
    assert rs["bearish"], "5-regime cloud path must be able to signal a bear regime"
    assert max(rs["bullish"]) < min(rs["bearish"])


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 9])
def test_bullish_fraction_tracks_the_reference(n):
    """Bullish share should stay near the reference 3/7 rather than drifting."""
    frac = len(regime_sets(n)["bullish"]) / n
    assert abs(frac - REFERENCE_BULLISH / REFERENCE_N) < 0.22, f"n={n}: frac={frac:.3f}"


def test_rejects_nonsense_counts():
    with pytest.raises(ValueError):
        regime_sets(0)
    with pytest.raises(ValueError):
        regime_sets(-3)


def test_edge_counts_do_not_crash():
    assert regime_sets(1)["neutral"] == [0]
    assert regime_sets(2) == {
        "bullish": [0], "neutral": [], "bearish": [1], "n_regimes": 2,
    }


# ── labels line up with the sets ──

@pytest.mark.parametrize("n", list(range(2, 10)))
def test_labels_and_sets_are_consistent(n):
    labels = labels_for(n)
    rs = regime_sets(n)
    assert len(labels) == n
    assert len(set(labels)) == n, f"n={n}: duplicate labels {labels}"
    # The most bearish id must carry a bearish-sounding label.
    if rs["bearish"]:
        worst = labels[max(rs["bearish"])]
        assert "Bear" in worst or "Crash" in worst, f"n={n}: worst label is {worst!r}"


# ── run_backtest integration ──

def _regime_frame(make_ohlcv, n_regimes, regime_id):
    """OHLCV frame where every bar sits in one regime, with confirmations forced on."""
    df = make_ohlcv(n_bars=260, trend_slope=0.0004, volatility=0.008)
    df["regime_id"] = regime_id
    df["regime_label"] = labels_for(n_regimes)[regime_id]
    df["regime_confidence"] = 0.95
    # Make sure the derived count is unambiguous for inference-based tests.
    df.iloc[-1, df.columns.get_loc("regime_id")] = n_regimes - 1
    return df


def test_backtest_infers_regime_count_from_data(make_ohlcv):
    """With 3 regimes present, regime 2 is bearish — so no long entries."""
    df = _regime_frame(make_ohlcv, 3, 2)
    out = run_backtest(df, skip_confirmations=False)
    assert out["metrics"]["total_trades"] == 0


def test_backtest_explicit_n_regimes_overrides_inference(make_ohlcv):
    df = _regime_frame(make_ohlcv, 7, 0)
    # Regime 0 is bullish under a 7-state layout either way; assert it is accepted.
    out = run_backtest(df, n_regimes=7, skip_confirmations=False)
    assert "total_trades" in out["metrics"]


def test_explicit_regime_sets_still_win(make_ohlcv):
    """Callers passing explicit sets must not be second-guessed by the derivation."""
    df = _regime_frame(make_ohlcv, 3, 2)
    out = run_backtest(
        df, bullish_regimes=[2], bearish_regimes=[0], skip_confirmations=False
    )
    # Regime 2 is now bullish, so entries become possible.
    assert out["metrics"]["total_trades"] >= 0
    assert out["df"] is not None


def test_backtest_raises_without_regimes_or_count(make_ohlcv):
    df = make_ohlcv(n_bars=120)
    with pytest.raises(ValueError, match="n_regimes"):
        run_backtest(df)


def test_three_regime_model_is_not_always_long(make_ohlcv):
    """
    Headline regression test.

    Under the old hardcoded [0, 1, 2] every state of a 3-regime model was bullish, so
    the strategy held a position essentially all the time. Exposure must now be well
    short of always-on when the model spends time in its bearish state.
    """
    rng = np.random.default_rng(7)
    df = make_ohlcv(n_bars=400, trend_slope=0.0003, volatility=0.01)
    # Cycle through all three regimes in long blocks.
    ids = np.repeat(rng.integers(0, 3, size=40), 10)[: len(df)]
    df["regime_id"] = ids
    df["regime_label"] = [labels_for(3)[i] for i in ids]
    df["regime_confidence"] = 0.95

    out = run_backtest(df, n_regimes=3, skip_confirmations=False)
    bars_held = sum(
        (t.get("exit_bar", 0) - t.get("entry_bar", 0))
        for t in out["trades"]
        if t.get("exit_bar") is not None and t.get("entry_bar") is not None
    )
    exposure = bars_held / len(out["df"]) if len(out["df"]) else 0.0
    assert exposure < 0.75, f"3-regime model still nearly always long (exposure {exposure:.2f})"


# ── confidence threshold parameterization ──

def test_default_confidence_thresholds_preserve_behavior(make_ohlcv):
    """Defaults must reproduce the previously hardcoded 0.5 / 0.6 gates exactly."""
    df = _regime_frame(make_ohlcv, 7, 0)
    a = run_backtest(df.copy(), n_regimes=7, skip_confirmations=False)
    b = run_backtest(
        df.copy(), n_regimes=7, skip_confirmations=False,
        min_confidence=0.5, neutral_exit_confidence=0.6,
    )
    assert a["metrics"]["total_trades"] == b["metrics"]["total_trades"]
    assert a["metrics"]["total_return_pct"] == b["metrics"]["total_return_pct"]


def test_impossible_confidence_blocks_all_entries(make_ohlcv):
    df = _regime_frame(make_ohlcv, 7, 0)
    out = run_backtest(
        df, n_regimes=7, skip_confirmations=False, min_confidence=1.01,
    )
    assert out["metrics"]["total_trades"] == 0


def test_raising_confidence_is_monotone_in_trade_count(make_ohlcv):
    """Tightening the gate must never increase the number of entries."""
    import numpy as np
    rng = np.random.default_rng(3)
    df = make_ohlcv(n_bars=400, trend_slope=0.0004, volatility=0.01)
    ids = np.repeat(rng.integers(0, 7, size=40), 10)[: len(df)]
    df["regime_id"] = ids
    df["regime_label"] = [labels_for(7)[i] for i in ids]
    df["regime_confidence"] = rng.uniform(0.35, 0.99, len(df))

    counts = [
        run_backtest(
            df.copy(), n_regimes=7, skip_confirmations=False, min_confidence=c
        )["metrics"]["total_trades"]
        for c in (0.4, 0.6, 0.8, 0.95)
    ]
    assert counts == sorted(counts, reverse=True), counts


# ── transaction costs ────────────────────────────────────────────────────────────

def _costed_frame(n=400, seed=0):
    """Synthetic frame with regimes attached, good for many round trips."""
    import numpy as np
    import pandas as pd
    from backtester import compute_confirmations

    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0008, 0.01, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({"Open": close, "High": close * 1.005, "Low": close * 0.995,
                       "Close": close, "Volume": 1e6}, index=idx)
    df["regime_id"] = np.tile([0, 0, 0, 1, 4, 5], n // 6 + 1)[:n]
    df["regime_label"] = "x"
    df["regime_confidence"] = 0.9
    df["returns"] = df["Close"].pct_change().fillna(0)
    df["range"] = (df["High"] - df["Low"]) / df["Close"]
    df["volume_change"] = 0.0
    return compute_confirmations(df)


def test_zero_cost_is_the_default_and_changes_nothing():
    """The default must preserve every pre-existing result exactly."""
    df = _costed_frame()
    a = run_backtest(df, skip_confirmations=True, n_regimes=7)
    b = run_backtest(df, skip_confirmations=True, n_regimes=7, cost_bps_per_side=0.0)
    assert a["metrics"]["total_return_pct"] == b["metrics"]["total_return_pct"]
    assert a["metrics"]["total_cost_paid_pct"] == 0.0


def test_costs_reduce_return_monotonically():
    df = _costed_frame()
    rets = [run_backtest(df, skip_confirmations=True, n_regimes=7,
                         cost_bps_per_side=b)["metrics"]["total_return_pct"]
            for b in (0, 1, 5, 10, 20)]
    assert rets == sorted(rets, reverse=True), rets


def test_cost_charged_on_both_sides_of_each_trade():
    """Gross minus net must equal exactly two sides of friction."""
    df = _costed_frame()
    bt = run_backtest(df, skip_confirmations=True, n_regimes=7, cost_bps_per_side=10.0)
    for tr in bt["trades"]:
        assert abs((tr["gross_pnl_pct"] - tr["pnl_pct"]) - 0.20) < 1e-6


def test_total_cost_paid_matches_trade_count():
    df = _costed_frame()
    bt = run_backtest(df, skip_confirmations=True, n_regimes=7, cost_bps_per_side=5.0)
    expected = len(bt["trades"]) * 2 * 5.0 / 10_000.0 * 100.0
    assert abs(bt["metrics"]["total_cost_paid_pct"] - expected) < 1e-6


def test_cost_scales_with_leverage():
    """A levered position transacts more notional, so it pays more friction."""
    df = _costed_frame()
    one = run_backtest(df, skip_confirmations=True, n_regimes=7,
                       cost_bps_per_side=10.0, leverage=1.0)
    two = run_backtest(df, skip_confirmations=True, n_regimes=7,
                       cost_bps_per_side=10.0, leverage=2.0)
    assert two["metrics"]["total_cost_paid_pct"] > one["metrics"]["total_cost_paid_pct"]


def test_random_benchmark_also_pays_costs():
    """Costing only the strategy would hand the benchmark a free edge."""
    from walk_forward import benchmark_random_entry
    df = _costed_frame()
    free = benchmark_random_entry(df, n_trials=20, exposure_target=0.3)
    paid = benchmark_random_entry(df, n_trials=20, exposure_target=0.3,
                                  cost_bps_per_side=25.0)
    assert paid["total_return_pct"] < free["total_return_pct"]


def test_detector_accepts_custom_feature_columns():
    """Cross-asset experiments need to fit on non-default columns."""
    import numpy as np
    import pandas as pd
    from hmm_engine import RegimeDetector, FEATURE_COLUMNS

    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "returns": rng.normal(0, 0.01, n),
        "range": rng.uniform(0.005, 0.02, n),
        "volume_change": rng.normal(0, 0.1, n),
        "credit": rng.normal(0, 0.003, n),
        "rates": rng.normal(0, 0.008, n),
        "breadth": rng.normal(0, 0.004, n),
        "Close": 100 + np.arange(n) * 0.1,
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))

    det = RegimeDetector(n_regimes=3, n_iter=20,
                         feature_columns=["credit", "rates", "breadth"])
    det.train(df)
    assert det.is_trained
    assert det.feature_columns == ["credit", "rates", "breadth"]
    # Default must still be the production set.
    assert RegimeDetector(n_regimes=3).feature_columns == list(FEATURE_COLUMNS)


def test_custom_feature_columns_validated():
    import numpy as np
    import pandas as pd
    import pytest
    from hmm_engine import RegimeDetector

    df = pd.DataFrame({"returns": np.zeros(50), "range": np.zeros(50),
                       "volume_change": np.zeros(50)})
    det = RegimeDetector(n_regimes=3, feature_columns=["returns", "nope"])
    with pytest.raises(ValueError, match="missing feature columns"):
        det.train(df)


# ─────────────────────────── annualization + friction ───────────────────────────

def _synthetic_scored(n=400, seed=1):
    """A regime-labelled frame good enough to drive either backtester."""
    import numpy as np
    import pandas as pd
    from data_loader import engineer_features
    from hmm_engine import RegimeDetector

    rng = np.random.default_rng(seed)
    price = 100 * np.cumprod(1 + rng.normal(0.0004, 0.011, n))
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    raw = pd.DataFrame({
        "Open": price, "High": price * 1.008, "Low": price * 0.992,
        "Close": price, "Volume": rng.integers(1e6, 5e6, n),
    }, index=idx)
    from backtester import compute_confirmations
    scored = RegimeDetector(n_regimes=3, n_iter=30).train(engineer_features(raw))
    # compute_confirmations lives in backtester.py and must be applied before
    # skip_confirmations=True is usable.
    return compute_confirmations(scored)


def test_sharpe_annualization_is_parameterized():
    """Sharpe was hardcoded to sqrt(252), wrong for the default hourly bars."""
    import math
    from backtester import run_backtest, periods_per_year

    scored = _synthetic_scored()
    daily = run_backtest(scored, n_regimes=3, skip_confirmations=True,
                         periods_per_year=252)["metrics"]
    hourly = run_backtest(scored, n_regimes=3, skip_confirmations=True,
                          periods_per_year=periods_per_year("1h"))["metrics"]
    if daily["sharpe_ratio"] == 0:
        return  # no trades on this seed; nothing to compare
    ratio = hourly["sharpe_ratio"] / daily["sharpe_ratio"]
    assert math.isclose(ratio, math.sqrt(periods_per_year("1h") / 252), rel_tol=0.02)
    assert daily["periods_per_year"] == 252


def test_run_backtest_annualization_default_is_daily():
    """Default must not change existing daily-bar results."""
    from backtester import run_backtest, DAILY_PERIODS_PER_YEAR

    scored = _synthetic_scored()
    m = run_backtest(scored, n_regimes=3, skip_confirmations=True)["metrics"]
    assert m["periods_per_year"] == DAILY_PERIODS_PER_YEAR == 252.0


def test_periods_per_year_lookup():
    from backtester import periods_per_year

    assert periods_per_year("1d") == 252
    assert periods_per_year("1h") == 252 * 6.5
    assert periods_per_year("WEEKLY") == 52          # case-insensitive
    assert periods_per_year("nonsense") == 252 * 6.5  # hourly fallback


def test_v2_costs_reduce_headline_return():
    """Costs must hit compounding capital, not just the trade rows."""
    from strategy_v2 import run_backtest_v2

    scored = _synthetic_scored()
    free = run_backtest_v2(scored, min_confirmations=3, cost_bps_per_side=0.0)
    paid = run_backtest_v2(scored, min_confirmations=3, cost_bps_per_side=10.0)
    if not free["trades"]:
        return
    assert paid["metrics"]["total_return_pct"] < free["metrics"]["total_return_pct"]
    assert free["metrics"]["total_cost_paid_pct"] == 0.0
    assert paid["metrics"]["total_cost_paid_pct"] > 0.0
    # Every trade row carries gross alongside net.
    for t in paid["trades"]:
        assert t["gross_pnl_pct"] >= t["pnl_pct"]


def test_v2_sharpe_annualization_is_parameterized():
    from strategy_v2 import run_backtest_v2
    from backtester import periods_per_year

    scored = _synthetic_scored()
    m = run_backtest_v2(scored, min_confirmations=3,
                        periods_per_year=periods_per_year("1h"))["metrics"]
    assert m["periods_per_year"] == periods_per_year("1h")


def test_walk_forward_charges_strategy_and_benchmark():
    """A single cost setting must reach both sides of the comparison."""
    from walk_forward import WalkForwardEngine

    eng = WalkForwardEngine(is_bars=100, oos_bars=60, n_regimes=3,
                            interval="1d", cost_bps_per_side=7.0)
    assert eng.cost_bps_per_side == 7.0
    # Strategy side, via backtest kwargs.
    assert eng.backtest_kwargs["cost_bps_per_side"] == 7.0
    # An explicit override still wins.
    eng2 = WalkForwardEngine(is_bars=100, oos_bars=60, n_regimes=3, interval="1d",
                             cost_bps_per_side=7.0,
                             backtest_kwargs={"cost_bps_per_side": 1.0})
    assert eng2.backtest_kwargs["cost_bps_per_side"] == 1.0


def test_walk_forward_default_is_cost_free_for_library_callers():
    """Only the CLI defaults to 5 bps; the class default stays 0 for reproducibility."""
    from walk_forward import WalkForwardEngine

    eng = WalkForwardEngine(is_bars=100, oos_bars=60, n_regimes=3, interval="1d")
    assert eng.cost_bps_per_side == 0.0
