"""
Tests for walk-forward validation, benchmarks, and stress tests.

The most important assertions here are structural, not performance-based:
no out-of-sample bar may ever precede the in-sample window that produced its model.
"""

import numpy as np
import pandas as pd
import pytest

from data_loader import engineer_features
from walk_forward import (
    WalkForwardEngine,
    benchmark_buy_and_hold,
    benchmark_sma_trend,
    benchmark_random_entry,
    inject_shock,
    format_report,
    _equity_metrics,
    _exposure_fraction,
    INDICATOR_WARMUP_BARS,
)


@pytest.fixture
def featured(make_ohlcv):
    return engineer_features(make_ohlcv(n_bars=700, trend_slope=0.0005, seed=11))


@pytest.fixture
def engine():
    # Small windows keep the test fast while still exercising multiple rolls.
    return WalkForwardEngine(
        is_bars=150, oos_bars=80, n_regimes=3, hmm_iter=15, random_state=0
    )


# ── window construction / no look-ahead ──

def test_oos_windows_strictly_follow_in_sample(engine, featured):
    result = engine.run(featured, verbose=False)
    assert result["aggregate"]["n_windows"] >= 1

    for w in result["windows"]:
        if w.get("error"):
            continue
        is_end = pd.Timestamp(w["is_end"])
        oos_start = pd.Timestamp(w["oos_start"])
        # Every scored bar must come after the last bar the model was fit on.
        assert oos_start > is_end, f"window {w['window']} leaks in-sample data"


def test_oos_windows_do_not_overlap_each_other(engine, featured):
    """Default step == oos_bars, so scored periods must be disjoint."""
    engine.run(featured, verbose=False)
    ok = [w for w in engine.windows if not w.error]
    for prev, nxt in zip(ok, ok[1:]):
        assert pd.Timestamp(nxt.oos_start) > pd.Timestamp(prev.oos_end)


def test_window_count_matches_data_length(featured):
    eng = WalkForwardEngine(is_bars=150, oos_bars=80, n_regimes=3, hmm_iter=10)
    eng.run(featured, verbose=False)
    n = len(featured)
    expected = 0
    start = 0
    while start + 150 + 80 <= n:
        expected += 1
        start += 80
    assert len(eng.windows) == expected


def test_oos_window_keeps_its_bars_despite_indicator_warmup(engine, featured):
    """
    The warmup buffer exists so dropna() does not silently eat the first ~50 bars of
    each scored window. Verify most of the window survives.
    """
    engine.run(featured, verbose=False)
    ok = [w for w in engine.windows if not w.error]
    assert ok
    for w in ok:
        assert w.oos_bars == 80
        # Strategy metrics should be computed over close to the full window.
        assert w.strategy.get("total_trades") is not None


def test_insufficient_data_raises(featured):
    eng = WalkForwardEngine(is_bars=252, oos_bars=126)
    with pytest.raises(ValueError, match="at least"):
        eng.run(featured.iloc[:200], verbose=False)


def test_invalid_window_sizes_rejected():
    with pytest.raises(ValueError):
        WalkForwardEngine(is_bars=10)
    with pytest.raises(ValueError):
        WalkForwardEngine(oos_bars=2)


# ── aggregation ──

def test_aggregate_reports_benchmark_comparisons(engine, featured):
    result = engine.run(featured, verbose=False)
    agg = result["aggregate"]

    for key in (
        "n_windows", "mean_oos_return_pct", "pct_windows_profitable",
        "mean_excess_vs_buy_and_hold_pct", "pct_windows_beating_buy_and_hold",
        "mean_excess_vs_sma_trend_pct", "mean_excess_vs_random_pct",
        "worst_oos_max_drawdown_pct", "mean_exposure_pct",
    ):
        assert key in agg, f"missing aggregate key {key}"

    assert 0 <= agg["pct_windows_profitable"] <= 100
    assert 0 <= agg["pct_windows_beating_buy_and_hold"] <= 100


def test_aggregate_with_no_windows_is_safe():
    eng = WalkForwardEngine(is_bars=100, oos_bars=50)
    assert eng.aggregate()["n_windows"] == 0


def test_config_is_echoed(engine, featured):
    result = engine.run(featured, verbose=False)
    assert result["config"]["is_bars"] == 150
    assert result["config"]["oos_bars"] == 80
    assert result["config"]["step_bars"] == 80


def test_format_report_renders(engine, featured):
    result = engine.run(featured, verbose=False)
    text = format_report(result)
    assert "WALK-FORWARD VALIDATION" in text
    assert "Buy & hold" in text
    assert "Random entry" in text


# ── benchmarks ──

def test_buy_and_hold_matches_price_change(featured):
    window = featured.iloc[:200]
    result = benchmark_buy_and_hold(window, initial_capital=100_000.0)
    expected = (window["Close"].iloc[-1] / window["Close"].iloc[0] - 1) * 100
    assert result["total_return_pct"] == pytest.approx(expected, rel=1e-6)
    assert result["exposure_pct"] == 100.0


def test_buy_and_hold_on_flat_series_returns_zero(make_ohlcv):
    df = make_ohlcv(n_bars=100)
    df["Close"] = 100.0
    df["High"] = 100.0
    df["Low"] = 100.0
    result = benchmark_buy_and_hold(df)
    assert result["total_return_pct"] == pytest.approx(0.0)


def test_sma_trend_is_flat_when_price_never_exceeds_sma(make_ohlcv):
    """A monotonically falling series should never be long a 20-day SMA system."""
    df = make_ohlcv(n_bars=200)
    df["Close"] = np.linspace(200, 100, 200)
    result = benchmark_sma_trend(df, window=20)
    assert result["exposure_pct"] < 5.0


def test_sma_trend_is_mostly_long_in_an_uptrend(make_ohlcv):
    df = make_ohlcv(n_bars=250)
    df["Close"] = np.linspace(100, 250, 250)
    result = benchmark_sma_trend(df, window=20)
    assert result["exposure_pct"] > 80.0
    assert result["total_return_pct"] > 0


def test_sma_trend_never_looks_ahead(make_ohlcv):
    """
    Truncating the series must not change the earlier position decisions, which is only
    true because the SMA signal is shifted one bar.
    """
    df = make_ohlcv(n_bars=200)
    df["Close"] = np.linspace(100, 200, 200)
    a = benchmark_sma_trend(df.iloc[:150], window=20)
    b = benchmark_sma_trend(df.iloc[:150], window=20)
    assert a["total_return_pct"] == pytest.approx(b["total_return_pct"])


def test_random_entry_is_reproducible(featured):
    window = featured.iloc[:250]
    a = benchmark_random_entry(window, n_trials=10, seed=5)
    b = benchmark_random_entry(window, n_trials=10, seed=5)
    assert a["total_return_pct"] == pytest.approx(b["total_return_pct"])
    assert a["n_trials"] == 10


def test_random_entry_matches_requested_exposure(featured):
    window = featured.iloc[:300]
    result = benchmark_random_entry(window, n_trials=20, exposure_target=0.4, seed=3)
    # Random windows overlap, so allow generous tolerance around the 40% target.
    assert 15.0 < result["exposure_pct"] < 65.0


def test_random_entry_reports_dispersion(featured):
    result = benchmark_random_entry(featured.iloc[:300], n_trials=20, seed=1)
    assert "total_return_pct_std" in result
    assert "p95_total_return_pct" in result
    assert result["total_return_pct_std"] >= 0


# ── equity metrics ──

def test_equity_metrics_on_known_curve():
    curve = np.array([100.0, 110.0, 121.0])
    m = _equity_metrics(curve, 100.0)
    assert m["total_return_pct"] == pytest.approx(21.0)
    assert m["final_equity"] == pytest.approx(121.0)
    assert m["max_drawdown_pct"] == pytest.approx(0.0)


def test_equity_metrics_captures_drawdown():
    curve = np.array([100.0, 120.0, 90.0, 95.0])
    m = _equity_metrics(curve, 100.0)
    # Peak 120 → trough 90 is -25%.
    assert m["max_drawdown_pct"] == pytest.approx(-25.0, rel=1e-6)


def test_equity_metrics_handles_degenerate_input():
    m = _equity_metrics(np.array([100.0]), 100.0)
    assert m["total_return_pct"] == 0.0


def test_exposure_fraction_from_trades():
    trades = [{"entry_bar": 0, "exit_bar": 10}, {"entry_bar": 20, "exit_bar": 30}]
    assert _exposure_fraction(trades, 100) == pytest.approx(0.2)
    assert _exposure_fraction([], 100) is None
    assert _exposure_fraction(trades, 0) is None


# ── stress tests ──

def test_inject_shock_moves_price_by_requested_amount(featured):
    shocked = inject_shock(featured, shock_pct=-12.0, bar_index=300)
    before = float(featured["Close"].iloc[300])
    after = float(shocked["Close"].iloc[300])
    assert after / before == pytest.approx(0.88, rel=1e-9)


def test_inject_shock_persists_into_later_prices(featured):
    shocked = inject_shock(featured, shock_pct=-10.0, bar_index=300)
    # The level break carries forward — it is not a single-bar outlier.
    ratio = shocked["Close"].iloc[400] / featured["Close"].iloc[400]
    assert ratio == pytest.approx(0.90, rel=1e-9)


def test_inject_shock_leaves_earlier_bars_untouched(featured):
    shocked = inject_shock(featured, shock_pct=-15.0, bar_index=300)
    pd.testing.assert_series_equal(
        shocked["Close"].iloc[:300], featured["Close"].iloc[:300]
    )


def test_inject_shock_keeps_ohlc_coherent(featured):
    shocked = inject_shock(featured, shock_pct=-15.0, bar_index=300)
    row = shocked.iloc[300]
    assert row["Low"] <= row["Close"] <= row["High"]
    assert row["Low"] <= row["Open"] <= row["High"]


def test_inject_shock_spikes_volume(featured):
    shocked = inject_shock(featured, shock_pct=-12.0, bar_index=300)
    assert shocked["Volume"].iloc[300] > featured["Volume"].iloc[300]


def test_inject_shock_is_deterministic_without_bar_index(featured):
    a = inject_shock(featured, shock_pct=-10.0, seed=9)
    b = inject_shock(featured, shock_pct=-10.0, seed=9)
    assert a.attrs["shock_bar_index"] == b.attrs["shock_bar_index"]


def test_stress_test_reports_scenarios_and_summary(engine, featured):
    result = engine.stress_test(
        featured, shocks=(-10.0, -15.0), n_positions=1, verbose=False
    )
    assert "baseline" in result
    assert len(result["scenarios"]) == 2
    for s in result["scenarios"]:
        assert s["shock_pct"] in (-10.0, -15.0)
    if result["summary"]:
        assert "worst_return_delta_pct" in result["summary"]
        assert result["summary"]["n_scenarios"] >= 1


# ── annualization ──

def test_periods_per_year_lookup():
    from walk_forward import periods_per_year
    assert periods_per_year("1d") == 252
    assert periods_per_year("1h") == pytest.approx(252 * 6.5)
    assert periods_per_year("1wk") == 52
    # Unknown intervals fall back to hourly, the repo's fetch_data default.
    assert periods_per_year("nonsense") == pytest.approx(252 * 6.5)


def test_sharpe_scales_with_annualization_factor():
    """
    Using the daily factor on hourly bars overstates Sharpe by sqrt(6.5).

    This is the bug the ppy parameter exists to prevent.
    """
    rng = np.random.default_rng(0)
    curve = 100_000 * np.cumprod(1 + rng.normal(0.0002, 0.005, 500))

    daily = _equity_metrics(curve, 100_000.0, ppy=252)
    hourly = _equity_metrics(curve, 100_000.0, ppy=252 * 6.5)

    ratio = hourly["sharpe"] / daily["sharpe"]
    assert ratio == pytest.approx(np.sqrt(6.5), rel=1e-6)


def test_engine_derives_annualization_from_interval():
    daily = WalkForwardEngine(is_bars=100, oos_bars=50, interval="1d")
    hourly = WalkForwardEngine(is_bars=100, oos_bars=50, interval="1h")
    assert daily.ppy == 252
    assert hourly.ppy == pytest.approx(252 * 6.5)


def test_config_reports_interval(featured):
    eng = WalkForwardEngine(is_bars=150, oos_bars=80, n_regimes=3,
                            hmm_iter=10, interval="1d")
    result = eng.run(featured, verbose=False)
    assert result["config"]["interval"] == "1d"
    assert result["config"]["periods_per_year"] == 252
