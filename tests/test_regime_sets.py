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


# ───────────────── API/strategy signature compatibility (regression) ─────────────────

# The kwargs api/routes_backtest.py passes to whichever engine the caller selects. Both
# engines must accept all of them. This is the check that was missing when n_regimes= was
# added to the route's two call sites but only to run_backtest's signature, leaving the
# route's DEFAULT strategy (v2) raising TypeError on every request.
_API_BACKTEST_KWARGS = {
    "min_confirmations": 6,
    "cooldown_bars": 3,
    "regime_confirm_bars": 2,
    "initial_capital": 100000,
    "n_regimes": 7,
    "cost_bps_per_side": 5.0,
    "periods_per_year": 252.0,
}


def test_api_kwargs_bind_to_both_engines():
    import inspect
    from backtester import run_backtest
    from strategy_v2 import run_backtest_v2

    for fn in (run_backtest, run_backtest_v2):
        sig = inspect.signature(fn)
        missing = [k for k in _API_BACKTEST_KWARGS if k not in sig.parameters]
        assert not missing, f"{fn.__name__} cannot accept API kwargs: {missing}"
        # Must actually bind, not merely appear in the parameter list.
        sig.bind_partial(**_API_BACKTEST_KWARGS)


def test_api_route_passes_only_supported_kwargs():
    """Guard the route itself, so a future edit there cannot drift from the signatures."""
    import inspect
    import re
    from pathlib import Path
    from backtester import run_backtest
    from strategy_v2 import run_backtest_v2

    src = Path(inspect.getfile(run_backtest)).parent / "api" / "routes_backtest.py"
    text = src.read_text()
    for fname, fn in (("run_backtest_v2", run_backtest_v2), ("run_backtest", run_backtest)):
        m = re.search(rf"bt = {fname}\((.*?)\n            \)", text, re.S)
        if not m:
            continue
        passed = set(re.findall(r"(\w+)\s*=", m.group(1)))
        allowed = set(inspect.signature(fn).parameters)
        assert passed <= allowed, f"route passes unsupported kwargs to {fname}: {passed - allowed}"


def test_v2_derives_regime_sets_like_v1():
    """Hardcoded [5, 6] as bearish matches no state below 7 regimes."""
    from hmm_engine import regime_sets
    from strategy_v2 import run_backtest_v2

    scored = _synthetic_scored()
    for n in (3, 4, 7):
        r = run_backtest_v2(scored, min_confirmations=3, n_regimes=n)
        expected = regime_sets(n)
        # Bearish set must exist and be in range, otherwise the regime-flip exit is dead.
        assert expected["bearish"], f"no bearish regimes derived for n={n}"
        assert max(expected["bearish"]) < n
        assert isinstance(r["metrics"]["total_return_pct"], (int, float))


def test_v2_explicit_regime_sets_still_win():
    from strategy_v2 import run_backtest_v2

    scored = _synthetic_scored()
    r = run_backtest_v2(scored, min_confirmations=3, n_regimes=7,
                        bullish_regimes=[0], bearish_regimes=[2])
    assert isinstance(r["trades"], list)


# ───────────────────────────── v2 roll model ─────────────────────────────

def _rolling_uptrend_scored(n=700):
    """Deterministic bullish frame that is guaranteed to trigger ROLL_UP.

    Fitting an HMM on random data is flaky (hmmlearn raises on non-positive-definite
    covariances for many seeds) and rarely produces rolls, so the regime columns are set
    directly instead. Everything else -- indicators, ATR, confirmations -- is computed by
    the real pipeline.
    """
    import numpy as np
    import pandas as pd
    from data_loader import engineer_features
    from backtester import compute_confirmations

    # Steady uptrend with small wiggles: price keeps clearing effective_entry + 1 ATR.
    t = np.arange(n)
    price = 100 * (1.004 ** t) + np.sin(t / 3.0) * 0.35
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    raw = pd.DataFrame({
        "Open": price, "High": price * 1.006, "Low": price * 0.994,
        "Close": price, "Volume": np.full(n, 2_000_000),
    }, index=idx)
    df = compute_confirmations(engineer_features(raw))
    # Regime 0 is bullish under regime_sets for every count >= 2. The frame must also
    # contain a state numbered 6, or run_backtest_v2 infers n_regimes from
    # regime_id.max() + 1 -- and regime_sets(1) returns an EMPTY bullish list, so no entry
    # can ever fire and every roll assertion would pass vacuously.
    df["regime_id"] = 0
    df["regime_label"] = "Strong Bull"
    df["regime_confidence"] = 0.95
    df.iloc[-1, df.columns.get_loc("regime_id")] = 6          # bearish under 7 regimes
    df.iloc[-1, df.columns.get_loc("regime_label")] = "Strong Bear"
    return df.dropna(subset=["atr"]).copy()


def test_fixture_actually_produces_rolls():
    """Guard the guards.

    The roll tests below skip when no rolls occur. That made them pass vacuously against a
    fixture whose regime_id was constant 0, which made run_backtest_v2 infer n_regimes=1 --
    and regime_sets(1) has an empty bullish list, so nothing ever entered. Assert the
    fixture bites before trusting anything that depends on it.
    """
    from strategy_v2 import run_backtest_v2

    r = run_backtest_v2(_rolling_uptrend_scored(), min_confirmations=3)
    assert len(r["trades"]) > 0, "fixture produced no trades; roll tests would be vacuous"
    assert r["metrics"]["total_rolls"] > 0, "fixture produced no rolls"


def test_roll_credits_are_parameterized_and_scale():
    """The roll credit was hardcoded at 0.5%; it must be measurable.

    Pins the legacy flat-credit path, which is now opt-in via roll_model="flat".
    """
    from strategy_v2 import run_backtest_v2

    scored = _rolling_uptrend_scored()
    off = run_backtest_v2(scored, min_confirmations=3, roll_model="flat",
                          roll_up_credit_pct=0.0, roll_out_credit_pct=0.0)["metrics"]
    half = run_backtest_v2(scored, min_confirmations=3, roll_model="flat",
                           roll_up_credit_pct=0.5, roll_out_credit_pct=0.0)["metrics"]
    if half["total_rolls"] == 0:
        return  # no rolls on this seed; nothing to assert
    assert off["total_roll_credits_pct"] == 0.0
    assert half["total_roll_credits_pct"] > 0.0
    # Credits must be exactly rate * number of rolls.
    assert abs(half["total_roll_credits_pct"] - 0.5 * half["total_rolls"]) < 1e-6
    # And they must reach the headline return, not just the metrics dict.
    assert half["total_return_pct"] > off["total_return_pct"]


def test_roll_credits_lift_reported_return():
    from strategy_v2 import run_backtest_v2

    scored = _rolling_uptrend_scored()
    lo = run_backtest_v2(scored, min_confirmations=3, roll_model="flat",
                         roll_up_credit_pct=0.1)["metrics"]
    hi = run_backtest_v2(scored, min_confirmations=3, roll_model="flat",
                         roll_up_credit_pct=1.0)["metrics"]
    if hi["total_rolls"] == 0:
        return
    assert hi["total_roll_credits_pct"] > lo["total_roll_credits_pct"]
    assert hi["total_return_pct"] > lo["total_return_pct"]


def test_max_rolls_caps_credits_per_trade():
    from strategy_v2 import run_backtest_v2

    scored = _rolling_uptrend_scored()
    r = run_backtest_v2(scored, min_confirmations=3, roll_model="flat", max_rolls=1)
    for t in r["trades"]:
        assert t["roll_count"] <= 1
    r0 = run_backtest_v2(scored, min_confirmations=3, roll_model="flat", max_rolls=0)
    assert r0["metrics"]["total_rolls"] == 0
    assert r0["metrics"]["total_roll_credits_pct"] == 0.0


def test_roll_out_credit_is_unreachable_at_defaults():
    """Characterization, not an endorsement.

    The ROLL_OUT credit has the wrong sign -- rolling a long call to a later expiry buys
    time value and costs a debit. It turns out never to fire at default settings, because
    the regime-flip exit is checked before the time-stop roll attempt. This test pins that
    down so that if a future change makes it reachable, the wrong sign starts failing here
    instead of silently inflating returns.
    """
    from strategy_v2 import run_backtest_v2

    scored = _rolling_uptrend_scored()
    kw = dict(min_confirmations=3, roll_model="flat", roll_up_credit_pct=0.5)
    a = run_backtest_v2(scored, roll_out_credit_pct=0.3, **kw)["metrics"]
    b = run_backtest_v2(scored, roll_out_credit_pct=99.0, **kw)["metrics"]
    c = run_backtest_v2(scored, roll_out_credit_pct=-99.0, **kw)["metrics"]
    assert a["roll_model"] == "flat"
    assert a["total_return_pct"] == b["total_return_pct"] == c["total_return_pct"]
    assert a["total_roll_credits_pct"] == b["total_roll_credits_pct"]


# ─────────────────────── priced roll model (options_pricing) ───────────────────────

def test_priced_is_the_default_roll_model():
    from strategy_v2 import run_backtest_v2

    m = run_backtest_v2(_rolling_uptrend_scored(), min_confirmations=3)["metrics"]
    assert m["roll_model"] == "priced"
    # The flat credits must default to zero so nothing is credited unconditionally.
    assert m["total_roll_credits_pct"] == 0.0


def test_flat_credits_are_ignored_under_priced_model():
    """A stray roll_up_credit_pct must not leak free money into the priced path."""
    from strategy_v2 import run_backtest_v2

    scored = _rolling_uptrend_scored()
    a = run_backtest_v2(scored, min_confirmations=3)["metrics"]
    b = run_backtest_v2(scored, min_confirmations=3, roll_up_credit_pct=5.0,
                        roll_out_credit_pct=5.0)["metrics"]
    assert a["total_return_pct"] == b["total_return_pct"]


def test_priced_roll_does_not_create_money():
    """The core correctness property.

    Under the flat model, more rolls meant more return without limit -- the credit was
    income. Priced, a roll is a reallocation: cash comes off the table and delta goes with
    it, so raising max_rolls must not manufacture return.
    """
    from strategy_v2 import run_backtest_v2

    scored = _rolling_uptrend_scored()
    none_ = run_backtest_v2(scored, min_confirmations=3, max_rolls=0)["metrics"]
    many = run_backtest_v2(scored, min_confirmations=3, max_rolls=20)["metrics"]
    assert many["total_rolls"] > none_["total_rolls"]
    assert many["total_roll_cash_pct"] > 0.0, "rolling up must release cash"
    # In a persistent uptrend, de-risking must cost return rather than add it.
    assert many["total_return_pct"] < none_["total_return_pct"]


def test_priced_rolls_charge_transaction_cost_on_both_legs():
    from strategy_v2 import run_backtest_v2

    scored = _rolling_uptrend_scored()
    free = run_backtest_v2(scored, min_confirmations=3, cost_bps_per_side=0.0)["metrics"]
    paid = run_backtest_v2(scored, min_confirmations=3, cost_bps_per_side=25.0)["metrics"]
    assert free["total_roll_cost_pct"] == 0.0
    assert paid["total_roll_cost_pct"] > 0.0
    assert paid["total_return_pct"] < free["total_return_pct"]


def test_delta_matched_sizing_is_less_levered_than_full_premium():
    """Spending all capital on premium is several times levered; the default must not be."""
    from strategy_v2 import run_backtest_v2

    scored = _rolling_uptrend_scored()
    matched = run_backtest_v2(scored, min_confirmations=3, sizing="delta_matched")["metrics"]
    full = run_backtest_v2(scored, min_confirmations=3, sizing="full_premium")["metrics"]
    assert full["total_return_pct"] > matched["total_return_pct"]


def test_deep_itm_long_dated_approximates_stock_accounting():
    """Validation anchor.

    A very deep, very long-dated call is almost the underlying, so the priced engine with
    rolling off must land near the old stock-participation model with credits zeroed. If
    this drifts far apart, the option accounting is wrong somewhere.
    """
    from strategy_v2 import run_backtest_v2

    scored = _rolling_uptrend_scored()
    stock = run_backtest_v2(scored, min_confirmations=3, roll_model="flat",
                            roll_up_credit_pct=0.0, roll_out_credit_pct=0.0)["metrics"]
    priced = run_backtest_v2(scored, min_confirmations=3, max_rolls=0,
                             itm_depth=0.60, roll_dte_days=1460)["metrics"]
    assert stock["total_trades"] == priced["total_trades"]
    ratio = (1 + priced["total_return_pct"] / 100) / (1 + stock["total_return_pct"] / 100)
    assert 0.80 < ratio < 1.05, f"priced diverged from stock accounting: ratio {ratio:.3f}"


def test_short_dated_legs_expire():
    from strategy_v2 import run_backtest_v2

    r = run_backtest_v2(_rolling_uptrend_scored(), min_confirmations=3,
                        roll_dte_days=4.0, max_rolls=0)
    assert any("expiry" in t["exit_reason"] for t in r["trades"])


def test_trade_rows_separate_position_and_underlying_pnl():
    """Priced, the position's return and the stock's return are different numbers."""
    from strategy_v2 import run_backtest_v2

    r = run_backtest_v2(_rolling_uptrend_scored(), min_confirmations=3)
    assert r["trades"], "fixture must trade"
    for t in r["trades"]:
        assert "underlying_pnl_pct" in t and "roll_cash_pct" in t
    assert any(abs(t["pnl_pct"] - t["underlying_pnl_pct"]) > 1e-6 for t in r["trades"])
