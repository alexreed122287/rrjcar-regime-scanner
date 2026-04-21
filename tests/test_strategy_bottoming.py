"""Tests for strategy_bottoming.py."""
import pandas as pd
import pytest


def test_bottoming_buy_fixture_passes_gate_and_meets_buy_threshold(bottoming_buy_fixture):
    from strategy_bottoming import compute_bottoming_confirmations
    out = compute_bottoming_confirmations(bottoming_buy_fixture)
    latest = out.iloc[-1]

    # Must have exactly 12 confirmation columns
    conf_cols = [c for c in out.columns if c.startswith("conf_")]
    assert len(conf_cols) == 12, f"expected 12 conf columns, got {len(conf_cols)}"

    # Gate confs (Layer 1) must pass on the designed BUY fixture
    assert bool(latest["conf_01_drawdown_depth"]), "drawdown gate (conf_01) should pass"
    assert bool(latest["conf_02_off_lows"]), "off-lows gate (conf_02) should pass"

    # Total must meet BUY threshold (≥9/12) — synthetic fixtures don't always hit all 12
    # (e.g. higher-low structure depends on exact base phasing), but the fixture is
    # designed so the clear majority of signals fire.
    assert int(latest["confirmations_met"]) >= 9, (
        f"expected ≥9/12 confirmations, got {latest['confirmations_met']}"
    )


def test_drawdown_gate_fails_on_all_time_high_ticker(make_ohlcv, attach_regime):
    """A steadily up-trending stock near highs should fail conf_01 (drawdown depth)."""
    from strategy_bottoming import compute_bottoming_confirmations

    df = make_ohlcv(n_bars=260, start_price=50, trend_slope=0.003, volatility=0.005)
    df = attach_regime(df, regime_id=0, label="Bull Run")
    out = compute_bottoming_confirmations(df)
    latest = out.iloc[-1]

    assert bool(latest["conf_01_drawdown_depth"]) is False


def test_off_lows_gate_fails_on_freefalling_ticker(make_ohlcv, attach_regime):
    """A stock currently at its 52w low should fail conf_02 (off lows)."""
    from strategy_bottoming import compute_bottoming_confirmations

    df = make_ohlcv(n_bars=260, start_price=200, trend_slope=-0.005, volatility=0.01)
    df = attach_regime(df, regime_id=6, label="Crash / Capitulation")
    out = compute_bottoming_confirmations(df)
    latest = out.iloc[-1]

    # Price should be near its 52w low after 260 bars of downtrend → conf_02 False
    assert bool(latest["conf_02_off_lows"]) is False
