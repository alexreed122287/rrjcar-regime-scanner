"""Shared pytest fixtures for strategy tests."""
import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(
    n_bars: int = 260,
    start_price: float = 100.0,
    trend_slope: float = 0.0,
    volatility: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily OHLCV data."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend_slope, volatility, n_bars)
    closes = start_price * np.exp(np.cumsum(returns))
    highs = closes * (1 + np.abs(rng.normal(0, volatility / 2, n_bars)))
    lows = closes * (1 - np.abs(rng.normal(0, volatility / 2, n_bars)))
    opens = np.roll(closes, 1)
    opens[0] = start_price
    volume = rng.integers(500_000, 2_000_000, n_bars)
    dates = pd.date_range(end="2026-04-21", periods=n_bars, freq="B")
    return pd.DataFrame(
        {
            "Open": opens,
            "High": np.maximum.reduce([opens, highs, closes]),
            "Low": np.minimum.reduce([opens, lows, closes]),
            "Close": closes,
            "Volume": volume,
        },
        index=dates,
    )


def _attach_regime(df: pd.DataFrame, regime_id: int = 2, label: str = "Mild Bull") -> pd.DataFrame:
    """Attach regime columns matching hmm_engine output."""
    out = df.copy()
    out["regime_id"] = regime_id
    out["regime_label"] = label
    out["regime_confidence"] = 0.85
    return out


@pytest.fixture
def make_ohlcv():
    return _make_ohlcv


@pytest.fixture
def attach_regime():
    return _attach_regime


@pytest.fixture
def bottoming_buy_fixture(make_ohlcv, attach_regime):
    """
    Construct a DataFrame that satisfies all 12 bottoming confirmations.

    Shape of the setup:
      - 252 bars of downtrend (price fell from 200 to 60 — >=35% off high).
      - 20 bars of tight base at ~72 (15%+ off low of 60).
      - Final bar breaks out above the 20-day high on 2x volume with strong close.
      - 50 EMA reclaim, 10 EMA > 20 EMA, MACD rising.
    """
    n_down = 200
    n_base = 20
    n_breakout = 1
    dates = pd.date_range(end="2026-04-21", periods=n_down + n_base + n_breakout, freq="B")

    # Downtrend phase: 200 -> 60
    down_closes = np.linspace(200, 60, n_down)
    # Base phase: tight range around 70-75
    base_closes = 72 + np.sin(np.linspace(0, 3, n_base)) * 1.5
    # Breakout phase: 80 (> 20-day high of ~75)
    breakout_closes = np.array([80.0])

    closes = np.concatenate([down_closes, base_closes, breakout_closes])
    highs = closes.copy()
    lows = closes.copy()
    opens = np.roll(closes, 1)
    opens[0] = closes[0]

    # Engineer breakout bar: high=80, low=76, close=79.5 (strong close), vol 2x avg
    highs[-1] = 80.0
    lows[-1] = 76.0
    opens[-1] = 76.5
    closes[-1] = 79.5

    # Base bars: tight ranges
    for i in range(n_down, n_down + n_base):
        highs[i] = closes[i] + 0.5
        lows[i] = closes[i] - 0.5

    volume = np.full(len(closes), 1_000_000)
    volume[-1] = 2_500_000  # volume surge on breakout
    # Volume dry-up during base
    volume[n_down:n_down + n_base] = 600_000

    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volume},
        index=dates,
    )
    return attach_regime(df, regime_id=2, label="Mild Bull")
