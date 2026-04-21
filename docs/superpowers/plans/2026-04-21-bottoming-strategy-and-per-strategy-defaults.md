# Bottoming Strategy + Per-Strategy Defaults — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Minervini-style "Bottoming" strategy module and a backend-owned per-strategy recommended-defaults system with a one-click "Load recommended defaults" button.

**Architecture:** Each strategy module exports a `RECOMMENDED_SETTINGS` flat dict. A new FastAPI route in `api/routes_settings.py` aggregates them. The frontend fetches once and applies the selected strategy's defaults to the settings form on demand. The new strategy follows the established `strategy_*.py` pattern — a `compute_*_confirmations` DataFrame transform + a `get_current_signal_*` dict returner.

**Tech Stack:** Python 3.11 / FastAPI / pandas / `ta` (technical indicators) / pytest for tests / vanilla JS frontend.

**Working directory:** `/Users/alex/rrjcar-regime-scanner` on branch `main`.

**Spec:** `docs/superpowers/specs/2026-04-21-bottoming-strategy-and-per-strategy-defaults-design.md`

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `tests/__init__.py` | **new** | Empty — marks tests dir as package |
| `tests/conftest.py` | **new** | Shared pytest fixtures (synthetic OHLCV builder, fake regime dataframe) |
| `tests/test_strategy_bottoming.py` | **new** | Unit tests for `compute_bottoming_confirmations` + `get_current_signal_bottoming` |
| `tests/test_api_strategy_defaults.py` | **new** | Endpoint shape test |
| `strategy_bottoming.py` | **new** | 12-conf bottoming strategy: confirmations, signal, `RECOMMENDED_SETTINGS` |
| `backtester.py` | edit | Append `RECOMMENDED_SETTINGS` constant |
| `strategy_v2.py` | edit | Append `RECOMMENDED_SETTINGS` constant |
| `strategy_leaps.py` | edit | Append `RECOMMENDED_SETTINGS` constant |
| `settings_manager.py` | edit | Extend `DEFAULT_SETTINGS` with 5 new keys |
| `api/routes_settings.py` | edit | Add `SettingsUpdate` fields + `GET /strategy-defaults` route |
| `screener.py` | edit | Import + dispatch `"bottoming"`; raise `ValueError` on unknown strategy |
| `index.html` | edit | Add strategy `<option>`, "Load recommended defaults" button, `BOTTOM -- BUY/WATCH` filter options |
| `js/api.js` | edit | Add `getStrategyDefaults()` with session cache |
| `js/settings.js` | edit | Add `loadRecommendedDefaults()` and `flashButton()` |

---

## Task 1: Test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `tests/__init__.py`**

```bash
mkdir -p ~/rrjcar-regime-scanner/tests
touch ~/rrjcar-regime-scanner/tests/__init__.py
```

- [ ] **Step 2: Install pytest into the project venv**

```bash
cd ~/rrjcar-regime-scanner
source venv/bin/activate
pip install pytest pytest-asyncio httpx
pip freeze | grep -E 'pytest|httpx' >> requirements-dev.txt
```

Create `requirements-dev.txt` if it doesn't exist.

- [ ] **Step 3: Create `tests/conftest.py` with synthetic OHLCV builder**

```python
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
      - 252 bars of downtrend (price fell from 200 to 60 — ≥35% off high).
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
```

- [ ] **Step 4: Create `pytest.ini` at repo root**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
asyncio_mode = auto
```

- [ ] **Step 5: Verify pytest discovers nothing yet (no tests defined)**

Run: `cd ~/rrjcar-regime-scanner && pytest -q`
Expected: `no tests ran in 0.0Xs`

- [ ] **Step 6: Commit**

```bash
cd ~/rrjcar-regime-scanner
git add tests/__init__.py tests/conftest.py pytest.ini requirements-dev.txt
git commit -m "test: add pytest infrastructure with synthetic OHLCV fixtures"
```

---

## Task 2: Bottoming strategy — `compute_bottoming_confirmations`

**Files:**
- Create: `strategy_bottoming.py`
- Create: `tests/test_strategy_bottoming.py`

- [ ] **Step 1: Write failing test — all 12 confirmations pass on a designed BUY fixture**

Create `tests/test_strategy_bottoming.py`:

```python
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
```

- [ ] **Step 2: Run tests — confirm failure**

Run: `cd ~/rrjcar-regime-scanner && pytest tests/test_strategy_bottoming.py -v`
Expected: `ModuleNotFoundError: No module named 'strategy_bottoming'` (or equivalent ImportError)

- [ ] **Step 3: Implement `strategy_bottoming.py` — confirmations only**

Create `strategy_bottoming.py`:

```python
"""
strategy_bottoming.py — Bottoming Stocks Strategy

Minervini-style base-and-breakout with trend-reclaim overlay for stocks
that have been beaten down ≥35% off 52w high, recovered ≥15% off 52w low,
and are now breaking out of a tight base.

12 confirmations in 4 layers:
  Layer 1 — Drawdown gate (hard requirement):
    1. ≥35% off 52-week high
    2. ≥15% off 52-week low

  Layer 2 — Base formation:
    3. Tight base (20-day ATR/price ≤ 8%)
    4. Range contraction (20-day range <20% of price)
    5. Higher-low structure in base
    6. Volume dry-up (5-day avg < 20-day avg)

  Layer 3 — Breakout trigger:
    7. Breakout day (close > prior 20-day high)
    8. Volume surge (volume > 1.5× 20-day avg)
    9. Strong close (upper half of day's range)

  Layer 4 — Trend reclaim overlay:
    10. Price > 50 EMA
    11. 10 EMA > 20 EMA
    12. MACD histogram > 0 AND rising
"""

import numpy as np
import pandas as pd
import ta


def compute_bottoming_confirmations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 12 bottoming confirmations. Returns DataFrame with conf_* columns and confirmations_met."""
    out = df.copy()

    # ── Core rolling aggregates ──
    out["high_252"] = out["High"].rolling(252, min_periods=60).max()
    out["low_252"] = out["Low"].rolling(252, min_periods=60).min()

    # ── Technical indicators ──
    atr_ind = ta.volatility.AverageTrueRange(out["High"], out["Low"], out["Close"], window=20)
    out["atr_20"] = atr_ind.average_true_range()

    out["high_20"] = out["High"].rolling(20, min_periods=5).max()
    out["low_20"] = out["Low"].rolling(20, min_periods=5).min()

    out["low_10"] = out["Low"].rolling(10, min_periods=3).min()

    out["vol_ma_20"] = out["Volume"].rolling(20, min_periods=5).mean()
    out["vol_ma_5"] = out["Volume"].rolling(5, min_periods=2).mean()

    out["ema_10"] = ta.trend.EMAIndicator(out["Close"], window=10).ema_indicator()
    out["ema_20"] = ta.trend.EMAIndicator(out["Close"], window=20).ema_indicator()
    out["ema_50"] = ta.trend.EMAIndicator(out["Close"], window=50).ema_indicator()

    macd_ind = ta.trend.MACD(out["Close"], window_slow=26, window_fast=12, window_sign=9)
    out["macd_hist"] = macd_ind.macd_diff()

    # ═══ Layer 1 — Drawdown gate ═══

    # 1. ≥35% off 52-week high
    out["conf_01_drawdown_depth"] = out["Close"] <= 0.65 * out["high_252"]

    # 2. ≥15% off 52-week low
    out["conf_02_off_lows"] = out["Close"] >= 1.15 * out["low_252"]

    # ═══ Layer 2 — Base formation ═══

    # 3. Tight base: 20-day ATR/price ≤ 8%
    out["conf_03_tight_base"] = (out["atr_20"] / out["Close"]) <= 0.08

    # 4. Range contraction: (20-day High − 20-day Low) / Close < 20%
    out["conf_04_range_contraction"] = ((out["high_20"] - out["low_20"]) / out["Close"]) < 0.20

    # 5. Higher-low structure in base: current 10-day low > prior 10-day low
    out["conf_05_higher_lows"] = out["low_10"] > out["low_10"].shift(10)

    # 6. Volume dry-up: last 5-day avg < 20-day avg (supply exhausted)
    out["conf_06_volume_dryup"] = out["vol_ma_5"] < out["vol_ma_20"]

    # ═══ Layer 3 — Breakout trigger ═══

    # 7. Breakout day: close > prior 20-day high (shifted to exclude today)
    out["conf_07_breakout_day"] = out["Close"] > out["High"].shift(1).rolling(20, min_periods=5).max()

    # 8. Volume surge: today's volume > 1.5× 20-day avg
    out["conf_08_volume_surge"] = out["Volume"] > 1.5 * out["vol_ma_20"]

    # 9. Strong close: close in upper half of day's range
    midpoint = (out["High"] + out["Low"]) / 2
    out["conf_09_strong_close"] = out["Close"] > midpoint

    # ═══ Layer 4 — Trend reclaim overlay ═══

    # 10. Price > 50 EMA
    out["conf_10_above_50ema"] = out["Close"] > out["ema_50"]

    # 11. 10 EMA > 20 EMA
    out["conf_11_ema_stack"] = out["ema_10"] > out["ema_20"]

    # 12. MACD histogram > 0 AND rising
    out["conf_12_macd_rising"] = (out["macd_hist"] > 0) & (out["macd_hist"] > out["macd_hist"].shift(1))

    # ── Aggregate ──
    conf_cols = [c for c in out.columns if c.startswith("conf_")]
    for c in conf_cols:
        out[c] = out[c].fillna(False)
    out["confirmations_met"] = out[conf_cols].sum(axis=1).astype(int)

    return out
```

- [ ] **Step 4: Run tests — confirm all three pass**

Run: `cd ~/rrjcar-regime-scanner && pytest tests/test_strategy_bottoming.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
cd ~/rrjcar-regime-scanner
git add strategy_bottoming.py tests/test_strategy_bottoming.py
git commit -m "feat(bottoming): add compute_bottoming_confirmations with 12 signals"
```

---

## Task 3: Bottoming strategy — `get_current_signal_bottoming`

**Files:**
- Modify: `strategy_bottoming.py`
- Modify: `tests/test_strategy_bottoming.py`

- [ ] **Step 1: Add failing tests for signal output scenarios**

Append to `tests/test_strategy_bottoming.py`:

```python
def test_signal_buy_on_buy_fixture(bottoming_buy_fixture):
    from strategy_bottoming import get_current_signal_bottoming
    result = get_current_signal_bottoming(bottoming_buy_fixture, min_confirmations=8)

    assert result["signal"] == "BOTTOM -- BUY"
    assert result["confirmations_met"] >= 9      # BUY threshold
    assert result["confirmations_total"] == 12
    assert result["regime_label"] == "Mild Bull"


def test_signal_na_when_drawdown_gate_fails(make_ohlcv, attach_regime):
    """Stock near 52w high should return BOTTOM -- N/A (not a candidate)."""
    from strategy_bottoming import get_current_signal_bottoming

    df = make_ohlcv(n_bars=260, start_price=50, trend_slope=0.003, volatility=0.005)
    df = attach_regime(df, regime_id=0, label="Bull Run")
    result = get_current_signal_bottoming(df, min_confirmations=8)

    assert result["signal"] == "BOTTOM -- N/A"


def test_signal_avoid_in_crash_regime(bottoming_buy_fixture, attach_regime):
    """Even a valid bottoming setup should return AVOID in Crash regime."""
    from strategy_bottoming import get_current_signal_bottoming

    # Re-label regime to Crash on the BUY fixture
    df = bottoming_buy_fixture.copy()
    df["regime_id"] = 6
    df["regime_label"] = "Crash / Capitulation"

    result = get_current_signal_bottoming(df, min_confirmations=8)
    assert result["signal"] == "BOTTOM -- AVOID"


def test_signal_watch_in_bearish_regime(bottoming_buy_fixture):
    """Valid setup in a bearish (non-crash) regime → WATCH, not BUY."""
    from strategy_bottoming import get_current_signal_bottoming

    df = bottoming_buy_fixture.copy()
    df["regime_id"] = 5
    df["regime_label"] = "Bear Trend"

    result = get_current_signal_bottoming(df, min_confirmations=8)
    assert result["signal"] == "BOTTOM -- WATCH"


def test_returns_required_metadata_fields(bottoming_buy_fixture):
    from strategy_bottoming import get_current_signal_bottoming
    result = get_current_signal_bottoming(bottoming_buy_fixture, min_confirmations=8)

    required = {
        "signal", "action", "regime_id", "regime_label", "confidence",
        "confirmations_met", "confirmations_required", "confirmations_total",
        "confirmation_detail", "price", "pct_off_52w_high", "pct_off_52w_low",
    }
    missing = required - set(result.keys())
    assert not missing, f"missing keys: {missing}"
```

- [ ] **Step 2: Run tests — confirm failure**

Run: `cd ~/rrjcar-regime-scanner && pytest tests/test_strategy_bottoming.py -v`
Expected: 3 original tests pass, 5 new tests fail with `ImportError` or `AttributeError`.

- [ ] **Step 3: Append signal function to `strategy_bottoming.py`**

Append to `strategy_bottoming.py`:

```python
# Regime IDs: 0=Bull Run, 1=Bull Trend, 2=Mild Bull, 3=Neutral/Chop,
#             4=Mild Bear, 5=Bear Trend, 6=Crash/Capitulation
BULLISH_OR_NEUTRAL_REGIMES = {0, 1, 2, 3}
CRASH_REGIME = 6


def get_current_signal_bottoming(
    df: pd.DataFrame,
    min_confirmations: int = 8,
    regime_confirm_bars: int = 2,
) -> dict:
    """
    Produce the current bottoming signal for a ticker.

    Returns:
        dict with signal, action, regime info, confirmations breakdown, and
        bottoming-specific stats (pct_off_52w_high, pct_off_52w_low).
    """
    data = compute_bottoming_confirmations(df)
    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest

    regime_id = int(latest["regime_id"])
    regime_label = str(latest["regime_label"])
    confidence = float(latest.get("regime_confidence", 0.0))
    confs = int(latest["confirmations_met"])

    # Streak (how long has current regime held?)
    regime_ids = data["regime_id"].values.astype(int)
    streak = 1
    for j in range(len(regime_ids) - 2, -1, -1):
        if regime_ids[j] == regime_ids[-1]:
            streak += 1
        else:
            break

    # Drawdown gate — both confs required
    gate_passes = bool(latest["conf_01_drawdown_depth"]) and bool(latest["conf_02_off_lows"])

    # % off 52w high/low for display
    high_252 = float(latest.get("high_252", np.nan))
    low_252 = float(latest.get("low_252", np.nan))
    price = float(latest["Close"])
    pct_off_52w_high = (
        round((price / high_252 - 1.0) * 100, 1) if high_252 and not np.isnan(high_252) else None
    )
    pct_off_52w_low = (
        round((price / low_252 - 1.0) * 100, 1) if low_252 and not np.isnan(low_252) else None
    )

    # ── Signal determination ──
    if not gate_passes:
        signal = "BOTTOM -- N/A"
        action = f"Not a bottoming candidate (off-high {pct_off_52w_high}%, off-low {pct_off_52w_low}%)."
    elif regime_id == CRASH_REGIME:
        signal = "BOTTOM -- AVOID"
        action = f"Valid setup ({confs}/12) but regime is Crash/Capitulation. Do not catch falling knives."
    elif confs >= 9 and regime_id in BULLISH_OR_NEUTRAL_REGIMES:
        signal = "BOTTOM -- BUY"
        action = (
            f"Strong bottoming entry. {confs}/12 confs. {regime_label}. "
            f"Price {pct_off_52w_high}% from 52w high, {pct_off_52w_low}% above 52w low."
        )
    elif confs >= min_confirmations:
        signal = "BOTTOM -- WATCH"
        action = f"Setup valid ({confs}/12) but regime is {regime_label}. Watch for confirmation."
    else:
        signal = "BOTTOM -- WAIT"
        action = f"{confs}/12 confirmations in {regime_label}. Not enough for bottoming entry."

    # Confirmation breakdown
    conf_cols = [c for c in data.columns if c.startswith("conf_")]
    conf_detail = {}
    for c in conf_cols:
        name = c.replace("conf_", "").replace("_", " ").title()
        parts = name.split(" ", 1)
        if len(parts) > 1 and parts[0].isdigit():
            name = parts[1]
        conf_detail[name] = bool(latest[c])

    return {
        "signal": signal,
        "action": action,
        "regime_id": regime_id,
        "regime_label": regime_label,
        "confidence": confidence,
        "confirmations_met": confs,
        "confirmations_required": min_confirmations,
        "confirmations_total": 12,
        "confirmation_detail": conf_detail,
        "price": price,
        "pct_off_52w_high": pct_off_52w_high,
        "pct_off_52w_low": pct_off_52w_low,
        "regime_streak": streak,
        "regime_confirm_bars": regime_confirm_bars,
        "regime_confirmed": streak >= regime_confirm_bars,
        "prev_regime": str(prev["regime_label"]) if "regime_label" in prev else regime_label,
        "regime_changed": int(prev["regime_id"]) != regime_id if "regime_id" in prev else False,
    }
```

- [ ] **Step 4: Run tests — all 8 should pass**

Run: `cd ~/rrjcar-regime-scanner && pytest tests/test_strategy_bottoming.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
cd ~/rrjcar-regime-scanner
git add strategy_bottoming.py tests/test_strategy_bottoming.py
git commit -m "feat(bottoming): add get_current_signal_bottoming with BUY/WATCH/N/A/AVOID tiers"
```

---

## Task 4: `RECOMMENDED_SETTINGS` across all 4 strategies

**Files:**
- Modify: `strategy_bottoming.py` (append)
- Modify: `backtester.py` (append)
- Modify: `strategy_v2.py` (append)
- Modify: `strategy_leaps.py` (append)

- [ ] **Step 1: Append to `strategy_bottoming.py`**

```python


RECOMMENDED_SETTINGS = {
    "min_confs": 8,
    "regime_confirm": 2,
    "cooldown": 5,
    "min_dte": 30,
    "max_dte": 60,
    "min_avg_volume": 500_000,
    "min_price": 10,
    "max_price": None,
    "price_above_ema50": False,  # implicit via conf_10
    "ema10_above_20": False,     # implicit via conf_11
}
```

- [ ] **Step 2: Append to `backtester.py`**

```python


RECOMMENDED_SETTINGS = {
    "min_confs": 7,
    "regime_confirm": 2,
    "cooldown": 48,
    "min_dte": 14,
    "max_dte": 45,
    "min_avg_volume": 1_000_000,
    "min_price": 5,
    "max_price": None,
    "price_above_ema50": True,
    "ema10_above_20": False,
}
```

- [ ] **Step 3: Append to `strategy_v2.py`**

```python


RECOMMENDED_SETTINGS = {
    "min_confs": 8,
    "regime_confirm": 2,
    "cooldown": 3,
    "min_dte": 30,
    "max_dte": 60,
    "min_avg_volume": 500_000,
    "min_price": 10,
    "max_price": 1000,
    "price_above_ema50": False,
    "ema10_above_20": False,
}
```

- [ ] **Step 4: Append to `strategy_leaps.py`**

```python


RECOMMENDED_SETTINGS = {
    "min_confs": 7,
    "regime_confirm": 3,
    "cooldown": 10,
    "min_dte": 270,
    "max_dte": 540,
    "min_avg_volume": 1_000_000,
    "min_price": 20,
    "max_price": None,
    "price_above_ema50": True,
    "ema10_above_20": False,
}
```

- [ ] **Step 5: Verify import works**

Run: `cd ~/rrjcar-regime-scanner && python -c "from strategy_bottoming import RECOMMENDED_SETTINGS as B; from backtester import RECOMMENDED_SETTINGS as V1; from strategy_v2 import RECOMMENDED_SETTINGS as V2; from strategy_leaps import RECOMMENDED_SETTINGS as L; print(V1, V2, L, B)"`

Expected: four dicts printed, no errors.

- [ ] **Step 6: Commit**

```bash
cd ~/rrjcar-regime-scanner
git add strategy_bottoming.py backtester.py strategy_v2.py strategy_leaps.py
git commit -m "feat: add RECOMMENDED_SETTINGS constants to all strategy modules"
```

---

## Task 5: `settings_manager.py` — extend `DEFAULT_SETTINGS`

**Files:**
- Modify: `settings_manager.py`

- [ ] **Step 1: Read current `DEFAULT_SETTINGS` block**

Run: `cd ~/rrjcar-regime-scanner && grep -n "DEFAULT_SETTINGS" settings_manager.py`
Confirm it's a top-level dict starting around line 11.

- [ ] **Step 2: Add new keys inside `DEFAULT_SETTINGS`**

Use an edit to add the following keys before the closing `}` of `DEFAULT_SETTINGS`:

```python
    # New filter settings (previously frontend-only, now persisted)
    "min_avg_volume": 500_000,
    "min_price": 1,
    "max_price": None,
    "price_above_ema50": False,
    "ema10_above_20": False,
```

- [ ] **Step 3: Verify load/save round-trip preserves new keys**

Run:
```bash
cd ~/rrjcar-regime-scanner && python -c "
from settings_manager import DEFAULT_SETTINGS, load_settings
for k in ['min_avg_volume', 'min_price', 'max_price', 'price_above_ema50', 'ema10_above_20']:
    assert k in DEFAULT_SETTINGS, f'missing {k}'
print('OK:', {k: DEFAULT_SETTINGS[k] for k in ['min_avg_volume', 'min_price', 'max_price', 'price_above_ema50', 'ema10_above_20']})
"
```
Expected: `OK: {...}` with the five new keys.

- [ ] **Step 4: Commit**

```bash
cd ~/rrjcar-regime-scanner
git add settings_manager.py
git commit -m "feat(settings): extend DEFAULT_SETTINGS with filter-panel fields"
```

---

## Task 6: `api/routes_settings.py` — Pydantic model extension + new endpoint

**Files:**
- Modify: `api/routes_settings.py`
- Create: `tests/test_api_strategy_defaults.py`

- [ ] **Step 1: Write failing test for the endpoint**

Create `tests/test_api_strategy_defaults.py`:

```python
"""Tests for /api/strategy-defaults endpoint."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app import app
    return TestClient(app)


def test_strategy_defaults_returns_all_four_strategies(client):
    res = client.get("/api/strategy-defaults")
    assert res.status_code == 200
    body = res.json()
    assert set(body.keys()) == {"v1", "v2", "leaps", "bottoming"}


def test_each_strategy_has_required_default_keys(client):
    res = client.get("/api/strategy-defaults")
    body = res.json()
    required_keys = {
        "min_confs", "regime_confirm", "cooldown",
        "min_dte", "max_dte",
        "min_avg_volume", "min_price", "max_price",
        "price_above_ema50", "ema10_above_20",
    }
    for strategy, defaults in body.items():
        missing = required_keys - set(defaults.keys())
        assert not missing, f"{strategy} missing keys: {missing}"
```

- [ ] **Step 2: Run test — confirm failure**

Run: `cd ~/rrjcar-regime-scanner && pytest tests/test_api_strategy_defaults.py -v`
Expected: 404 Not Found on the endpoint.

- [ ] **Step 3: Extend `SettingsUpdate` Pydantic model**

In `api/routes_settings.py`, inside the `class SettingsUpdate(BaseModel):` block, add the following fields (after the existing fields, before the class ends):

```python
    min_avg_volume: Optional[int] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    price_above_ema50: Optional[bool] = None
    ema10_above_20: Optional[bool] = None
```

- [ ] **Step 4: Add the new endpoint**

Append to `api/routes_settings.py` (below `update_settings`):

```python


@router.get("/strategy-defaults")
async def get_strategy_defaults():
    """Return per-strategy recommended default settings for all strategies."""
    from backtester import RECOMMENDED_SETTINGS as V1_DEFAULTS
    from strategy_v2 import RECOMMENDED_SETTINGS as V2_DEFAULTS
    from strategy_leaps import RECOMMENDED_SETTINGS as LEAPS_DEFAULTS
    from strategy_bottoming import RECOMMENDED_SETTINGS as BOTTOMING_DEFAULTS
    return {
        "v1": V1_DEFAULTS,
        "v2": V2_DEFAULTS,
        "leaps": LEAPS_DEFAULTS,
        "bottoming": BOTTOMING_DEFAULTS,
    }
```

- [ ] **Step 5: Run test — confirm pass**

Run: `cd ~/rrjcar-regime-scanner && pytest tests/test_api_strategy_defaults.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
cd ~/rrjcar-regime-scanner
git add api/routes_settings.py tests/test_api_strategy_defaults.py
git commit -m "feat(api): add GET /api/strategy-defaults and extend SettingsUpdate"
```

---

## Task 7: `screener.py` — dispatch + ValueError

**Files:**
- Modify: `screener.py`

- [ ] **Step 1: Add the bottoming import**

Near the other strategy imports (line ~19), add:

```python
from strategy_bottoming import get_current_signal_bottoming
```

- [ ] **Step 2: Replace the dispatch block**

Locate the dispatch block around lines 323–332:

```python
        # Current signal with confirmations (V1, V2, or LEAPS)
        if strategy == "leaps":
            # LEAPS has 10 confs — use 5 as default for broader results
            leaps_min = min(min_confirmations, 5)
            signal_data = get_current_signal_leaps(regime_df, min_confirmations=leaps_min, regime_confirm_bars=regime_confirm_bars)
        elif strategy == "v2":
            signal_data = get_current_signal_v2(regime_df, min_confirmations=min_confirmations, regime_confirm_bars=regime_confirm_bars)
        else:
            signal_data = get_current_signal(regime_df, min_confirmations=min_confirmations, regime_confirm_bars=regime_confirm_bars)
```

Replace with:

```python
        # Current signal with confirmations (V1, V2, LEAPS, or Bottoming)
        if strategy == "leaps":
            # LEAPS has 10 confs — use 5 as default for broader results
            leaps_min = min(min_confirmations, 5)
            signal_data = get_current_signal_leaps(regime_df, min_confirmations=leaps_min, regime_confirm_bars=regime_confirm_bars)
        elif strategy == "v2":
            signal_data = get_current_signal_v2(regime_df, min_confirmations=min_confirmations, regime_confirm_bars=regime_confirm_bars)
        elif strategy == "bottoming":
            signal_data = get_current_signal_bottoming(regime_df, min_confirmations=min_confirmations, regime_confirm_bars=regime_confirm_bars)
        elif strategy == "v1":
            signal_data = get_current_signal(regime_df, min_confirmations=min_confirmations, regime_confirm_bars=regime_confirm_bars)
        else:
            raise ValueError(f"unknown strategy: {strategy!r} (expected one of: v1, v2, leaps, bottoming)")
```

- [ ] **Step 3: Verify imports resolve**

Run: `cd ~/rrjcar-regime-scanner && python -c "import screener; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Smoke-test dispatch error path**

Run:
```bash
cd ~/rrjcar-regime-scanner && python -c "
from screener import _scan_single_ticker
try:
    _scan_single_ticker('SPY', strategy='nonexistent_strategy_xyz')
    print('FAIL: should have raised')
except ValueError as e:
    print('OK:', e)
except Exception as e:
    # May raise earlier (data-fetch issues); that's acceptable — dispatch test isn't strictly required online
    print('Ran without reaching dispatch:', type(e).__name__)
"
```
Expected: `OK: unknown strategy: 'nonexistent_strategy_xyz' (...)` — or, if the fetch fails first, you'll at least confirm the import works.

- [ ] **Step 5: Commit**

```bash
cd ~/rrjcar-regime-scanner
git add screener.py
git commit -m "feat(screener): dispatch bottoming strategy; raise ValueError on unknown"
```

---

## Task 8: `index.html` — strategy option, button, signal filters

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Add the Bottoming option to the strategy dropdown**

Locate the `<select id="setting-strategy">` block around line 47. Change it from:

```html
<select id="setting-strategy">
    <option value="v2">V2 (12-conf)</option>
    <option value="v1">V1 (8-conf)</option>
    <option value="leaps">LEAPS (10-conf)</option>
</select>
```

To:

```html
<select id="setting-strategy">
    <option value="v2">V2 (12-conf)</option>
    <option value="v1">V1 (8-conf)</option>
    <option value="leaps">LEAPS (10-conf)</option>
    <option value="bottoming">Bottoming (12-conf)</option>
</select>
```

- [ ] **Step 2: Add the "Load recommended defaults" button**

Immediately after the `</select>` closing tag of `#setting-strategy` (still inside the same `<div class="setting-group">`), add:

```html
<button type="button"
        id="btn-load-strategy-defaults"
        class="btn-load-defaults"
        style="margin-top:0.4rem; padding:0.3rem 0.6rem; font-size:0.85rem; background:#1e2028; border:1px solid #2dd4bf; color:#2dd4bf; cursor:pointer; border-radius:4px;"
        title="Overwrite current settings with the recommended defaults for the selected strategy"
        onclick="Settings.loadRecommendedDefaults()">
    &#x21bb; Load recommended defaults
</button>
```

- [ ] **Step 3: Add Bottoming signal filter options**

Locate the `<select id="filter-signal" ...>` block around line 155. Add two new options at the end, before `</select>`:

```html
<option value="BOTTOM -- BUY">BOTTOM BUY</option>
<option value="BOTTOM -- WATCH">BOTTOM WATCH</option>
```

- [ ] **Step 4: Verify HTML parses and button appears**

```bash
cd ~/rrjcar-regime-scanner && python -c "
from html.parser import HTMLParser
class P(HTMLParser):
    def __init__(self): super().__init__(); self.found = {'btn': False, 'option': False, 'filter': 0}
    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if d.get('id') == 'btn-load-strategy-defaults': self.found['btn'] = True
        if d.get('value') == 'bottoming': self.found['option'] = True
        if d.get('value') in ('BOTTOM -- BUY', 'BOTTOM -- WATCH'): self.found['filter'] += 1
p = P()
p.feed(open('index.html').read())
assert p.found['btn'], 'button missing'
assert p.found['option'], 'bottoming option missing'
assert p.found['filter'] == 2, f'expected 2 filter options, got {p.found[\"filter\"]}'
print('OK')
"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
cd ~/rrjcar-regime-scanner
git add index.html
git commit -m "feat(ui): add Bottoming strategy option, defaults button, signal filters"
```

---

## Task 9: `js/api.js` — add `getStrategyDefaults`

**Files:**
- Modify: `js/api.js`

- [ ] **Step 1: Add `getStrategyDefaults` method to the `API` object literal**

`js/api.js` is an object literal of the form `const API = { method1() {...}, method2() {...}, ... };` with a shared `this.get(url)` wrapper that throws on non-2xx responses. Add this method next to `getSettings` / `saveSettings`:

```javascript
    async getStrategyDefaults() {
        if (this._strategyDefaultsCache) return this._strategyDefaultsCache;
        this._strategyDefaultsCache = await this.get('/api/strategy-defaults');
        return this._strategyDefaultsCache;
    },
```

The session cache (`_strategyDefaultsCache`) ensures the endpoint is hit only once per page load.

- [ ] **Step 2: Smoke-test (manual)**

Start the server:
```bash
cd ~/rrjcar-regime-scanner
source venv/bin/activate
uvicorn app:app --reload
```

In a second terminal:
```bash
curl -s http://localhost:8000/api/strategy-defaults | python -m json.tool
```
Expected: JSON with four keys (`v1`, `v2`, `leaps`, `bottoming`), each with recommended settings.

- [ ] **Step 3: Commit**

```bash
cd ~/rrjcar-regime-scanner
git add js/api.js
git commit -m "feat(api.js): add getStrategyDefaults with session cache"
```

---

## Task 10: `js/settings.js` — `loadRecommendedDefaults` + `flashButton`

**Files:**
- Modify: `js/settings.js`

- [ ] **Step 1: Add `loadRecommendedDefaults` and `flashButton` methods**

Add inside the `Settings` object literal (after `saveAndConfirm`):

```javascript
    async loadRecommendedDefaults() {
        const strategy = this.getVal('setting-strategy');
        if (!strategy) return;
        try {
            const allDefaults = await API.getStrategyDefaults();
            const defaults = allDefaults[strategy];
            if (!defaults) {
                console.warn('No recommended defaults for strategy:', strategy);
                return;
            }

            const FIELD_MAP = {
                min_confs:       'setting-min-confs',
                regime_confirm:  'setting-regime-confirm',
                cooldown:        'setting-cooldown',
                min_dte:         'setting-min-dte',
                max_dte:         'setting-max-dte',
                min_avg_volume:  'filter-min-volume',
                min_price:       'filter-min-price',
                max_price:       'filter-max-price',
            };
            const TOGGLE_MAP = {
                price_above_ema50: 'filter-price-above-ema50',
                ema10_above_20:    'filter-ema10-above-20',
            };

            Object.entries(FIELD_MAP).forEach(([key, id]) => {
                if (!(key in defaults)) return;
                const val = defaults[key];
                this.setVal(id, val === null ? '' : val);
            });

            Object.entries(TOGGLE_MAP).forEach(([key, id]) => {
                if (!(key in defaults)) return;
                const el = document.getElementById(id);
                if (el) el.checked = !!defaults[key];
            });

            // Re-run client-side filter pipeline manually
            // (filter-* fields bypass their onchange handlers when set programmatically)
            if (window.Screener && typeof Screener.render === 'function') {
                Screener.minPrice        = parseFloat(this.getVal('filter-min-price')) || 0;
                Screener.maxPrice        = parseFloat(this.getVal('filter-max-price')) || 0;
                Screener.minVolume       = parseFloat(this.getVal('filter-min-volume')) || 0;
                const emaEl = document.getElementById('filter-price-above-ema50');
                const stackEl = document.getElementById('filter-ema10-above-20');
                Screener.priceAboveEma50 = emaEl ? emaEl.checked : false;
                Screener.ema10Above20    = stackEl ? stackEl.checked : false;
                Screener.render(Screener.results, document.getElementById('screener-content'));
            }

            this.flashButton('btn-load-strategy-defaults', 'Loaded');
        } catch (err) {
            console.error('Load recommended defaults failed:', err);
        }
    },

    flashButton(id, msg) {
        const btn = document.getElementById(id);
        if (!btn) return;
        const orig = btn.textContent;
        btn.textContent = msg;
        setTimeout(() => { btn.textContent = orig; }, 1500);
    },
```

- [ ] **Step 2: Bump cache-busters in `index.html`**

`index.html` includes JS files with `?v=120` cache-busters. Bump both to `?v=121` so browsers re-fetch the updated files:

```bash
cd ~/rrjcar-regime-scanner
sed -i '' 's|api\.js?v=120|api.js?v=121|' index.html
sed -i '' 's|settings\.js?v=120|settings.js?v=121|' index.html
grep -n "v=12" index.html | head -5
```

Expected: at least two lines showing `?v=121`.

Note: if the version has already advanced past 120 (other changes merged in the interim), use the current version + 1.

- [ ] **Step 3: Manual smoke test**

1. Start server: `uvicorn app:app --reload`
2. Open browser to `http://localhost:8000/`
3. Open DevTools console. Check for JS errors on page load.
4. Select "V1 (8-conf)" from Strategy dropdown.
5. Click "↻ Load recommended defaults".
6. Verify: Min Confs shows 7, Cooldown shows 48, Min DTE 14, Max DTE 45, Min Avg Volume 1000000, Min Price 5, Price > 50 EMA toggle is ON.
7. Repeat for V2, LEAPS, Bottoming — confirm each loads their Task 4 values.

- [ ] **Step 4: Commit**

```bash
cd ~/rrjcar-regime-scanner
git add js/settings.js index.html
git commit -m "feat(settings.js): add loadRecommendedDefaults and flashButton"
```

---

## Task 11: End-to-end manual smoke test

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

```bash
cd ~/rrjcar-regime-scanner && pytest -v
```
Expected: all tests pass (at minimum: 8 bottoming + 2 API = 10).

- [ ] **Step 2: Start server**

```bash
cd ~/rrjcar-regime-scanner
source venv/bin/activate
uvicorn app:app --reload
```

- [ ] **Step 3: Manual scenarios (through the browser)**

1. **V1 regression** — Strategy=V1, Scan → verify results still return the expected V1 signals (ENTER/CONFIRMING/etc.).
2. **V2 regression** — Strategy=V2, Scan → same, verify V2 signal labels.
3. **LEAPS regression** — Strategy=LEAPS, Scan → same, verify LEAPS signal labels.
4. **Bottoming — BUY candidate** — Type a known bottoming candidate (a stock that was ≥35% off highs and has been basing for ~20 days, e.g. pick one from your recent LEAP scanner output where 52w filter was >40%). Strategy=Bottoming, custom tickers=<that symbol>, Scan. Expected: `BOTTOM -- BUY` or `BOTTOM -- WATCH`.
5. **Bottoming — N/A** — Custom tickers=`NVDA` (or any recent high-flyer), Strategy=Bottoming, Scan. Expected: `BOTTOM -- N/A` (filtered out of results).
6. **Bottoming — AVOID** — Not easily testable without a crash-regime ticker; verify through unit tests only.
7. **Defaults button for every strategy** — Select each of the four strategies from the dropdown, click "Load recommended defaults", confirm fields populate correctly and toggles flip where expected.

- [ ] **Step 4: Save and reload**

1. Select Bottoming, click Load recommended defaults.
2. Click Save Settings.
3. Hard-refresh the browser (Cmd+Shift+R).
4. Verify fields persist (loaded from `.dashboard_settings.json`).

- [ ] **Step 5: (Optional) Push to origin**

```bash
cd ~/rrjcar-regime-scanner
git log --oneline -15   # sanity check commit list
git push origin main
```

---

## Summary of commits produced

1. `test: add pytest infrastructure with synthetic OHLCV fixtures`
2. `feat(bottoming): add compute_bottoming_confirmations with 12 signals`
3. `feat(bottoming): add get_current_signal_bottoming with BUY/WATCH/N/A/AVOID tiers`
4. `feat: add RECOMMENDED_SETTINGS constants to all strategy modules`
5. `feat(settings): extend DEFAULT_SETTINGS with filter-panel fields`
6. `feat(api): add GET /api/strategy-defaults and extend SettingsUpdate`
7. `feat(screener): dispatch bottoming strategy; raise ValueError on unknown`
8. `feat(ui): add Bottoming strategy option, defaults button, signal filters`
9. `feat(api.js): add getStrategyDefaults with session cache`
10. `feat(settings.js): add loadRecommendedDefaults and flashButton`

11 tasks, 10 commits, ~3–4 hours total implementation time for an engineer familiar with Python + FastAPI + vanilla JS.
