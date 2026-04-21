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


RECOMMENDED_SETTINGS = {
    "min_confs": 8,
    "regime_confirm": 2,
    "cooldown": 5,
    "min_dte": 30,
    "max_dte": 60,
    "min_avg_volume": 500_000,
    "min_price": 1,
    "max_price": None,
    "price_above_ema50": False,  # implicit via conf_10
    "ema10_above_20": False,     # implicit via conf_11
}
