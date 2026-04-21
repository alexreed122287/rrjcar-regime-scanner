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
