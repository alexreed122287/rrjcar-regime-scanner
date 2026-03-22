"""
strategy_v2.py — Long-Call Optimized Strategy Engine
Built specifically for daily timeframe single-leg long call entries.

Key differences from V1:
  - 12 confirmations (6 kept from V1 refined, 6 new call-specific)
  - Multi-exit: ATR trailing stop, time stop, profit target, regime flip, RSI exit
  - VIX overlay: market-wide fear filter
  - HV rank filter: avoid buying expensive calls
  - Pullback/dip-buy logic instead of momentum-chasing
"""

import numpy as np
import pandas as pd
import ta
import yfinance as yf
from datetime import datetime, timedelta


# ═══════════════════════════════════════════
#  VIX DATA
# ═══════════════════════════════════════════

def fetch_vix(start_date: str = None, end_date: str = None) -> pd.Series:
    """Fetch VIX close prices aligned to trading days."""
    try:
        if start_date and end_date:
            vix = yf.Ticker("^VIX").history(start=start_date, end=end_date, interval="1d")
        else:
            vix = yf.Ticker("^VIX").history(period="2y", interval="1d")

        if vix is not None and not vix.empty:
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            return vix["Close"]
    except Exception:
        pass
    return pd.Series(dtype=float)


# ═══════════════════════════════════════════
#  ENHANCED CONFIRMATIONS (12 signals)
# ═══════════════════════════════════════════

def compute_confirmations_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    12 confirmation signals optimized for long call entries on daily bars.

    Kept from V1 (refined):
      1. RSI 30-70 sweet spot (not overbought or deeply oversold)
      2. Price above 20 EMA (short-term trend)
      3. Price above 50 EMA (medium-term trend)
      4. MACD histogram positive
      5. ADX > 20 (trending market)
      6. Volume above 20-period average

    New for calls:
      7. RSI pullback (RSI was > 50 in last 5 bars — dip-buy setup)
      8. Momentum acceleration (5-bar ROC > 10-bar ROC)
      9. ATR percentile < 70% (not buying at volatility extremes)
     10. Price within 5% of 20 EMA (not too extended)
     11. Breakout day (close > prior day high)
     12. Higher low (current low > low from 3 bars ago)
    """
    out = df.copy()

    # ── Core indicators ──
    out["rsi"] = ta.momentum.RSIIndicator(out["Close"], window=14).rsi()

    atr_ind = ta.volatility.AverageTrueRange(out["High"], out["Low"], out["Close"], window=14)
    out["atr"] = atr_ind.average_true_range()

    adx_ind = ta.trend.ADXIndicator(out["High"], out["Low"], out["Close"], window=14)
    out["adx"] = adx_ind.adx()

    out["ema_20"] = ta.trend.EMAIndicator(out["Close"], window=20).ema_indicator()
    out["ema_50"] = ta.trend.EMAIndicator(out["Close"], window=50).ema_indicator()

    macd_ind = ta.trend.MACD(out["Close"], window_slow=26, window_fast=12, window_sign=9)
    out["macd_hist"] = macd_ind.macd_diff()

    out["vol_ma_20"] = out["Volume"].rolling(20, min_periods=1).mean()

    # ── ROC (rate of change) ──
    out["roc_5"] = out["Close"].pct_change(5) * 100
    out["roc_10"] = out["Close"].pct_change(10) * 100

    # ── ATR percentile (rolling 60-day window) ──
    out["atr_pctl"] = out["atr"].rolling(60, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # ── Historical volatility rank ──
    out["hv_20"] = out["Close"].pct_change().rolling(20).std() * np.sqrt(252)
    out["hv_rank"] = out["hv_20"].rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # ── RSI rolling max (was RSI > 50 recently?) ──
    out["rsi_max_5"] = out["rsi"].rolling(5, min_periods=1).max()

    # ── Distance from 20 EMA ──
    out["ema20_dist_pct"] = (out["Close"] - out["ema_20"]) / out["ema_20"] * 100

    # ═══ 12 CONFIRMATION SIGNALS ═══

    # 1. RSI in sweet spot (30-70) — not overbought, not deeply oversold
    out["conf_01_rsi_sweet"] = (out["rsi"] >= 30) & (out["rsi"] <= 70)

    # 2. Price above 20 EMA (short-term trend up)
    out["conf_02_above_ema20"] = out["Close"] > out["ema_20"]

    # 3. Price above 50 EMA (medium-term trend up)
    out["conf_03_above_ema50"] = out["Close"] > out["ema_50"]

    # 4. MACD histogram positive (bullish momentum)
    out["conf_04_macd_bull"] = out["macd_hist"] > 0

    # 5. ADX > 20 (market is trending, not ranging)
    out["conf_05_adx_trend"] = out["adx"] > 20

    # 6. Volume above 20-period average (institutional participation)
    out["conf_06_vol_confirm"] = out["Volume"] > out["vol_ma_20"]

    # 7. RSI pullback — RSI hit > 50 in last 5 bars (was bullish, now dipping = buy opportunity)
    out["conf_07_rsi_pullback"] = out["rsi_max_5"] > 50

    # 8. Momentum acceleration — 5-bar ROC > 10-bar ROC (speeding up, not fading)
    out["conf_08_momo_accel"] = out["roc_5"] > out["roc_10"]

    # 9. ATR percentile < 70% (not in volatility extreme — calls cheaper)
    out["conf_09_atr_ok"] = out["atr_pctl"] < 0.70

    # 10. Price within 5% of 20 EMA (not too extended — better call entry)
    out["conf_10_not_extended"] = out["ema20_dist_pct"].abs() < 5.0

    # 11. Breakout day — close > prior day's high
    out["conf_11_breakout"] = out["Close"] > out["High"].shift(1)

    # 12. Higher low — current low > low from 3 bars ago (uptrend structure)
    out["conf_12_higher_low"] = out["Low"] > out["Low"].shift(3)

    # ── Aggregate ──
    conf_cols = [c for c in out.columns if c.startswith("conf_")]
    out["confirmations_met"] = out[conf_cols].sum(axis=1).astype(int)

    return out


# ═══════════════════════════════════════════
#  BACKTEST ENGINE V2
# ═══════════════════════════════════════════

def run_backtest_v2(
    df: pd.DataFrame,
    min_confirmations: int = 6,
    cooldown_bars: int = 3,
    regime_confirm_bars: int = 2,
    bullish_regimes: list = None,
    bearish_regimes: list = None,
    initial_capital: float = 100_000.0,
    # ── Call-specific exits ──
    atr_trail_mult: float = 2.0,
    time_stop_bars: int = 15,
    profit_target_pct: float = 10.0,
    rsi_exit_threshold: float = 75.0,
    min_gain_for_rsi_exit: float = 8.0,
    # ── VIX overlay ──
    vix_caution: float = 25.0,
    vix_halt: float = 35.0,
    vix_caution_confs: int = 8,
    # ── HV rank filter ──
    hv_rank_max: float = 0.75,
    hv_rank_caution: float = 0.50,
    hv_caution_confs: int = 8,
) -> dict:
    """
    Long-call optimized backtest with multi-exit logic, VIX overlay, and HV rank filter.
    """
    if bullish_regimes is None:
        bullish_regimes = [0, 1, 2]
    if bearish_regimes is None:
        bearish_regimes = [5, 6]

    # Compute enhanced confirmations
    data = compute_confirmations_v2(df)
    data = data.dropna().copy()

    # Pre-compute regime streak
    regime_ids = data["regime_id"].values.astype(int)
    regime_streak = np.zeros(len(regime_ids), dtype=int)
    regime_streak[0] = 1
    for j in range(1, len(regime_ids)):
        if regime_ids[j] == regime_ids[j - 1]:
            regime_streak[j] = regime_streak[j - 1] + 1
        else:
            regime_streak[j] = 1
    data["regime_streak"] = regime_streak

    # Fetch VIX data
    try:
        start_str = str(data.index[0].date()) if hasattr(data.index[0], 'date') else str(data.index[0])[:10]
        end_str = str(data.index[-1].date()) if hasattr(data.index[-1], 'date') else str(data.index[-1])[:10]
        vix = fetch_vix(start_str, end_str)
        # Align VIX to data index
        if not vix.empty:
            vix.index = vix.index.tz_localize(None) if vix.index.tz else vix.index
            data_idx = data.index.tz_localize(None) if data.index.tz else data.index
            vix_aligned = vix.reindex(data_idx, method="ffill")
            data["vix"] = vix_aligned.values
        else:
            data["vix"] = 20.0
    except Exception:
        data["vix"] = 20.0

    # State tracking
    position_open = False
    entry_price = 0.0
    entry_bar = 0
    entry_regime = None
    highest_since_entry = 0.0
    cooldown_remaining = 0
    capital = initial_capital
    roll_count = 0
    roll_credits_pct = 0.0
    last_roll_bar = 0
    effective_entry = 0.0  # adjusted entry after rolls

    trades = []
    equity = [initial_capital]
    signals = []

    for i in range(1, len(data)):
        row = data.iloc[i]
        prev = data.iloc[i - 1]
        regime = int(row["regime_id"])
        confs = int(row["confirmations_met"])
        price = float(row["Close"])
        prev_price = float(prev["Close"])
        streak = int(row["regime_streak"])
        current_atr = float(row["atr"]) if pd.notna(row["atr"]) else 0
        current_rsi = float(row["rsi"]) if pd.notna(row["rsi"]) else 50
        current_vix = float(row["vix"]) if pd.notna(row.get("vix", 20)) else 20
        current_hv_rank = float(row["hv_rank"]) if pd.notna(row.get("hv_rank")) else 0.5

        # Track unrealized PnL
        if position_open:
            bar_return = (price - prev_price) / prev_price
            capital *= (1 + bar_return)
            highest_since_entry = max(highest_since_entry, price)

        # ── ROLL CHECK (before exit logic) ──
        if position_open and current_atr > 0:
            bars_since_roll = i - last_roll_bar if last_roll_bar > 0 else i - entry_bar

            # Roll UP: price >= effective_entry + 1 ATR (and still in bullish regime)
            if (price >= effective_entry + current_atr
                and regime in bullish_regimes
                and roll_count < 3
                and bars_since_roll >= 2):
                # Simulate roll: collect ~0.5% credit, move effective entry up
                roll_credit = 0.5  # % of stock price
                roll_credits_pct += roll_credit
                capital *= (1 + roll_credit / 100)  # add credit to capital
                effective_entry = price  # reset entry to current price after roll
                roll_count += 1
                last_roll_bar = i
                signals.append("ROLL_UP")
                equity.append(capital)
                continue

        # ── EXIT LOGIC (multi-trigger) ──
        if position_open:
            exit_reason = None
            bars_held = i - entry_bar
            gain_pct = (price - entry_price) / entry_price * 100

            # 1. Regime flip to bearish (immediate, no roll possible)
            if regime in bearish_regimes:
                exit_reason = f"Regime flip - {row['regime_label']}"

            # 2. ATR trailing stop from peak
            elif current_atr > 0 and highest_since_entry > 0:
                trail_stop = highest_since_entry - (atr_trail_mult * current_atr)
                if price < trail_stop and bars_held >= 3:
                    exit_reason = f"ATR trail stop (peak ${highest_since_entry:.2f})"

            # 3. Time stop — but try roll-out first (simulated DTE check)
            if not exit_reason and bars_held >= time_stop_bars and gain_pct <= 0:
                # Simulate: can we roll out for credit?
                if regime in bullish_regimes and roll_count < 3:
                    roll_credit = 0.3
                    roll_credits_pct += roll_credit
                    capital *= (1 + roll_credit / 100)
                    roll_count += 1
                    last_roll_bar = i
                    signals.append("ROLL_OUT")
                    equity.append(capital)
                    continue
                else:
                    exit_reason = f"Time stop, can't roll ({bars_held} bars, {gain_pct:+.1f}%)"

            # 4. RSI overbought exit (take profits at extremes)
            if not exit_reason and current_rsi > rsi_exit_threshold and gain_pct > min_gain_for_rsi_exit:
                exit_reason = f"RSI exit ({current_rsi:.0f}, +{gain_pct:.1f}%)"

            # 5. Neutral regime
            if not exit_reason and regime not in bullish_regimes and regime not in bearish_regimes:
                if float(row["regime_confidence"]) > 0.6:
                    exit_reason = f"Regime neutral - {row['regime_label']}"

            if exit_reason:
                pnl_pct = gain_pct
                trades.append({
                    "entry_bar": entry_bar,
                    "entry_date": str(data.index[entry_bar]),
                    "entry_price": entry_price,
                    "entry_regime": entry_regime,
                    "exit_bar": i,
                    "exit_date": str(data.index[i]),
                    "exit_price": price,
                    "exit_regime": row["regime_label"],
                    "exit_reason": exit_reason,
                    "pnl_pct": round(pnl_pct, 2),
                    "bars_held": bars_held,
                    "peak_price": round(highest_since_entry, 2),
                    "peak_gain_pct": round((highest_since_entry - entry_price) / entry_price * 100, 2),
                    "confirmations_at_entry": int(data.iloc[entry_bar]["confirmations_met"]),
                    "roll_count": roll_count,
                    "roll_credits_pct": round(roll_credits_pct, 2),
                })
                position_open = False
                cooldown_remaining = cooldown_bars
                roll_count = 0
                roll_credits_pct = 0.0
                last_roll_bar = 0
                effective_entry = 0.0
                signals.append("EXIT")
                equity.append(capital)
                continue

        # ── COOLDOWN ──
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            signals.append("COOLDOWN")
            equity.append(capital)
            continue

        # ── ENTRY LOGIC (with VIX overlay + HV rank + regime confirmation) ──
        if not position_open:
            is_bullish = regime in bullish_regimes
            high_confidence = float(row["regime_confidence"]) > 0.5
            regime_confirmed = streak >= regime_confirm_bars

            # Dynamic confirmation threshold based on VIX and HV rank
            required_confs = min_confirmations
            if current_vix > vix_halt:
                # VIX panic — no entries
                signals.append("VIX_HALT")
                equity.append(capital)
                continue
            elif current_vix > vix_caution:
                required_confs = max(required_confs, vix_caution_confs)

            if current_hv_rank > hv_rank_max:
                # IV too expensive — skip
                signals.append("IV_HIGH")
                equity.append(capital)
                continue
            elif current_hv_rank > hv_rank_caution:
                required_confs = max(required_confs, hv_caution_confs)

            enough_confs = confs >= required_confs

            if is_bullish and enough_confs and high_confidence and regime_confirmed:
                position_open = True
                entry_price = price
                effective_entry = price
                entry_bar = i
                entry_regime = row["regime_label"]
                highest_since_entry = price
                roll_count = 0
                roll_credits_pct = 0.0
                last_roll_bar = 0
                signals.append("ENTRY")
                equity.append(capital)
                continue
            elif is_bullish and not regime_confirmed:
                signals.append("CONFIRMING")
                equity.append(capital)
                continue

        # ── HOLD or WAIT ──
        if position_open:
            signals.append("HOLDING")
        else:
            signals.append("CASH")
        equity.append(capital)

    # Close open position at end
    if position_open:
        final_price = float(data.iloc[-1]["Close"])
        gain_pct = (final_price - entry_price) / entry_price * 100
        trades.append({
            "entry_bar": entry_bar,
            "entry_date": str(data.index[entry_bar]),
            "entry_price": entry_price,
            "entry_regime": entry_regime,
            "exit_bar": len(data) - 1,
            "exit_date": str(data.index[-1]),
            "exit_price": final_price,
            "exit_regime": data.iloc[-1]["regime_label"],
            "exit_reason": "End of backtest",
            "pnl_pct": round(gain_pct, 2),
            "bars_held": len(data) - 1 - entry_bar,
            "peak_price": round(highest_since_entry, 2),
            "peak_gain_pct": round((highest_since_entry - entry_price) / entry_price * 100, 2),
            "confirmations_at_entry": int(data.iloc[entry_bar]["confirmations_met"]),
            "roll_count": roll_count,
            "roll_credits_pct": round(roll_credits_pct, 2),
        })

    # Pad
    while len(signals) < len(data):
        signals.append("CASH")
    while len(equity) < len(data):
        equity.append(equity[-1] if equity else initial_capital)
    signals = signals[:len(data)]
    equity = equity[:len(data)]

    data["signal"] = signals
    data["equity"] = equity

    metrics = _compute_metrics_v2(trades, equity, data, initial_capital)
    return {"trades": trades, "equity_curve": pd.Series(equity, index=data.index), "metrics": metrics, "df": data}


def get_current_signal_v2(df: pd.DataFrame, min_confirmations: int = 6, regime_confirm_bars: int = 2) -> dict:
    """Get current V2 signal for live use."""
    data = compute_confirmations_v2(df)
    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest

    regime_id = int(latest["regime_id"])
    regime_label = latest["regime_label"]
    confidence = float(latest["regime_confidence"])
    confs = int(latest["confirmations_met"])
    prev_regime = int(prev["regime_id"])
    hv_rank = float(latest["hv_rank"]) if pd.notna(latest.get("hv_rank")) else 0.5

    # Compute streak
    regime_ids = data["regime_id"].values.astype(int)
    streak = 1
    for j in range(len(regime_ids) - 2, -1, -1):
        if regime_ids[j] == regime_ids[-1]:
            streak += 1
        else:
            break

    regime_confirmed = streak >= regime_confirm_bars
    is_bullish = regime_id in [0, 1, 2]
    was_bullish = prev_regime in [0, 1, 2]
    is_bearish = regime_id in [5, 6]

    # Signal determination
    if is_bullish and confs >= min_confirmations and regime_confirmed:
        if was_bullish:
            signal = "LONG -- HOLD"
            action = "Maintain position. Regime still bullish."
        else:
            signal = "LONG -- ENTER"
            action = f"Bullish regime confirmed ({streak} bars). {confs}/12 confirmations. Enter long call."
    elif is_bullish and confs >= min_confirmations and not regime_confirmed:
        signal = "LONG -- CONFIRMING"
        action = f"Bullish detected, confirming ({streak}/{regime_confirm_bars} bars). Prepare entry."
    elif is_bearish:
        if was_bullish:
            signal = "EXIT -- REGIME FLIP"
            action = "Regime flipped bearish. Close all calls immediately."
        else:
            signal = "CASH -- BEARISH"
            action = "Bearish regime. No calls."
    else:
        signal = "CASH -- NEUTRAL"
        action = f"Regime: {regime_label}. {confs}/12 confirmations. No trade."

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
        "price": float(latest["Close"]),
        "rsi": float(latest["rsi"]) if pd.notna(latest["rsi"]) else None,
        "adx": float(latest["adx"]) if pd.notna(latest["adx"]) else None,
        "macd_hist": float(latest["macd_hist"]) if pd.notna(latest["macd_hist"]) else None,
        "hv_rank": round(hv_rank, 2),
        "regime_changed": regime_id != prev_regime,
        "prev_regime": prev["regime_label"],
        "regime_streak": streak,
        "regime_confirm_bars": regime_confirm_bars,
        "regime_confirmed": regime_confirmed,
    }


def _compute_metrics_v2(trades, equity, df, initial_capital) -> dict:
    """Compute performance metrics with call-relevant additions."""
    if not trades:
        return {
            "total_return_pct": 0, "alpha_vs_buyhold": 0, "buyhold_return_pct": 0,
            "win_rate": 0, "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "max_drawdown_pct": 0, "sharpe_ratio": 0, "profit_factor": 0,
            "avg_win_pct": 0, "avg_loss_pct": 0, "avg_bars_held": 0,
            "avg_peak_gain": 0, "avg_giveback": 0,
            "final_equity": initial_capital, "initial_capital": initial_capital,
        }

    final_equity = equity[-1] if equity else initial_capital
    total_return = (final_equity - initial_capital) / initial_capital * 100

    bh_start = float(df["Close"].iloc[0])
    bh_end = float(df["Close"].iloc[-1])
    bh_return = (bh_end - bh_start) / bh_start * 100
    alpha = total_return - bh_return

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0

    gross_profit = sum(t["pnl_pct"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_pct"] for t in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    eq_series = pd.Series(equity)
    peak = eq_series.expanding().max()
    drawdown = (eq_series - peak) / peak * 100
    max_dd = drawdown.min()

    returns = eq_series.pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else 0

    # Call-specific metrics
    avg_bars = np.mean([t.get("bars_held", 0) for t in trades]) if trades else 0
    avg_peak = np.mean([t.get("peak_gain_pct", 0) for t in trades]) if trades else 0
    avg_giveback = np.mean([t.get("peak_gain_pct", 0) - t["pnl_pct"] for t in trades]) if trades else 0

    # Roll metrics
    total_rolls = sum(t.get("roll_count", 0) for t in trades)
    trades_with_rolls = sum(1 for t in trades if t.get("roll_count", 0) > 0)
    avg_rolls = total_rolls / len(trades) if trades else 0
    total_roll_credits = sum(t.get("roll_credits_pct", 0) for t in trades)

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        reason = t.get("exit_reason", "unknown").split("(")[0].strip()
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    return {
        "total_return_pct": round(total_return, 2),
        "alpha_vs_buyhold": round(alpha, 2),
        "buyhold_return_pct": round(bh_return, 2),
        "win_rate": round(win_rate, 1),
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "avg_bars_held": round(avg_bars, 1),
        "avg_peak_gain": round(avg_peak, 2),
        "avg_giveback": round(avg_giveback, 2),
        "total_rolls": total_rolls,
        "trades_with_rolls": trades_with_rolls,
        "avg_rolls_per_trade": round(avg_rolls, 1),
        "total_roll_credits_pct": round(total_roll_credits, 2),
        "exit_reasons": exit_reasons,
        "final_equity": round(final_equity, 2),
        "initial_capital": initial_capital,
    }
