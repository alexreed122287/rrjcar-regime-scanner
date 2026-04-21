"""
backtester.py — Strategy Engine & Backtester
Layers confirmation-based entry/exit logic on top of HMM regime detection.

Entry: Only in bullish regimes + N-of-8 confirmation signals met.
Exit:  Immediately on regime flip to bearish.
Risk:  Cooldown period after exits, configurable leverage.
"""

import numpy as np
import pandas as pd
import ta


def compute_confirmations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 8 confirmation signals used for trade entry.
    Each is a boolean column (True = condition met).

    Confirmations:
    1. RSI < 90 (not overbought extreme)
    2. RSI > 25 (not deeply oversold — momentum present)
    3. Momentum positive (close > close 10 bars ago)
    4. Volatility expanding (ATR rising over 14-period lookback)
    5. Volume above 20-period average
    6. ADX > 20 (trending, not ranging)
    7. Price above 50-period EMA (trend filter)
    8. MACD histogram positive (bullish momentum)
    """
    out = df.copy()

    # RSI (14)
    out["rsi"] = ta.momentum.RSIIndicator(out["Close"], window=14).rsi()

    # ATR (14)
    atr = ta.volatility.AverageTrueRange(out["High"], out["Low"], out["Close"], window=14)
    out["atr"] = atr.average_true_range()
    out["atr_rising"] = out["atr"] > out["atr"].shift(1)

    # ADX (14)
    adx = ta.trend.ADXIndicator(out["High"], out["Low"], out["Close"], window=14)
    out["adx"] = adx.adx()

    # EMAs
    out["ema_50"] = ta.trend.EMAIndicator(out["Close"], window=50).ema_indicator()

    # MACD
    macd = ta.trend.MACD(out["Close"], window_slow=26, window_fast=12, window_sign=9)
    out["macd_hist"] = macd.macd_diff()

    # Volume MA
    out["vol_ma_20"] = out["Volume"].rolling(20, min_periods=1).mean()

    # Momentum (10-bar)
    out["momentum_10"] = out["Close"] - out["Close"].shift(10)

    # --- 8 Confirmation Signals ---
    out["conf_1_rsi_not_overbought"] = out["rsi"] < 90
    out["conf_2_rsi_not_oversold"] = out["rsi"] > 25
    out["conf_3_momentum_positive"] = out["momentum_10"] > 0
    out["conf_4_volatility_expanding"] = out["atr_rising"]
    out["conf_5_volume_above_avg"] = out["Volume"] > out["vol_ma_20"]
    out["conf_6_adx_trending"] = out["adx"] > 20
    out["conf_7_price_above_ema50"] = out["Close"] > out["ema_50"]
    out["conf_8_macd_bullish"] = out["macd_hist"] > 0

    conf_cols = [c for c in out.columns if c.startswith("conf_")]
    out["confirmations_met"] = out[conf_cols].sum(axis=1).astype(int)

    return out


def run_backtest(
    df: pd.DataFrame,
    leverage: float = 1.0,
    min_confirmations: int = 5,
    cooldown_bars: int = 3,
    regime_confirm_bars: int = 2,
    bullish_regimes: list = None,
    bearish_regimes: list = None,
    initial_capital: float = 100_000.0,
    aggressive_mode: bool = False,
) -> dict:
    """
    Run the full regime-based backtest.

    Parameters
    ----------
    df : pd.DataFrame
        Must have regime_id, regime_label, regime_confidence, and OHLCV + features.
    leverage : float
        Position leverage multiplier (default 2.5x).
    min_confirmations : int
        Minimum confirmations needed to enter (default 7 of 8).
    cooldown_bars : int
        Bars to wait after exit before re-entry (default 5 on daily).
    regime_confirm_bars : int
        Signal hysteresis — regime must persist for this many consecutive bars
        before it's confirmed for entry. Prevents whipsawing on noisy
        regime transitions. (default 3 = regime must hold for 3 bars)
    bullish_regimes : list
        regime_ids considered bullish for entry (default [0, 1]).
    bearish_regimes : list
        regime_ids that trigger immediate exit (default [5, 6]).
    initial_capital : float
        Starting equity.
    aggressive_mode : bool
        If True: leverage=4x, min_confirmations=5, cooldown=3, confirm=2.

    Returns
    -------
    dict with keys:
        trades: list of trade dicts
        equity_curve: pd.Series
        metrics: dict of performance stats
        df: annotated dataframe
    """
    if bullish_regimes is None:
        bullish_regimes = [0, 1, 2]
    if bearish_regimes is None:
        bearish_regimes = [5, 6]

    if aggressive_mode:
        leverage = 4.0
        min_confirmations = 5
        cooldown_bars = 3
        regime_confirm_bars = 2

    # Compute confirmations
    data = compute_confirmations(df)
    data = data.dropna().copy()

    # Pre-compute regime streak (consecutive bars in same regime)
    regime_ids = data["regime_id"].values.astype(int)
    regime_streak = np.zeros(len(regime_ids), dtype=int)
    regime_streak[0] = 1
    for j in range(1, len(regime_ids)):
        if regime_ids[j] == regime_ids[j - 1]:
            regime_streak[j] = regime_streak[j - 1] + 1
        else:
            regime_streak[j] = 1
    data["regime_streak"] = regime_streak

    # State tracking
    position_open = False
    entry_price = 0.0
    entry_bar = 0
    entry_regime = None
    cooldown_remaining = 0
    capital = initial_capital

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

        # Track unrealized PnL
        if position_open:
            bar_return = (price - prev_price) / prev_price
            capital *= (1 + bar_return * leverage)

        # --- EXIT LOGIC ---
        if position_open:
            exit_reason = None

            # Exit if regime flips to bearish (immediate — no hysteresis on exits)
            if regime in bearish_regimes:
                exit_reason = f"Regime flip - {row['regime_label']}"

            # Exit if regime is no longer bullish (neutral zone)
            elif regime not in bullish_regimes and regime not in bearish_regimes:
                # Only exit neutral if confidence is high that we left bull
                if float(row["regime_confidence"]) > 0.6:
                    exit_reason = f"Regime neutral - {row['regime_label']} (conf {row['regime_confidence']:.0%})"

            # Trailing stop in aggressive mode: -8% from peak
            if aggressive_mode and position_open:
                peak_equity = max(equity)
                if capital < peak_equity * 0.92:
                    exit_reason = "Trailing stop (-8% from peak)"

            if exit_reason:
                pnl_pct = (price - entry_price) / entry_price * leverage * 100
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
                    "confirmations_at_entry": int(data.iloc[entry_bar]["confirmations_met"]),
                })
                position_open = False
                cooldown_remaining = cooldown_bars
                signals.append("EXIT")
                equity.append(capital)
                continue

        # --- COOLDOWN ---
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            signals.append("COOLDOWN")
            equity.append(capital)
            continue

        # --- ENTRY LOGIC (with regime confirmation lag / signal hysteresis) ---
        if not position_open:
            is_bullish = regime in bullish_regimes
            enough_confs = confs >= min_confirmations
            high_confidence = float(row["regime_confidence"]) > 0.5
            regime_confirmed = streak >= regime_confirm_bars

            if is_bullish and enough_confs and high_confidence and regime_confirmed:
                position_open = True
                entry_price = price
                entry_bar = i
                entry_regime = row["regime_label"]
                signals.append("ENTRY")
                equity.append(capital)
                continue
            elif is_bullish and not regime_confirmed:
                signals.append("CONFIRMING")
                equity.append(capital)
                continue

        # --- HOLD or WAIT ---
        if position_open:
            signals.append("HOLDING")
        else:
            signals.append("CASH")
        equity.append(capital)

    # Close any open position at end
    if position_open:
        final_price = float(data.iloc[-1]["Close"])
        pnl_pct = (final_price - entry_price) / entry_price * leverage * 100
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
            "pnl_pct": round(pnl_pct, 2),
            "confirmations_at_entry": int(data.iloc[entry_bar]["confirmations_met"]),
        })

    # Pad signals/equity to match dataframe length
    while len(signals) < len(data):
        signals.append("CASH")
    while len(equity) < len(data):
        equity.append(equity[-1] if equity else initial_capital)

    # Trim to data length
    signals = signals[: len(data)]
    equity = equity[: len(data)]

    data["signal"] = signals
    data["equity"] = equity

    # --- METRICS ---
    metrics = _compute_metrics(trades, equity, data, initial_capital, leverage)

    return {
        "trades": trades,
        "equity_curve": pd.Series(equity, index=data.index),
        "metrics": metrics,
        "df": data,
    }


def get_current_signal(df: pd.DataFrame, min_confirmations: int = 5, regime_confirm_bars: int = 2) -> dict:
    """
    Determine the live trading signal from the latest bar.
    Includes regime confirmation lag (signal hysteresis).

    Returns
    -------
    dict with signal, regime, confidence, confirmations breakdown
    """
    data = compute_confirmations(df)
    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest

    regime_id = int(latest["regime_id"])
    regime_label = latest["regime_label"]
    confidence = float(latest["regime_confidence"])
    confs = int(latest["confirmations_met"])

    prev_regime = int(prev["regime_id"])

    # Compute regime streak (how many consecutive bars in current regime)
    regime_ids = data["regime_id"].values.astype(int)
    streak = 1
    for j in range(len(regime_ids) - 2, -1, -1):
        if regime_ids[j] == regime_ids[-1]:
            streak += 1
        else:
            break

    regime_confirmed = streak >= regime_confirm_bars

    # Determine signal
    is_bullish = regime_id in [0, 1, 2]
    was_bullish = prev_regime in [0, 1, 2]
    is_bearish = regime_id in [5, 6]

    if is_bullish and confs >= min_confirmations and regime_confirmed:
        if was_bullish:
            signal = "LONG -- HOLD"
            action = "Maintain long position. Regime still bullish."
        else:
            signal = "LONG -- ENTER"
            action = f"Bullish regime confirmed ({streak} bars). {confs}/8 confirmations met. Enter long."
    elif is_bullish and confs >= min_confirmations and not regime_confirmed:
        signal = "LONG -- CONFIRMING"
        action = f"Bullish regime detected but not yet confirmed ({streak}/{regime_confirm_bars} bars). Wait for confirmation."
    elif is_bearish:
        if was_bullish:
            signal = "EXIT -- REGIME FLIP"
            action = "Regime flipped bearish. Close all longs immediately."
        else:
            signal = "CASH -- BEARISH"
            action = "Market in bearish regime. Stay flat."
    else:
        signal = "CASH -- NEUTRAL"
        action = f"Regime is {regime_label}. Only {confs}/8 confirmations. No trade."

    # Confirmation breakdown
    conf_cols = [c for c in data.columns if c.startswith("conf_")]
    conf_detail = {}
    for c in conf_cols:
        name = c.replace("conf_", "").replace("_", " ").title()
        # Remove leading number
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
        "confirmation_detail": conf_detail,
        "price": float(latest["Close"]),
        "rsi": float(latest["rsi"]) if pd.notna(latest["rsi"]) else None,
        "adx": float(latest["adx"]) if pd.notna(latest["adx"]) else None,
        "macd_hist": float(latest["macd_hist"]) if pd.notna(latest["macd_hist"]) else None,
        "regime_changed": regime_id != prev_regime,
        "prev_regime": prev["regime_label"],
        "regime_streak": streak,
        "regime_confirm_bars": regime_confirm_bars,
        "regime_confirmed": regime_confirmed,
    }


def _compute_metrics(trades, equity, df, initial_capital, leverage) -> dict:
    """Compute performance metrics from backtest results."""
    if not trades:
        return {
            "total_return_pct": 0,
            "alpha_vs_buyhold": 0,
            "win_rate": 0,
            "total_trades": 0,
            "max_drawdown_pct": 0,
            "sharpe_ratio": 0,
            "profit_factor": 0,
            "avg_win_pct": 0,
            "avg_loss_pct": 0,
            "final_equity": initial_capital,
            "initial_capital": initial_capital,
            "leverage": leverage,
        }

    final_equity = equity[-1] if equity else initial_capital
    total_return = (final_equity - initial_capital) / initial_capital * 100

    # Buy & hold return
    bh_start = float(df["Close"].iloc[0])
    bh_end = float(df["Close"].iloc[-1])
    bh_return = (bh_end - bh_start) / bh_start * 100
    alpha = total_return - bh_return

    # Win/loss
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0

    gross_profit = sum(t["pnl_pct"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_pct"] for t in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    eq_series = pd.Series(equity)
    peak = eq_series.expanding().max()
    drawdown = (eq_series - peak) / peak * 100
    max_dd = drawdown.min()

    # Sharpe (annualized, assume daily bars = 252 trading days/year)
    returns = eq_series.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

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
        "final_equity": round(final_equity, 2),
        "initial_capital": initial_capital,
        "leverage": leverage,
    }


RECOMMENDED_SETTINGS = {
    "min_confs": 7,
    "regime_confirm": 2,
    "cooldown": 48,
    "min_dte": 14,
    "max_dte": 45,
    "min_avg_volume": 1_000_000,
    "min_price": 1,
    "max_price": None,
    "price_above_ema50": True,
    "ema10_above_20": False,
}
