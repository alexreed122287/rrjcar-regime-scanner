"""
strategy_leaps.py — LEAPS-Optimized Screening Strategy

LEAPS (Long-Term Equity Anticipation Securities) are options with 180-730+ DTE.
They behave more like leveraged stock than short-term options, so the screening
criteria focuses on:

  1. DURABLE uptrends (not just recent momentum)
  2. Fundamental strength (earnings, revenue growth proxied by price action)
  3. LOW implied volatility (cheap premium = better entry)
  4. HIGH delta (deep ITM or ATM for stock replacement)
  5. Strong institutional participation (volume, relative strength)

10 LEAPS-Specific Confirmations:
  1. Price above 200 EMA (long-term uptrend — most important for LEAPS)
  2. Price above 50 EMA (medium-term trend intact)
  3. 50 EMA above 200 EMA (golden cross structure)
  4. RSI 40-70 (not overbought — want to buy on pullbacks, not extensions)
  5. ADX > 20 (trending market, not ranging)
  6. MACD above signal line (bullish momentum confirmation)
  7. 52-week relative strength > 0 (outperforming over past year)
  8. IV Rank < 50% (implied volatility is cheap — options are affordable)
  9. Monthly higher lows (uptrend structure over 3 months)
  10. Volume above 50-day average (institutional participation)
"""

import numpy as np
import pandas as pd
import ta
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional
from scipy.stats import norm


def compute_leaps_confirmations(df: pd.DataFrame) -> pd.DataFrame:
    """
    10 confirmation signals optimized for LEAPS entries.
    Focused on durable trends and cheap IV rather than short-term momentum.
    """
    out = df.copy()

    # ── Core indicators ──
    out["rsi"] = ta.momentum.RSIIndicator(out["Close"], window=14).rsi()

    atr_ind = ta.volatility.AverageTrueRange(out["High"], out["Low"], out["Close"], window=14)
    out["atr"] = atr_ind.average_true_range()

    adx_ind = ta.trend.ADXIndicator(out["High"], out["Low"], out["Close"], window=14)
    out["adx"] = adx_ind.adx()

    # EMAs — long-term focus
    out["ema_50"] = ta.trend.EMAIndicator(out["Close"], window=50).ema_indicator()
    out["ema_200"] = ta.trend.EMAIndicator(out["Close"], window=200).ema_indicator()

    # MACD
    macd_ind = ta.trend.MACD(out["Close"], window_slow=26, window_fast=12, window_sign=9)
    out["macd_line"] = macd_ind.macd()
    out["macd_signal"] = macd_ind.macd_signal()
    out["macd_hist"] = macd_ind.macd_diff()

    # Volume
    out["vol_ma_50"] = out["Volume"].rolling(50, min_periods=20).mean()

    # 52-week performance (relative strength)
    out["pct_52w"] = out["Close"].pct_change(252) * 100

    # Historical volatility (20-day) and HV rank over 1 year
    out["hv_20"] = out["Close"].pct_change().rolling(20).std() * np.sqrt(252)
    out["hv_rank"] = out["hv_20"].rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # Monthly lows for higher-low detection (check last 3 months = ~63 trading days)
    out["low_1m"] = out["Low"].rolling(21, min_periods=10).min()
    out["low_2m"] = out["Low"].shift(21).rolling(21, min_periods=10).min()
    out["low_3m"] = out["Low"].shift(42).rolling(21, min_periods=10).min()

    # ═══ 10 LEAPS CONFIRMATIONS ═══

    # 1. Price above 200 EMA (long-term uptrend)
    out["conf_01_above_ema200"] = out["Close"] > out["ema_200"]

    # 2. Price above 50 EMA (medium-term trend)
    out["conf_02_above_ema50"] = out["Close"] > out["ema_50"]

    # 3. Golden cross structure (50 EMA > 200 EMA)
    out["conf_03_golden_cross"] = out["ema_50"] > out["ema_200"]

    # 4. RSI sweet spot (40-70) — want pullback entries, not chasing
    out["conf_04_rsi_sweet"] = (out["rsi"] >= 40) & (out["rsi"] <= 70)

    # 5. ADX > 20 (trending, not ranging)
    out["conf_05_adx_trend"] = out["adx"] > 20

    # 6. MACD above signal (bullish momentum)
    out["conf_06_macd_bull"] = out["macd_line"] > out["macd_signal"]

    # 7. 52-week relative strength > 0 (positive YoY performance)
    out["conf_07_52w_strength"] = out["pct_52w"] > 0

    # 8. IV rank < 50% (cheap implied volatility — affordable premium)
    out["conf_08_low_iv"] = out["hv_rank"] < 0.50

    # 9. Monthly higher lows (3-month uptrend structure)
    out["conf_09_higher_lows"] = (out["low_1m"] > out["low_2m"]) & (out["low_2m"] > out["low_3m"])

    # 10. Volume above 50-day average (institutional participation)
    out["conf_10_vol_confirm"] = out["Volume"] > out["vol_ma_50"]

    # ── Aggregate ──
    conf_cols = [c for c in out.columns if c.startswith("conf_")]
    # Fill NaN with False so missing data doesn't break aggregation
    for c in conf_cols:
        out[c] = out[c].fillna(False)
    out["confirmations_met"] = out[conf_cols].sum(axis=1).astype(int)

    return out


def get_current_signal_leaps(df: pd.DataFrame, min_confirmations: int = 7, regime_confirm_bars: int = 3) -> dict:
    """Get current LEAPS signal for live use."""
    data = compute_leaps_confirmations(df)
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
    is_neutral_bull = regime_id in [0, 1, 2, 3]  # Include Neutral for LEAPS (long-term play)
    was_bullish = prev_regime in [0, 1, 2]
    is_bearish = regime_id in [5, 6]

    # LEAPS signal determination — more permissive than short-term strategies
    # because LEAPS care about long-term trend (200 EMA, golden cross) more than
    # short-term regime. If confirmations pass, the trend is intact.
    if confs >= min_confirmations and is_bullish and regime_confirmed:
        signal = "LEAPS -- BUY"
        action = f"Strong LEAPS entry. {confs}/10 confirmations. {regime_label} confirmed ({streak} bars). IV rank: {hv_rank:.0%}."
    elif confs >= min_confirmations and is_bullish and not regime_confirmed:
        signal = "LEAPS -- WATCH"
        action = f"Bullish with {confs}/10 confs, confirming regime ({streak}/{regime_confirm_bars} bars). Prepare LEAPS entry."
    elif confs >= min_confirmations and is_neutral_bull:
        # Neutral regime but confirmations pass — long-term trend intact
        signal = "LEAPS -- WATCH"
        action = f"Trend intact ({confs}/10 confs) but regime is {regime_label}. Wait for bullish confirmation or enter on dip."
    elif confs >= min_confirmations - 1 and is_neutral_bull:
        # Close to threshold — worth monitoring
        signal = "LEAPS -- WATCH"
        action = f"Near threshold ({confs}/10 confs). {regime_label}. Monitor for confirmation."
    elif is_bearish:
        if was_bullish:
            signal = "LEAPS -- EXIT"
            action = "Regime flipped bearish. Consider closing or rolling LEAPS."
        else:
            signal = "LEAPS -- AVOID"
            action = "Bearish regime. No LEAPS entry."
    elif confs >= min_confirmations and is_bearish:
        signal = "LEAPS -- AVOID"
        action = f"Confirmations pass ({confs}/10) but regime is bearish. Avoid."
    else:
        signal = "LEAPS -- WAIT"
        action = f"Regime: {regime_label}. {confs}/10 confirmations. Not enough for LEAPS entry."

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
        "confirmations_total": 10,
        "confirmation_detail": conf_detail,
        "price": float(latest["Close"]),
        "rsi": float(latest["rsi"]) if pd.notna(latest["rsi"]) else None,
        "adx": float(latest["adx"]) if pd.notna(latest["adx"]) else None,
        "macd_hist": float(latest["macd_hist"]) if pd.notna(latest["macd_hist"]) else None,
        "hv_rank": round(hv_rank, 2),
        "pct_52w": round(float(latest["pct_52w"]), 1) if pd.notna(latest.get("pct_52w")) else None,
        "regime_changed": regime_id != prev_regime,
        "prev_regime": prev["regime_label"],
        "regime_streak": streak,
        "regime_confirm_bars": regime_confirm_bars,
        "regime_confirmed": regime_confirmed,
    }


# ═══════════════════════════════════════════
#  LEAPS CONTRACT SELECTOR
# ═══════════════════════════════════════════

def _bs_delta(S, K, T, r, sigma):
    """Black-Scholes call delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1))


def _bs_theta(S, K, T, r, sigma):
    """Black-Scholes daily theta for calls."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    return float((term1 + term2) / 365)


def score_leaps_contract(strike, mid, bid, ask, dte, delta, iv, oi, volume, spot, hv_rank):
    """
    Score a LEAPS contract. Higher = better.

    LEAPS-specific criteria:
    - Deep ITM or ATM (delta 0.60-0.80 ideal for stock replacement)
    - Long DTE (270-540 sweet spot, >180 minimum)
    - Low IV relative to HV (cheap premium)
    - Good liquidity (tight spread, volume, open interest)
    - Low daily theta decay relative to delta
    """
    score = 0.0

    if mid <= 0.05 or ask <= 0:
        return -999

    # ── Delta: 0.60-0.80 ideal for LEAPS (stock replacement) ──
    if 0.65 <= delta <= 0.80:
        score += 35  # sweet spot
    elif 0.55 <= delta <= 0.85:
        score += 25
    elif 0.45 <= delta <= 0.90:
        score += 15
    elif 0.30 <= delta <= 0.95:
        score += 5
    else:
        score -= 10

    # ── DTE: 270-540 days ideal ──
    if 270 <= dte <= 540:
        score += 30  # sweet spot
    elif 180 <= dte <= 730:
        score += 20
    elif 150 <= dte <= 365:
        score += 10
    else:
        score -= 10

    # ── IV: prefer low IV (cheap premium) ──
    if iv < 0.25:
        score += 20  # very cheap
    elif iv < 0.35:
        score += 15
    elif iv < 0.45:
        score += 10
    elif iv < 0.60:
        score += 5
    else:
        score -= 5  # expensive

    # ── Liquidity ──
    if volume >= 50 and oi >= 500:
        score += 20
    elif volume >= 20 and oi >= 100:
        score += 12
    elif volume >= 5 and oi >= 50:
        score += 6
    elif oi >= 10:
        score += 2
    else:
        score -= 10

    # ── Spread tightness ──
    if bid > 0 and ask > 0:
        spread_pct = (ask - bid) / mid * 100
        if spread_pct < 3:
            score += 15
        elif spread_pct < 5:
            score += 10
        elif spread_pct < 8:
            score += 5
        elif spread_pct < 15:
            score += 0
        else:
            score -= 10

    # ── Theta efficiency: theta/delta ratio (lower = better for LEAPS) ──
    if delta > 0:
        theta_per_delta = abs(_bs_theta(spot, strike, dte / 365, 0.045, iv)) / delta
        if theta_per_delta < 0.005:
            score += 10  # very efficient
        elif theta_per_delta < 0.01:
            score += 5

    # ── Moneyness bonus: slightly ITM preferred for LEAPS ──
    moneyness = (spot - strike) / spot
    if 0.02 <= moneyness <= 0.15:
        score += 10  # slightly ITM
    elif -0.03 <= moneyness <= 0.02:
        score += 5  # near ATM

    return round(score, 1)


def find_best_leaps(
    symbol: str,
    spot_price: float,
    hv_rank: float = 0.5,
    min_dte: int = 180,
    max_dte: int = 730,
    top_n: int = 5,
    risk_free_rate: float = 0.045,
) -> Dict:
    """
    Find the best LEAPS contracts for a symbol.

    Returns top N scored contracts with strike, expiration, Greeks, and score.
    """
    result = {
        "symbol": symbol.upper(),
        "spot_price": spot_price,
        "recommendations": [],
        "error": None,
    }

    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options

        if not expirations:
            result["error"] = f"No options for {symbol}"
            return result

        today = datetime.now().date()
        scored = []

        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days

            if dte < min_dte or dte > max_dte:
                continue

            try:
                chain = ticker.option_chain(exp_str)
                calls = chain.calls
            except Exception:
                continue

            if calls.empty:
                continue

            T = dte / 365.0

            for _, row in calls.iterrows():
                strike = row["strike"]
                iv = row.get("impliedVolatility", 0.3) or 0.3
                bid = float(row.get("bid", 0) or 0)
                ask = float(row.get("ask", 0) or 0)
                mid = (bid + ask) / 2 if (bid + ask) > 0 else float(row.get("lastPrice", 0) or 0)
                vol = int(row["volume"]) if pd.notna(row.get("volume")) else 0
                oi = int(row["openInterest"]) if pd.notna(row.get("openInterest")) else 0

                delta = _bs_delta(spot_price, strike, T, risk_free_rate, iv)
                theta = _bs_theta(spot_price, strike, T, risk_free_rate, iv)

                sc = score_leaps_contract(strike, mid, bid, ask, dte, delta, iv, oi, vol, spot_price, hv_rank)

                if sc > 0:
                    scored.append({
                        "contractSymbol": row.get("contractSymbol", ""),
                        "strike": strike,
                        "expiration": exp_str,
                        "dte": dte,
                        "bid": bid,
                        "ask": ask,
                        "mid": round(mid, 2),
                        "volume": vol,
                        "openInterest": oi,
                        "iv": round(iv, 4),
                        "iv_pct": round(iv * 100, 1),
                        "delta": round(delta, 3),
                        "theta": round(theta, 3),
                        "moneyness": round((spot_price - strike) / spot_price * 100, 2),
                        "inTheMoney": spot_price > strike,
                        "score": sc,
                    })

        scored.sort(key=lambda x: -x["score"])
        result["recommendations"] = scored[:top_n]
        result["total_evaluated"] = len(scored)

    except Exception as e:
        result["error"] = str(e)

    return result
