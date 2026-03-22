"""
options_picker.py — Options Strike & Expiration Selector
Finds optimal call options for regime-based entries.

Given a bullish regime signal, recommends:
  - Best expiration (targeting 14-60 DTE sweet spot)
  - Best strike (targeting ~0.30-0.40 delta OTM calls for long calls)
  - Scores options by liquidity, delta, IV, and risk/reward

Uses Black-Scholes to estimate delta since yfinance doesn't provide Greeks.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timedelta
from typing import List, Dict, Optional


def black_scholes_delta(S, K, T, r, sigma, option_type="call"):
    """
    Compute Black-Scholes delta for a European option.

    Parameters
    ----------
    S : float - Current stock price
    K : float - Strike price
    T : float - Time to expiration in years
    r : float - Risk-free rate (annualized)
    sigma : float - Implied volatility (annualized)
    option_type : str - "call" or "put"
    """
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        return float(norm.cdf(d1))
    else:
        return float(norm.cdf(d1) - 1)


def black_scholes_gamma(S, K, T, r, sigma):
    """Compute gamma (same for calls and puts)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def black_scholes_theta(S, K, T, r, sigma, option_type="call"):
    """Compute theta (daily, negative = time decay)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)

    return float((term1 + term2) / 365)  # daily theta


def score_option(row, price, dte, regime_id, confirmations):
    """
    Score an option contract for suitability.

    Higher score = better pick. Factors:
    - Delta sweet spot (0.25-0.45 for OTM plays, 0.50-0.70 for ITM momentum)
    - Liquidity (volume + open interest)
    - Bid-ask spread tightness
    - DTE sweet spot (21-45 days optimal)
    - IV relative value (lower IV = cheaper premium)
    """
    score = 0.0
    delta = abs(row.get("delta", 0))
    iv = row.get("impliedVolatility", 0)
    volume = row.get("volume", 0) or 0
    oi = row.get("openInterest", 0) or 0
    bid = row.get("bid", 0) or 0
    ask = row.get("ask", 0) or 0
    mid = (bid + ask) / 2 if (bid + ask) > 0 else row.get("lastPrice", 0)

    # Skip garbage
    if mid <= 0.01 or ask <= 0:
        return -999

    # ── Delta scoring ──
    # Strong bullish regime (0-1): prefer slightly higher delta (0.35-0.55)
    # Mild bullish (2): prefer lower delta (0.25-0.40) for cheaper exposure
    if regime_id <= 1:
        # Momentum play — want more delta
        if 0.35 <= delta <= 0.55:
            score += 30
        elif 0.25 <= delta <= 0.65:
            score += 20
        elif 0.15 <= delta <= 0.75:
            score += 10
    else:
        # Cheaper speculative play
        if 0.25 <= delta <= 0.40:
            score += 30
        elif 0.15 <= delta <= 0.50:
            score += 20
        elif 0.10 <= delta <= 0.60:
            score += 10

    # ── DTE scoring ──
    if 21 <= dte <= 45:
        score += 25  # sweet spot
    elif 14 <= dte <= 60:
        score += 15
    elif 7 <= dte <= 90:
        score += 5
    else:
        score -= 10  # too short or too long

    # ── Liquidity scoring ──
    if volume >= 100 and oi >= 500:
        score += 25
    elif volume >= 50 and oi >= 100:
        score += 15
    elif volume >= 10 and oi >= 50:
        score += 8
    elif volume >= 1 or oi >= 10:
        score += 2
    else:
        score -= 15  # illiquid

    # ── Spread scoring ──
    if bid > 0 and ask > 0:
        spread_pct = (ask - bid) / mid * 100
        if spread_pct < 3:
            score += 15
        elif spread_pct < 5:
            score += 10
        elif spread_pct < 10:
            score += 5
        else:
            score -= 5

    # ── IV scoring (prefer moderate IV, not extreme) ──
    if 0.15 <= iv <= 0.40:
        score += 10
    elif 0.10 <= iv <= 0.60:
        score += 5
    elif iv > 0.80:
        score -= 10  # expensive

    # ── Confirmations bonus ──
    if confirmations >= 7:
        score += 10
    elif confirmations >= 6:
        score += 5

    # ── Price reasonableness (avoid options > 10% of stock price) ──
    if mid < price * 0.10:
        score += 5
    if mid < price * 0.03:
        score += 5  # cheap relative to stock

    return round(score, 1)


def get_options_recommendations(
    symbol: str,
    current_price: float,
    regime_id: int,
    regime_label: str,
    confirmations: int,
    signal: str,
    min_dte: int = 7,
    max_dte: int = 90,
    risk_free_rate: float = 0.045,
    top_n: int = 5,
) -> Dict:
    """
    Get top call option recommendations for a ticker.

    Only recommends calls when signal is bullish.
    Returns empty if no options available or signal is not bullish.
    """
    result = {
        "symbol": symbol,
        "price": current_price,
        "regime_label": regime_label,
        "signal": signal,
        "recommendations": [],
        "all_scored": [],
        "error": None,
    }

    # Show options for all regimes — user decides whether to trade

    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options

        if not expirations:
            result["error"] = f"No options available for {symbol}"
            return result

        today = datetime.now().date()
        scored_options = []

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
                iv = row.get("impliedVolatility", 0.3)
                if iv <= 0:
                    iv = 0.3  # fallback

                # Compute Greeks
                delta = black_scholes_delta(current_price, strike, T, risk_free_rate, iv, "call")
                gamma = black_scholes_gamma(current_price, strike, T, risk_free_rate, iv)
                theta = black_scholes_theta(current_price, strike, T, risk_free_rate, iv, "call")

                bid = row.get("bid", 0) or 0
                ask = row.get("ask", 0) or 0
                mid = (bid + ask) / 2 if (bid + ask) > 0 else row.get("lastPrice", 0)

                vol_raw = row.get("volume", 0)
                oi_raw = row.get("openInterest", 0)
                volume = int(vol_raw) if pd.notna(vol_raw) else 0
                open_interest = int(oi_raw) if pd.notna(oi_raw) else 0

                opt = {
                    "contractSymbol": row["contractSymbol"],
                    "expiration": exp_str,
                    "dte": dte,
                    "strike": strike,
                    "bid": bid,
                    "ask": ask,
                    "mid": round(mid, 2),
                    "lastPrice": row.get("lastPrice", 0) if pd.notna(row.get("lastPrice", 0)) else 0,
                    "volume": volume,
                    "openInterest": open_interest,
                    "impliedVolatility": round(iv, 4),
                    "iv_pct": round(iv * 100, 1),
                    "inTheMoney": bool(row.get("inTheMoney", False)),
                    "delta": round(delta, 3),
                    "gamma": round(gamma, 5),
                    "theta": round(theta, 3),
                    "moneyness": round((current_price - strike) / current_price * 100, 2),
                    "score": 0,
                }

                # Add delta to row for scoring
                row_dict = row.to_dict()
                row_dict["delta"] = delta
                opt["score"] = score_option(row_dict, current_price, dte, regime_id, confirmations)

                scored_options.append(opt)

        # Sort by score descending
        scored_options.sort(key=lambda x: -x["score"])

        result["all_scored"] = scored_options
        result["recommendations"] = scored_options[:top_n]
        result["total_contracts_evaluated"] = len(scored_options)

    except Exception as e:
        result["error"] = str(e)

    return result


def scan_options_for_watchlist(
    scan_results: List[Dict],
    min_dte: int = 7,
    max_dte: int = 90,
    top_n: int = 3,
) -> List[Dict]:
    """
    Given screener scan results, find best options for every bullish ticker.

    Returns list of options recommendations (only for tickers with bullish regimes).
    """
    recommendations = []

    for r in scan_results:
        regime_id = r.get("regime_id")
        if regime_id is None:
            continue

        price = r.get("price")
        if not price:
            continue

        rec = get_options_recommendations(
            symbol=r["symbol"],
            current_price=price,
            regime_id=regime_id,
            regime_label=r.get("regime_label", ""),
            confirmations=r.get("confirmations_met", 0),
            signal=r.get("signal", ""),
            min_dte=min_dte,
            max_dte=max_dte,
            top_n=top_n,
        )
        recommendations.append(rec)

    return recommendations
