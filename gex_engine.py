"""
gex_engine.py — Gamma Exposure (GEX) Analysis Engine

Computes per-strike and aggregate GEX from options chain data.
Identifies key levels: call wall, put wall, GEX flip point, max gamma strike.
Provides GEX-informed contract strategy recommendations.

GEX = Gamma × Open Interest × 100 × Spot Price
  - Call GEX is positive (dealers long gamma → mean-reverting)
  - Put GEX is negative (dealers short gamma → trending)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Optional
from scipy.stats import norm


def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma (same for calls and puts)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def fetch_options_chain(symbol: str, min_dte: int = 0, max_dte: int = 730) -> Dict:
    """
    Fetch full options chain (calls + puts) for all valid expirations.

    Returns dict with:
      - calls: list of dicts per contract
      - puts: list of dicts per contract
      - expirations: list of expiration dates used
      - spot_price: current stock price
    """
    ticker = yf.Ticker(symbol)
    expirations = ticker.options

    if not expirations:
        return {"calls": [], "puts": [], "expirations": [], "spot_price": 0, "error": f"No options for {symbol}"}

    # Get spot price
    hist = ticker.history(period="2d")
    if hist.empty:
        return {"calls": [], "puts": [], "expirations": [], "spot_price": 0, "error": "No price data"}
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    spot = float(hist["Close"].iloc[-1])

    today = datetime.now().date()
    all_calls = []
    all_puts = []
    used_exps = []

    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days

        if dte < min_dte or dte > max_dte:
            continue

        try:
            chain = ticker.option_chain(exp_str)
        except Exception:
            continue

        used_exps.append(exp_str)

        for _, row in chain.calls.iterrows():
            all_calls.append({
                "strike": row["strike"],
                "expiration": exp_str,
                "dte": dte,
                "volume": int(row["volume"]) if pd.notna(row.get("volume")) else 0,
                "openInterest": int(row["openInterest"]) if pd.notna(row.get("openInterest")) else 0,
                "impliedVolatility": float(row.get("impliedVolatility", 0.3)) or 0.3,
                "bid": float(row.get("bid", 0)) if pd.notna(row.get("bid")) else 0,
                "ask": float(row.get("ask", 0)) if pd.notna(row.get("ask")) else 0,
                "lastPrice": float(row.get("lastPrice", 0)) if pd.notna(row.get("lastPrice")) else 0,
                "inTheMoney": bool(row.get("inTheMoney", False)),
                "contractSymbol": row.get("contractSymbol", ""),
            })

        for _, row in chain.puts.iterrows():
            all_puts.append({
                "strike": row["strike"],
                "expiration": exp_str,
                "dte": dte,
                "volume": int(row["volume"]) if pd.notna(row.get("volume")) else 0,
                "openInterest": int(row["openInterest"]) if pd.notna(row.get("openInterest")) else 0,
                "impliedVolatility": float(row.get("impliedVolatility", 0.3)) or 0.3,
                "bid": float(row.get("bid", 0)) if pd.notna(row.get("bid")) else 0,
                "ask": float(row.get("ask", 0)) if pd.notna(row.get("ask")) else 0,
                "lastPrice": float(row.get("lastPrice", 0)) if pd.notna(row.get("lastPrice")) else 0,
                "inTheMoney": bool(row.get("inTheMoney", False)),
                "contractSymbol": row.get("contractSymbol", ""),
            })

    return {
        "calls": all_calls,
        "puts": all_puts,
        "expirations": used_exps,
        "spot_price": spot,
        "error": None,
    }


def compute_gex_profile(
    symbol: str,
    min_dte: int = 0,
    max_dte: int = 730,
    risk_free_rate: float = 0.045,
) -> Dict:
    """
    Compute full GEX profile for a symbol.

    Returns:
      - gex_by_strike: list of {strike, call_gex, put_gex, net_gex}
      - total_gex: aggregate net GEX
      - call_wall: strike with highest call GEX (resistance/magnet)
      - put_wall: strike with highest |put GEX| (support)
      - gex_flip: strike where net GEX flips sign (key pivot)
      - max_gamma_strike: strike with highest |net GEX|
      - gex_bias: "positive" or "negative"
      - spot_price: current price
    """
    chain = fetch_options_chain(symbol, min_dte=min_dte, max_dte=max_dte)

    if chain.get("error"):
        return {"error": chain["error"], "symbol": symbol}

    spot = chain["spot_price"]
    if spot <= 0:
        return {"error": "Invalid spot price", "symbol": symbol}

    # Aggregate GEX by strike across all expirations
    strike_gex = {}  # strike -> {call_gex, put_gex}

    for c in chain["calls"]:
        K = c["strike"]
        oi = c["openInterest"]
        iv = c["impliedVolatility"]
        T = c["dte"] / 365.0

        if oi <= 0 or T <= 0:
            continue

        gamma = _bs_gamma(spot, K, T, risk_free_rate, iv)
        # Call GEX: positive (dealers long gamma from selling calls to customers)
        gex = gamma * oi * 100 * spot

        if K not in strike_gex:
            strike_gex[K] = {"call_gex": 0, "put_gex": 0}
        strike_gex[K]["call_gex"] += gex

    for p in chain["puts"]:
        K = p["strike"]
        oi = p["openInterest"]
        iv = p["impliedVolatility"]
        T = p["dte"] / 365.0

        if oi <= 0 or T <= 0:
            continue

        gamma = _bs_gamma(spot, K, T, risk_free_rate, iv)
        # Put GEX: negative (dealers short gamma from selling puts to customers)
        gex = -gamma * oi * 100 * spot

        if K not in strike_gex:
            strike_gex[K] = {"call_gex": 0, "put_gex": 0}
        strike_gex[K]["put_gex"] += gex

    if not strike_gex:
        return {"error": "No valid options data", "symbol": symbol}

    # Build sorted profile
    strikes = sorted(strike_gex.keys())
    gex_list = []
    for K in strikes:
        cg = strike_gex[K]["call_gex"]
        pg = strike_gex[K]["put_gex"]
        gex_list.append({
            "strike": K,
            "call_gex": round(cg, 2),
            "put_gex": round(pg, 2),
            "net_gex": round(cg + pg, 2),
        })

    # Key levels
    total_gex = sum(g["net_gex"] for g in gex_list)
    call_wall = max(gex_list, key=lambda g: g["call_gex"])["strike"]
    put_wall = min(gex_list, key=lambda g: g["put_gex"])["strike"]  # most negative put_gex
    max_gamma_strike = max(gex_list, key=lambda g: abs(g["net_gex"]))["strike"]

    # GEX flip point: strike closest to where net_gex changes sign near spot
    # Focus on strikes within 20% of spot price
    near_spot = [g for g in gex_list if abs(g["strike"] - spot) / spot < 0.20]
    gex_flip = None
    if len(near_spot) >= 2:
        for i in range(1, len(near_spot)):
            if near_spot[i - 1]["net_gex"] * near_spot[i]["net_gex"] < 0:
                # Sign flip between these strikes — interpolate
                g1, g2 = near_spot[i - 1], near_spot[i]
                ratio = abs(g1["net_gex"]) / (abs(g1["net_gex"]) + abs(g2["net_gex"]))
                gex_flip = round(g1["strike"] + ratio * (g2["strike"] - g1["strike"]), 2)
                break

    gex_bias = "positive" if total_gex > 0 else "negative"

    return {
        "symbol": symbol.upper(),
        "spot_price": round(spot, 2),
        "gex_by_strike": gex_list,
        "total_gex": round(total_gex, 2),
        "call_wall": call_wall,
        "put_wall": put_wall,
        "gex_flip": gex_flip,
        "max_gamma_strike": max_gamma_strike,
        "gex_bias": gex_bias,
        "expirations_used": len(chain["expirations"]),
        "total_calls": len(chain["calls"]),
        "total_puts": len(chain["puts"]),
        "error": None,
    }


# ── GEX Strategy Matrix ──

GEX_STRATEGIES = {
    "pin_play": {
        "label": "Pin Play",
        "description": "Positive GEX = price pins near call wall. Buy calls below call wall, target it as profit zone.",
        "strike_guidance": "ATM to slightly OTM, below call wall",
        "dte_range": (7, 30),
    },
    "controlled_drift": {
        "label": "Controlled Drift",
        "description": "Positive GEX with mild bullish trend. Price drifts up slowly toward call wall.",
        "strike_guidance": "Slightly OTM, near call wall",
        "dte_range": (21, 45),
    },
    "breakout_ride": {
        "label": "Breakout Ride",
        "description": "Negative GEX = dealers short gamma, trending likely. Ride momentum with OTM calls.",
        "strike_guidance": "OTM, above max gamma strike",
        "dte_range": (14, 45),
    },
    "trend_follow": {
        "label": "Trend Follow",
        "description": "Negative GEX with sustained bullish trend. Directional calls with more time.",
        "strike_guidance": "Slightly OTM",
        "dte_range": (30, 60),
    },
    "gamma_scalp": {
        "label": "Gamma Scalp",
        "description": "Price near GEX flip point = high volatility zone. Short DTE, quick in/out.",
        "strike_guidance": "ATM",
        "dte_range": (0, 14),
    },
}


def gex_contract_strategy(
    gex_profile: Dict,
    regime_id: int,
    regime_label: str = "",
) -> Dict:
    """
    Determine optimal contract strategy based on GEX profile and regime.

    Returns strategy recommendation with strike range, DTE range, and rationale.
    """
    if gex_profile.get("error"):
        return {"strategy": None, "error": gex_profile["error"]}

    spot = gex_profile["spot_price"]
    bias = gex_profile["gex_bias"]
    call_wall = gex_profile["call_wall"]
    put_wall = gex_profile["put_wall"]
    gex_flip = gex_profile.get("gex_flip")
    is_bullish = regime_id in [0, 1, 2]
    is_strong_bull = regime_id in [0, 1]

    # Check if price is near GEX flip point
    near_flip = False
    if gex_flip and spot > 0:
        near_flip = abs(spot - gex_flip) / spot < 0.02  # within 2%

    # Strategy selection
    if near_flip and is_bullish:
        strat_key = "gamma_scalp"
    elif bias == "positive" and is_strong_bull:
        strat_key = "pin_play"
    elif bias == "positive" and is_bullish:
        strat_key = "controlled_drift"
    elif bias == "negative" and is_strong_bull:
        strat_key = "breakout_ride"
    elif bias == "negative" and is_bullish:
        strat_key = "trend_follow"
    else:
        # Not bullish — no call strategy
        return {
            "strategy": None,
            "strategy_key": None,
            "reason": f"No call strategy for {regime_label} regime",
            "gex_bias": bias,
        }

    strat = GEX_STRATEGIES[strat_key]

    # Compute recommended strike range
    if strat_key == "pin_play":
        strike_min = round(spot * 0.98, 2)
        strike_max = round(min(spot * 1.05, call_wall), 2)
    elif strat_key == "controlled_drift":
        strike_min = round(spot, 2)
        strike_max = round(call_wall * 1.02, 2)
    elif strat_key == "breakout_ride":
        strike_min = round(spot * 1.01, 2)
        strike_max = round(spot * 1.10, 2)
    elif strat_key == "trend_follow":
        strike_min = round(spot * 0.99, 2)
        strike_max = round(spot * 1.07, 2)
    elif strat_key == "gamma_scalp":
        strike_min = round(spot * 0.98, 2)
        strike_max = round(spot * 1.02, 2)
    else:
        strike_min = round(spot * 0.95, 2)
        strike_max = round(spot * 1.10, 2)

    return {
        "strategy": strat["label"],
        "strategy_key": strat_key,
        "description": strat["description"],
        "strike_guidance": strat["strike_guidance"],
        "recommended_strike_min": strike_min,
        "recommended_strike_max": strike_max,
        "recommended_dte_min": strat["dte_range"][0],
        "recommended_dte_max": strat["dte_range"][1],
        "gex_bias": bias,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "gex_flip": gex_flip,
        "stop_reference": put_wall,
        "target_reference": call_wall,
        "regime_label": regime_label,
    }
