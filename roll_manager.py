"""
roll_manager.py — Options Roll Strategy
Manages rolling call options to reduce risk and extend trade life.

Roll UP:  Price >= entry + 1 ATR → roll to 70-80 delta at SAME expiration for credit
Roll OUT: DTE <= 7 → roll to NEXT expiration at 70-80 delta for credit
EXIT:     If can't roll for credit → close position
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from typing import Dict, Optional, Tuple, List
from options_picker import black_scholes_delta
from tradier_broker import place_option_order, is_configured as tradier_configured


def check_roll_trigger(
    entry_price: float,
    current_price: float,
    atr: float,
    current_dte: int,
    roll_up_atr_mult: float = 1.0,
    roll_out_dte: int = 7,
) -> Optional[str]:
    """
    Check if a roll should be triggered.

    Returns: "roll_up", "roll_out", or None
    """
    if current_dte <= roll_out_dte:
        return "roll_out"

    if atr > 0 and current_price >= entry_price + (roll_up_atr_mult * atr):
        return "roll_up"

    return None


def find_roll_target(
    symbol: str,
    current_price: float,
    current_contract_bid: float,
    target_delta_range: Tuple[float, float] = (0.70, 0.80),
    same_expiry: str = None,
    next_expiry: bool = False,
    risk_free_rate: float = 0.045,
) -> Optional[Dict]:
    """
    Find the best roll target contract.

    For roll_up: same expiration, higher strike, 70-80 delta, must be for credit
    For roll_out: next expiration, 70-80 delta, must be for credit
    """
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            return None

        target_exp = None
        if same_expiry and same_expiry in expirations:
            target_exp = same_expiry
        elif next_expiry and same_expiry:
            # Find the next expiration after current
            try:
                idx = list(expirations).index(same_expiry)
                if idx + 1 < len(expirations):
                    target_exp = expirations[idx + 1]
            except ValueError:
                # Current expiry not in list, find nearest future
                from datetime import datetime
                today = datetime.now().date()
                for exp in expirations:
                    exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                    if same_expiry:
                        curr_date = datetime.strptime(same_expiry, "%Y-%m-%d").date()
                        if exp_date > curr_date:
                            target_exp = exp
                            break

        if not target_exp:
            return None

        chain = ticker.option_chain(target_exp)
        calls = chain.calls
        if calls.empty:
            return None

        from datetime import datetime
        exp_date = datetime.strptime(target_exp, "%Y-%m-%d").date()
        dte = (exp_date - datetime.now().date()).days
        T = max(dte, 1) / 365.0

        candidates = []
        for _, row in calls.iterrows():
            strike = row["strike"]
            iv = row.get("impliedVolatility", 0.3)
            if iv <= 0:
                iv = 0.3
            bid = row.get("bid", 0) or 0
            ask = row.get("ask", 0) or 0
            mid = (bid + ask) / 2 if (bid + ask) > 0 else 0

            delta = black_scholes_delta(current_price, strike, T, risk_free_rate, iv)

            if target_delta_range[0] <= delta <= target_delta_range[1]:
                # Check if we can roll for a credit (current bid > target ask)
                credit = current_contract_bid - ask
                if credit > 0:
                    candidates.append({
                        "contractSymbol": row["contractSymbol"],
                        "expiration": target_exp,
                        "dte": dte,
                        "strike": strike,
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "delta": round(delta, 3),
                        "iv": round(iv, 4),
                        "credit": round(credit, 2),
                        "volume": int(row.get("volume", 0) or 0) if pd.notna(row.get("volume")) else 0,
                        "openInterest": int(row.get("openInterest", 0) or 0) if pd.notna(row.get("openInterest")) else 0,
                    })

        if not candidates:
            return None

        # Pick the one with best credit and closest to 0.75 delta
        candidates.sort(key=lambda c: (-c["credit"], abs(c["delta"] - 0.75)))
        return candidates[0]

    except Exception:
        return None


def simulate_roll(
    entry_price: float,
    current_price: float,
    atr: float,
    bars_held: int,
    initial_dte: int = 30,
    roll_up_atr_mult: float = 1.0,
    roll_out_dte: int = 7,
    avg_credit_pct: float = 0.5,
) -> Dict:
    """
    Simulate roll behavior for backtesting (no live option chain needed).

    Estimates:
    - Roll up credit: ~0.5% of stock price per roll (conservative estimate)
    - Roll out credit: ~0.3% per roll (rolling to next exp is usually smaller credit)
    - Max 3 rolls per trade before exiting

    Returns dict with roll_count, total_credits_pct, final_action
    """
    rolls = []
    cumulative_atr_moves = 0
    remaining_dte = initial_dte - bars_held
    current_entry = entry_price
    max_rolls = 3

    # Simulate roll-ups based on price movement
    price_move = current_price - entry_price
    if atr > 0:
        atr_moves = price_move / atr
        # Each 1 ATR move = potential roll up
        while atr_moves >= roll_up_atr_mult and len(rolls) < max_rolls:
            credit_pct = avg_credit_pct
            rolls.append({
                "type": "roll_up",
                "credit_pct": credit_pct,
                "at_price": current_entry + (roll_up_atr_mult * atr),
            })
            current_entry += roll_up_atr_mult * atr
            atr_moves -= roll_up_atr_mult

    # Check if roll-out needed
    if remaining_dte <= roll_out_dte and len(rolls) < max_rolls:
        rolls.append({
            "type": "roll_out",
            "credit_pct": 0.3,
        })

    total_credit_pct = sum(r["credit_pct"] for r in rolls)
    can_roll = len(rolls) > 0

    return {
        "roll_count": len(rolls),
        "rolls": rolls,
        "total_credit_pct": round(total_credit_pct, 2),
        "can_roll": can_roll,
        "final_action": "rolled" if can_roll else "exit",
    }


def execute_roll_via_tradier(
    symbol: str,
    current_contract: str,
    target_contract: str,
    quantity: int,
) -> Dict:
    """
    Execute a roll via Tradier: sell-to-close current + buy-to-open target.
    Returns order results.
    """
    if not tradier_configured():
        return {"error": "Tradier not configured"}

    # Sell current
    sell_result = place_option_order(
        symbol=symbol,
        option_symbol=current_contract,
        side="sell_to_close",
        quantity=quantity,
        order_type="market",
        preview=False,
    )

    # Buy target
    buy_result = place_option_order(
        symbol=symbol,
        option_symbol=target_contract,
        side="buy_to_open",
        quantity=quantity,
        order_type="market",
        preview=False,
    )

    return {
        "sell": sell_result,
        "buy": buy_result,
        "rolled_from": current_contract,
        "rolled_to": target_contract,
    }
