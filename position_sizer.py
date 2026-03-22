"""
position_sizer.py — Confidence-Based Position Sizing
Scales position size by regime confidence + confirmation count.
Max risk per trade: 10% of account (configurable).
"""

from typing import Dict


def compute_position_size(
    account_equity: float,
    entry_price: float,
    atr: float,
    regime_confidence: float,
    confirmations_met: int,
    confirmations_total: int = 12,
    option_mid: float = None,
    max_risk_pct: float = 0.10,
    buying_power: float = None,
) -> Dict:
    """
    Compute position size based on signal confidence.

    Tiers:
      FULL (10%):  confidence > 80% AND confs >= 10/12
      HIGH (7.5%): confidence > 70% AND confs >= 8/12
      MED  (5%):   confidence > 60% AND confs >= 6/12
      MIN  (2.5%): below thresholds

    Shares: risk_amount / (2 * ATR)  [stop distance = 2x ATR]
    Contracts: risk_amount / (option_mid * 100)
    """
    conf_ratio = confirmations_met / max(confirmations_total, 1)

    if regime_confidence > 0.80 and conf_ratio >= 0.83:
        tier = "FULL"
        risk_mult = 1.0
    elif regime_confidence > 0.70 and conf_ratio >= 0.67:
        tier = "HIGH"
        risk_mult = 0.75
    elif regime_confidence > 0.60 and conf_ratio >= 0.50:
        tier = "MED"
        risk_mult = 0.50
    else:
        tier = "MIN"
        risk_mult = 0.25

    risk_dollars = account_equity * max_risk_pct * risk_mult
    stop_distance = 2 * atr if atr > 0 else entry_price * 0.05

    # Shares
    shares = int(risk_dollars / stop_distance) if stop_distance > 0 else 0
    shares = max(shares, 1)

    # Contracts
    contracts = 0
    if option_mid and option_mid > 0:
        contracts = int(risk_dollars / (option_mid * 100))
        contracts = max(contracts, 1)
        if buying_power and (contracts * option_mid * 100) > buying_power:
            contracts = int(buying_power / (option_mid * 100))
            contracts = max(contracts, 1)

    return {
        "confidence_tier": tier,
        "risk_multiplier": risk_mult,
        "risk_dollars": round(risk_dollars, 2),
        "shares": shares,
        "contracts": contracts,
        "stop_distance": round(stop_distance, 2),
        "max_risk_pct": max_risk_pct,
    }
