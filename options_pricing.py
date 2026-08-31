"""Minimal Black-Scholes for the long-call leg.

Deliberately small. The strategy is described as deep-ITM calls used as a stock
replacement, and the backtester needs exactly two things that a roll changes: what the
call is worth, and how much of the underlying's move it participates in. So: European
call price and delta, no dividends, no smile, no early exercise, no term structure.

Why this exists at all: ``strategy_v2`` used to hand out a flat unconditional credit on
every roll and keep compounding full capital on the underlying afterwards. Measured on
five tickers, that credit was a median 41% of the reported return (see
``docs/validation-findings.md``). A roll up is not income -- it converts exposure into
cash and leaves you with less delta. Pricing the two legs is the only way to get the
sign and the magnitude of that trade right.
"""

from __future__ import annotations

import math

_SQRT2 = math.sqrt(2.0)


def norm_cdf(x: float) -> float:
    """Standard normal CDF via erf, so this module needs no scipy."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> tuple:
    """Price and delta of a European call.

    Parameters
    ----------
    S, K : spot and strike, same units.
    T : time to expiry in YEARS.
    sigma : annualized volatility as a decimal (0.25 == 25%).
    r : continuously-compounded risk-free rate.

    Returns
    -------
    (price, delta) : delta is in [0, 1].

    Degenerate inputs collapse to intrinsic value rather than raising, because a
    backtest walking a position toward expiry will legitimately reach ``T == 0``.
    """
    if S <= 0.0 or K <= 0.0:
        return (0.0, 0.0)
    intrinsic = max(S - K, 0.0)
    if T <= 0.0 or sigma <= 0.0:
        # At expiry delta is a step function; 0.5 at the kink is arbitrary but never
        # reached in practice by a float comparison.
        return (intrinsic, 1.0 if S > K else 0.0)

    sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sqrt_t
    d2 = d1 - sqrt_t
    price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    # Floating point can put the formula a hair under intrinsic for very deep ITM calls.
    return (max(price, intrinsic), norm_cdf(d1))


def call_leverage(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Dollar-delta per dollar of premium: ``delta * S / price``.

    This is the number that makes a call a "stock replacement" -- it is how many dollars
    of underlying exposure one dollar of premium buys. Deep ITM and long dated it tends
    toward 1.0-ish leverage on a much smaller outlay; near the money it explodes.
    """
    price, delta = bs_call(S, K, T, sigma, r)
    if price <= 0.0:
        return 0.0
    return delta * S / price
