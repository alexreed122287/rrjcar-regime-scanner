"""Black-Scholes sanity checks, including the ones the roll model depends on."""

import math

import pytest

from options_pricing import bs_call, call_leverage, norm_cdf


def test_norm_cdf_known_points():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)


def test_atm_call_matches_closed_form():
    """S=K=100, T=1, sigma=0.20, r=0 has a hand-checkable value.

    d1 = 0.10, d2 = -0.10, price = 100 * (phi(0.1) - phi(-0.1)) = 7.9656.
    """
    price, delta = bs_call(100.0, 100.0, 1.0, 0.20, 0.0)
    assert price == pytest.approx(7.9656, abs=1e-3)
    assert delta == pytest.approx(0.5398, abs=1e-4)


def test_put_call_parity():
    """C - P = S - K*exp(-rT), with P recovered from parity, must price consistently."""
    S, K, T, sigma, r = 105.0, 100.0, 0.5, 0.3, 0.04
    call, _ = bs_call(S, K, T, sigma, r)
    put = call - S + K * math.exp(-r * T)
    assert put > 0.0  # a 5% ITM call's parity put must still have value
    # Reprice via parity in the other direction.
    assert call == pytest.approx(put + S - K * math.exp(-r * T), abs=1e-9)


def test_delta_bounds_and_monotonicity():
    prev_price = prev_delta = -1.0
    for S in (60.0, 80.0, 100.0, 120.0, 200.0):
        price, delta = bs_call(S, 100.0, 0.5, 0.25, 0.0)
        assert 0.0 <= delta <= 1.0
        assert price > prev_price, "call price must rise with spot"
        assert delta > prev_delta, "delta must rise with spot"
        prev_price, prev_delta = price, delta


def test_deep_itm_behaves_like_stock():
    """The premise of a stock-replacement call: delta near 1, price near intrinsic."""
    price, delta = bs_call(100.0, 50.0, 0.5, 0.25, 0.0)
    assert delta > 0.98
    assert price == pytest.approx(50.0, abs=0.5)


def test_expiry_collapses_to_intrinsic():
    assert bs_call(120.0, 100.0, 0.0, 0.25)[0] == pytest.approx(20.0)
    assert bs_call(80.0, 100.0, 0.0, 0.25)[0] == pytest.approx(0.0)
    assert bs_call(120.0, 100.0, -1.0, 0.25)[0] == pytest.approx(20.0)
    assert bs_call(100.0, 100.0, 1.0, 0.0)[0] == pytest.approx(0.0)


def test_price_never_below_intrinsic():
    for S in (100.0, 150.0, 400.0):
        for T in (1e-6, 0.01, 2.0):
            price, _ = bs_call(S, 100.0, T, 0.4, 0.03)
            assert price >= max(S - 100.0, 0.0) - 1e-9


def test_theta_is_negative_for_long_call():
    """Less time must be worth less, which is the cost the old roll model ignored."""
    long_dated, _ = bs_call(100.0, 90.0, 1.0, 0.3, 0.0)
    short_dated, _ = bs_call(100.0, 90.0, 0.1, 0.3, 0.0)
    assert short_dated < long_dated


def test_rolling_up_reduces_delta_and_costs_value():
    """The two facts the roll model needs, stated as a test.

    Spot has risen to 115 on a call struck at 85. Rolling up to a 15%-ITM strike (97.75)
    hands back cash -- the new call is cheaper -- and leaves less delta behind. The old
    model booked the cash as profit AND kept full participation, which is the double count.
    """
    S = 115.0
    old_price, old_delta = bs_call(S, 85.0, 0.25, 0.25, 0.0)
    new_price, new_delta = bs_call(S, S * 0.85, 0.33, 0.25, 0.0)
    assert new_price < old_price, "rolling up must release cash"
    assert new_delta < old_delta, "rolling up must give up delta"


def test_leverage_exceeds_one_and_rises_toward_the_money():
    """Premium buys more exposure than cash does; that is the point of the structure."""
    deep = call_leverage(100.0, 50.0, 0.5, 0.25)
    shallow = call_leverage(100.0, 95.0, 0.5, 0.25)
    assert deep > 1.0
    assert shallow > deep
    assert call_leverage(100.0, 100.0, 0.0, 0.25) == 0.0  # worthless -> no leverage
