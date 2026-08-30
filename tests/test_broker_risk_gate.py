"""
Integration tests for the risk gate on the order-submission route.

These assert the behavior that actually protects the account: POST /api/broker/ladder
must refuse to submit new risk when a circuit breaker is active, and must never be
reachable when trading is halted.
"""

import pytest
from fastapi.testclient import TestClient

import app as app_module
import api.routes_broker as routes_broker
from risk_manager import RiskStatus

HTTP_LOCKED = 423


@pytest.fixture
def client():
    return TestClient(app_module.app)


@pytest.fixture
def broker_configured(monkeypatch):
    """
    Pretend Tradier is connected so we reach the risk gate, not the config check.

    _run_ladder is replaced with a recorder so no test can reach the real broker. The
    recorder list is returned so tests can assert whether submission was attempted.
    """
    import tradier_broker

    monkeypatch.setattr(tradier_broker, "is_configured", lambda: True)

    submitted = []
    monkeypatch.setattr(
        routes_broker, "_run_ladder",
        lambda order_id, symbol, side, quantity, *a, **k: submitted.append(
            {"symbol": symbol, "side": side, "quantity": quantity}
        ),
    )
    return submitted


def set_status(monkeypatch, status: RiskStatus):
    import risk_manager

    monkeypatch.setattr(risk_manager, "check_risk_status", lambda **kw: status)


def order(client, side="buy", quantity=10):
    return client.post(
        "/api/broker/ladder",
        json={"symbol": "SPY", "side": side, "quantity": quantity},
    )


# ── halted ──

def test_halted_account_refuses_buy(client, broker_configured, monkeypatch):
    set_status(monkeypatch, RiskStatus(
        status="HALTED", halted=True, allow_new_entries=False,
        size_multiplier=0.0, reasons=["Peak drawdown breached."],
    ))
    r = order(client, "buy")
    assert r.status_code == HTTP_LOCKED
    assert r.json()["risk_status"] == "HALTED"
    assert broker_configured == [], "a halted account must not reach the broker"


def test_halted_account_refuses_sell_too(client, broker_configured, monkeypatch):
    """A hard halt stops everything — that is what the sentinel means."""
    set_status(monkeypatch, RiskStatus(
        status="HALTED", halted=True, allow_new_entries=False, size_multiplier=0.0,
    ))
    assert order(client, "sell").status_code == HTTP_LOCKED


# ── entries blocked, exits allowed ──

def test_no_entry_status_refuses_buy(client, broker_configured, monkeypatch):
    set_status(monkeypatch, RiskStatus(
        status="NO_ENTRY", allow_new_entries=False, size_multiplier=0.0,
        reasons=["Daily loss limit breached."],
    ))
    r = order(client, "buy")
    assert r.status_code == HTTP_LOCKED
    assert "Daily loss limit breached." in r.json()["reasons"]
    assert broker_configured == [], "a blocked entry must not reach the broker"


def test_no_entry_status_still_allows_exit(client, broker_configured, monkeypatch):
    """Positions must never be trapped by a risk breaker."""
    set_status(monkeypatch, RiskStatus(
        status="NO_ENTRY", allow_new_entries=False, allow_exits=True, size_multiplier=0.0,
    ))
    r = order(client, "sell")
    # Not a 423 — the exit is permitted (it fails later only because _run_ladder is
    # stubbed out, which happens in a background thread and does not affect the response).
    assert r.status_code == 200
    assert r.json().get("order_id")


# ── reduced sizing ──

def test_reduced_status_halves_quantity(client, broker_configured, monkeypatch):
    set_status(monkeypatch, RiskStatus(status="REDUCED", size_multiplier=0.5))
    r = order(client, "buy", quantity=10)
    assert r.status_code == 200
    body = r.json()
    assert body["quantity"] == 5
    assert body["requested_quantity"] == 10
    # The reduced size is what actually reached the broker layer, not just the response.
    assert broker_configured == [{"symbol": "SPY", "side": "buy", "quantity": 5}]


def test_ok_status_passes_quantity_through(client, broker_configured, monkeypatch):
    set_status(monkeypatch, RiskStatus(status="OK", size_multiplier=1.0))
    r = order(client, "buy", quantity=7)
    assert r.status_code == 200
    assert r.json()["quantity"] == 7
    assert broker_configured[0]["quantity"] == 7


# ── fail closed ──

def test_unreadable_risk_state_blocks_entry(client, broker_configured, monkeypatch):
    import risk_manager

    def boom(**kwargs):
        raise RuntimeError("broker unreachable")

    monkeypatch.setattr(risk_manager, "check_risk_status", boom)
    r = order(client, "buy")
    assert r.status_code == HTTP_LOCKED
    assert r.json()["risk_status"] == "UNKNOWN"


def test_unreadable_risk_state_still_allows_exit(client, broker_configured, monkeypatch):
    import risk_manager

    def boom(**kwargs):
        raise RuntimeError("broker unreachable")

    monkeypatch.setattr(risk_manager, "check_risk_status", boom)
    assert order(client, "sell").status_code == 200


# ── the gate cannot be bypassed by skipping configuration ──

def test_unconfigured_broker_never_places_orders(client, monkeypatch):
    import tradier_broker

    monkeypatch.setattr(tradier_broker, "is_configured", lambda: False)
    r = order(client, "buy")
    assert "error" in r.json()
