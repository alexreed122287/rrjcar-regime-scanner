"""Tests for /api/strategy-defaults endpoint."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app import app
    return TestClient(app)


def test_strategy_defaults_returns_all_four_strategies(client):
    res = client.get("/api/strategy-defaults")
    assert res.status_code == 200
    body = res.json()
    assert set(body.keys()) == {"v1", "v2", "leaps", "bottoming"}


def test_each_strategy_has_required_default_keys(client):
    res = client.get("/api/strategy-defaults")
    body = res.json()
    required_keys = {
        "min_confs", "regime_confirm", "cooldown",
        "min_dte", "max_dte",
        "min_avg_volume", "min_price", "max_price",
        "price_above_ema50", "ema10_above_20",
    }
    for strategy, defaults in body.items():
        missing = required_keys - set(defaults.keys())
        assert not missing, f"{strategy} missing keys: {missing}"
