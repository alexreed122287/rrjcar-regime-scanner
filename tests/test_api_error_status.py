"""A failed request must not be reported as HTTP 200.

The route handlers caught every exception and returned ``{"error": ...}``, which FastAPI
serializes with a 200 status. Any client checking ``response.ok`` read a failure as a
success; only clients that happened to inspect the body for an ``"error"`` key noticed.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app as app_module
from api.errors import error_response


@pytest.fixture(scope="module")
def client():
    return TestClient(app_module.app, raise_server_exceptions=False)


def test_error_response_sets_status_and_keeps_body_shape():
    resp = error_response("boom", 502, symbol="SPY")
    assert resp.status_code == 502
    import json

    body = json.loads(resp.body)
    # The "error" key must survive so existing front-end checks keep working.
    assert body["error"] == "boom"
    assert body["symbol"] == "SPY"


def test_error_response_defaults_to_500():
    assert error_response("boom").status_code == 500


def test_backtest_failure_is_not_reported_as_success(client):
    resp = client.get("/api/backtest/NOTAREALTICKERXYZ")
    assert not resp.is_success, "a failed backtest must not return a 2xx status"
    assert resp.status_code >= 400
    assert "error" in resp.json()


def test_healthy_route_still_returns_200(client):
    """The guard must not turn working endpoints into errors."""
    assert client.get("/").status_code == 200
