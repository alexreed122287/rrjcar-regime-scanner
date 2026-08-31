"""Guard against fake-async route handlers that freeze the whole server.

Background
----------
Every route in this app was declared ``async def`` while doing blocking work --
``fetch_data`` (network), ``RegimeDetector.train`` (CPU-bound EM), ``scan_watchlist``
(a ThreadPoolExecutor joined synchronously). FastAPI runs ``async def`` handlers *on the event
loop*, so any one of those calls froze the entire process until it finished.

Measured on a 21-ticker scan before the fix: ``/api/apis`` -- a trivial, unrelated endpoint --
took **7.48 s** against a 0.5 ms baseline. Worse, ``/api/scan/status`` exists so the UI can
poll scan progress, and it could never report ``"scanning"``: the loop was blocked until the
scan finished, by which point the cached status had already flipped to ``"done"``. The progress
indicator was unreachable by construction.

The fix is to declare those handlers ``def``. FastAPI then runs them in its threadpool, which
is exactly what blocking code needs. None of the 24 handlers used ``await``, so nothing was
gained by the ``async`` keyword in the first place.

This test encodes the rule rather than the fix, so a genuinely async handler is still allowed:
**a route may be a coroutine function only if it actually awaits something.**
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
from fastapi.routing import APIRoute

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app  # noqa: E402

# Handlers that do blocking work and must never go back on the event loop.
# Paths as declared on the routers, i.e. without the "/api" prefix app.py adds.
BLOCKING_ENDPOINTS = [
    "/scan",
    "/backtest/{symbol}",
    "/scan/{symbol}",
    "/vix",
]


def _routes():
    """Collect every APIRoute, including ones behind included routers.

    FastAPI versions differ here: some flatten included routers into ``app.routes``, others
    (the version pinned here) wrap them in a ``_IncludedRouter`` that exposes the original
    router instead. Handle both, and assert non-emptiness in a dedicated test so a future
    version change surfaces as a failure rather than a silent pass.
    """
    found = []
    for entry in app.routes:
        if isinstance(entry, APIRoute):
            found.append(entry)
            continue
        inner = getattr(entry, "original_router", None) or getattr(entry, "app", None)
        for sub in getattr(inner, "routes", []):
            if isinstance(sub, APIRoute):
                found.append(sub)
    return found


def test_the_app_actually_has_routes():
    """Without this, every assertion below would pass vacuously."""
    assert len(_routes()) >= 20, (
        "found almost no routes -- the traversal in _routes() has probably broken against a "
        "new FastAPI version, which would make every check below vacuous")


def test_no_route_is_async_without_awaiting():
    """A coroutine handler that never awaits blocks the loop for no benefit."""
    offenders = []
    for route in _routes():
        fn = route.endpoint
        if not inspect.iscoroutinefunction(fn):
            continue
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):  # pragma: no cover - source always available here
            src = ""
        if "await " not in src:
            offenders.append(f"{route.path} -> {fn.__name__}")
    assert not offenders, (
        "These handlers are 'async def' but never await. FastAPI runs them on the event "
        "loop, so any blocking call inside freezes the whole server. Declare them 'def' so "
        "they run in the threadpool:\n  " + "\n  ".join(offenders))


@pytest.mark.parametrize("path", BLOCKING_ENDPOINTS)
def test_known_blocking_endpoints_run_in_the_threadpool(path):
    """These do network I/O, HMM fitting, or both. They must be sync handlers."""
    match = [r for r in _routes() if r.path == path]
    assert match, f"route {path} not found -- was it renamed? update this test"
    for route in match:
        assert not inspect.iscoroutinefunction(route.endpoint), (
            f"{path} is a coroutine function. It performs blocking work, so declaring it "
            "'async def' freezes the event loop for the duration of every request.")


def test_scan_status_is_reachable_while_a_scan_holds_the_threadpool():
    """The regression that motivated all of this.

    ``/api/scan/status`` is only useful if it can answer *during* a scan. It cannot be a
    coroutine (it would be fine on its own, but the scan handler must not block the loop),
    and the scan handler must be sync so the two can overlap.
    """
    status = [r for r in _routes() if r.path == "/scan/status"]
    scan = [r for r in _routes() if r.path == "/scan"]
    assert status and scan
    assert not inspect.iscoroutinefunction(scan[0].endpoint), (
        "the scan handler blocks; if it runs on the event loop, no status poll can be "
        "served while it runs, and the progress indicator can only ever report 'done'")


def test_streaming_scan_uses_a_sync_generator():
    """Starlette iterates sync generators in a threadpool, so SSE was already safe.

    Recorded so nobody 'fixes' it into an async generator that then blocks on CPU work.
    """
    from api import routes_scan

    src = inspect.getsource(routes_scan.run_scan_stream)
    assert "def generate():" in src, "expected a sync generator body"
    assert "async def generate(" not in src, (
        "an async generator would be iterated on the event loop, reintroducing the freeze "
        "for every yielded chunk")
