"""The scan cache is shared mutable state read by three modules from the AnyIO
threadpool. These tests encode the rules that keep it consistent.

Before the event-loop fix (PR #33) every handler ran *on* the loop, so the loop itself
serialized all access and none of this could go wrong. Making the handlers threadpooled
made the races real, which is why this file exists.

The rules:
  1. "scanning" is always released -- on success, on exception, and on client disconnect.
  2. A finished scan publishes all of its fields together, never piecemeal.
  3. An older scan never overwrites a newer one.
  4. Nothing outside the locked helpers touches _scan_cache.
"""

import ast
import asyncio
import gc
import pathlib
import threading
import time

import pytest

from api import routes_scan as rs
from api.routes_scan import ScanRequest


HELPERS = {"_begin_scan", "_publish_scan", "_end_scan", "_cache_snapshot", "cached_results_full"}


@pytest.fixture(autouse=True)
def _restore_cache():
    """Every test gets a clean cache and leaves the module as it found it."""
    with rs._scan_lock:
        saved = dict(rs._scan_cache)
        saved_gen = rs._scan_generation
        rs._scan_cache.clear()
        rs._scan_cache.update({"results": [], "timestamp": None, "status": "idle"})
    yield
    with rs._scan_lock:
        rs._scan_cache.clear()
        rs._scan_cache.update(saved)
        rs._scan_generation = saved_gen


def _status():
    return rs._cache_snapshot()["status"]


# --- rule 1: "scanning" is always released ----------------------------------------

def test_failed_scan_releases_the_scanning_status(monkeypatch):
    """The bug: status was set to "scanning" with no try/finally, so one exception
    pinned it there for the life of the process and the progress UI never recovered."""
    def boom(**kwargs):
        raise RuntimeError("upstream died")

    monkeypatch.setattr(rs, "scan_watchlist", boom)

    with pytest.raises(RuntimeError):
        rs.run_scan(ScanRequest(custom_tickers="SPY"))

    assert _status() == "idle", "a failed scan left the status pinned at scanning"


def test_empty_watchlist_releases_the_scanning_status():
    req = ScanRequest(watchlist="does-not-exist", custom_tickers="")
    out = rs.run_scan(req)
    assert "error" in out
    assert _status() == "idle"


def test_successful_scan_ends_in_done(monkeypatch):
    monkeypatch.setattr(rs, "scan_watchlist", lambda **kw: [])
    rs.run_scan(ScanRequest(custom_tickers="SPY"))
    assert _status() == "done"


def _fake_ticker(monkeypatch):
    monkeypatch.setattr(rs, "_IS_CLOUD", True)  # threads, not processes
    monkeypatch.setattr(
        rs, "_scan_ticker_light",
        lambda args: {"symbol": args[0], "regime_id": 1, "signal": "HOLD", "price": 1.0},
    )


def _drain(body_iterator):
    """StreamingResponse wraps a sync generator in iterate_in_threadpool, so the events
    are only reachable through an event loop."""
    async def go():
        return [chunk async for chunk in body_iterator]
    return asyncio.run(go())


def test_closing_the_stream_releases_the_status(monkeypatch):
    """Closing the generator must release the status rather than leave it "scanning".

    Be precise about what this covers, because the obvious reading is wrong. Starlette's
    iterate_in_threadpool forwards aclose() to the *async* wrapper, not into our sync
    generator, so the finally only runs when the sync generator is finalized -- which the
    gc.collect() below forces. Against a live server a disconnecting client does not take
    this path at all: uvicorn keeps pulling the generator until the scan finishes and it
    publishes "done" normally (killed client 3s in, resolved at +16s on a cold pool and
    +15s warm, 2 of 2). So this asserts the finally is wired up, not that disconnects
    would otherwise strand the UI. They do not."""
    _fake_ticker(monkeypatch)

    resp = rs.run_scan_stream(ScanRequest(custom_tickers="SPY,QQQ,NVDA"))
    body = resp.body_iterator

    async def start_then_abandon():
        first = await body.__anext__()      # scan is now running
        assert "data:" in first
        assert _status() == "scanning", "status should report scanning while streaming"
        await body.aclose()                 # the client goes away

    asyncio.run(start_then_abandon())

    del body, resp
    gc.collect()

    assert _status() != "scanning", "a disconnected client left the status wedged"


def test_an_abandoned_scan_is_reported_stale_not_scanning():
    """The backstop for a hang no finally block can see. Its justification is one
    unreproduced live observation -- a stream that sat at "scanning" for 5+ minutes with
    live workers and never published -- plus the fact that nothing in the fetch path has
    a timeout. Not a demonstrated bug; a bounded failure mode."""
    rs._begin_scan()
    assert rs.scan_status()["status"] == "scanning"

    with rs._scan_lock:  # pretend the scan began before the deadline
        rs._scan_cache["started_at"] = time.time() - rs._STALE_SCAN_SECONDS - 1

    out = rs.scan_status()
    assert out["status"] == "idle", "an abandoned scan still strands the UI on scanning"
    assert out["stale_scan"] is True


def test_a_running_scan_is_not_reported_stale():
    rs._begin_scan()
    out = rs.scan_status()
    assert out["status"] == "scanning"
    assert "stale_scan" not in out, "a live scan was written off as abandoned"


def test_a_stale_scan_can_still_publish_if_it_survives():
    """The deadline only changes what readers are told. If the scan is genuinely alive it
    keeps its token, so its results are not thrown away."""
    gen = rs._begin_scan()
    with rs._scan_lock:
        rs._scan_cache["started_at"] = time.time() - rs._STALE_SCAN_SECONDS - 1
    assert rs.scan_status()["status"] == "idle"
    assert rs._publish_scan(gen, results=[{"symbol": "SPY"}], status="done", elapsed=1.0) is True
    assert rs.scan_status()["status"] == "done"


def test_stream_reports_scanning_then_done(monkeypatch):
    _fake_ticker(monkeypatch)
    resp = rs.run_scan_stream(ScanRequest(custom_tickers="SPY,QQQ"))
    events = _drain(resp.body_iterator)
    assert any('"type": "done"' in e for e in events)
    assert _status() == "done"


# --- rule 2: publication is atomic -----------------------------------------------

def test_publish_sets_every_field_together():
    gen = rs._begin_scan()
    assert rs._publish_scan(
        gen, results_full=[{"symbol": "SPY"}], results=[{"symbol": "SPY"}],
        timestamp=1.0, status="done", elapsed=2.5,
    ) is True
    snap = rs._cache_snapshot()
    assert snap["status"] == "done"
    assert snap["elapsed"] == 2.5
    assert snap["timestamp"] == 1.0
    assert len(snap["results"]) == 1


def _watch_for_torn_reads(publish_one, rounds=40):
    """Hammer the cache from a reader thread while `publish_one(n)` publishes scans, and
    return every inconsistency seen. `elapsed` doubles as the expected row count, so any
    reader that sees status="done" with a mismatched pair has caught a torn publish."""
    stop = threading.Event()
    violations = []

    def reader():
        while not stop.is_set():
            snap = rs._cache_snapshot()
            if snap.get("status") == "done":
                if snap.get("elapsed") is None or snap.get("timestamp") is None:
                    violations.append(("missing field", snap.get("elapsed"), snap.get("timestamp")))
                elif len(snap.get("results", [])) != snap.get("elapsed"):
                    violations.append(("mismatch", len(snap.get("results", [])), snap.get("elapsed")))

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        for n in range(1, rounds):
            publish_one(n)
    finally:
        stop.set()
        t.join(timeout=5)
    return violations


def test_the_torn_read_detector_actually_detects_tearing():
    """Positive control, and it is here because it is needed: with the real locked
    publisher the window between key writes is so short that the reader below almost
    never lands inside it, so a passing negative test proves very little on its own.
    This deliberately reintroduces the old key-by-key publish -- exactly the shape the
    code had before this fix -- and asserts the detector fires. If this ever stops
    failing, the test after it has become decorative."""
    def torn_publish(n):
        rows = [{"symbol": f"T{i}"} for i in range(n)]
        # the original assignment order, unlocked
        rs._scan_cache["results_full"] = rows
        rs._scan_cache["results"] = rows
        rs._scan_cache["timestamp"] = time.time()
        rs._scan_cache["status"] = "done"
        time.sleep(0.002)              # the gap a real reader slips into
        rs._scan_cache["elapsed"] = n

    violations = _watch_for_torn_reads(torn_publish, rounds=15)
    assert violations, "the detector cannot see a torn publish, so it guarantees nothing"


def test_readers_never_see_done_beside_stale_results():
    """The torn read this fix exists to prevent: status was assigned *before* elapsed and
    *after* results, so a reader could observe "done" next to the previous scan's
    numbers. Sensitivity is established by the positive control above."""
    def publish_one(n):
        gen = rs._begin_scan()
        rows = [{"symbol": f"T{i}"} for i in range(n)]
        rs._publish_scan(
            gen, results_full=rows, results=rows,
            timestamp=time.time(), status="done", elapsed=n,
        )

    violations = _watch_for_torn_reads(publish_one)
    assert not violations, f"observed a half-published scan: {violations[:3]}"


def test_concurrent_readers_never_raise(monkeypatch):
    """scan_status read _scan_cache["status"] by subscript and run_scan read
    _scan_cache["elapsed"] the same way -- both KeyError paths under concurrency."""
    errors = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                rs.scan_status()
                rs.get_cached()
                rs.cached_results_full()
            except Exception as exc:  # noqa: BLE001 - the assertion is that none occur
                errors.append(exc)
                return

    monkeypatch.setattr(rs, "scan_watchlist", lambda **kw: [])
    threads = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for _ in range(20):
            rs.run_scan(ScanRequest(custom_tickers="SPY"))
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=5)

    assert not errors, f"reader raised under concurrency: {errors[0]!r}"


def test_run_scan_reports_elapsed_without_reading_the_cache(monkeypatch):
    """The old code returned _scan_cache["elapsed"], which raises KeyError whenever
    publication was skipped. elapsed is a local now."""
    monkeypatch.setattr(rs, "scan_watchlist", lambda **kw: [])
    monkeypatch.setattr(rs, "_publish_scan", lambda *a, **kw: False)  # superseded
    out = rs.run_scan(ScanRequest(custom_tickers="SPY"))
    assert isinstance(out["summary"]["elapsed"], float)
    assert "elapsed" not in rs._cache_snapshot()


# --- rule 3: an older scan never overwrites a newer one ---------------------------

def test_superseded_scan_does_not_publish():
    first = rs._begin_scan()
    second = rs._begin_scan()
    assert second != first

    assert rs._publish_scan(first, results=[{"symbol": "STALE"}], status="done") is False
    assert rs._cache_snapshot()["status"] == "scanning", "stale scan published over a running one"
    assert rs._cache_snapshot()["results"] == []

    assert rs._publish_scan(second, results=[{"symbol": "FRESH"}], status="done") is True
    assert rs._cache_snapshot()["results"] == [{"symbol": "FRESH"}]


def test_superseded_scan_does_not_reset_a_running_status():
    first = rs._begin_scan()
    second = rs._begin_scan()
    rs._end_scan(first)  # the older scan unwinds
    assert _status() == "scanning", "an older scan cleared the newer scan's status"
    rs._end_scan(second)
    assert _status() == "idle"


def test_end_scan_leaves_a_published_result_alone():
    gen = rs._begin_scan()
    rs._publish_scan(gen, status="done", elapsed=1.0)
    rs._end_scan(gen)
    assert _status() == "done", "the finally block clobbered a completed scan"


def test_overlapping_scans_leave_a_consistent_cache(monkeypatch):
    """Two real scans overlapping: whichever starts last owns the cache, and the
    result set must be one scan's output rather than a mixture of both."""
    def slow(**kwargs):
        time.sleep(0.05)
        return [{"symbol": s, "regime_id": 1, "price": 1.0} for s in kwargs["symbols"]]

    monkeypatch.setattr(rs, "scan_watchlist", slow)
    threads = [
        threading.Thread(target=rs.run_scan, args=(ScanRequest(custom_tickers=t),), daemon=True)
        for t in ("SPY,QQQ", "NVDA,AAPL,XLF")
    ]
    for t in threads:
        t.start()
        time.sleep(0.01)
    for t in threads:
        t.join(timeout=10)

    snap = rs._cache_snapshot()
    assert snap["status"] == "done"
    symbols = {r["symbol"] for r in snap["results"]}
    assert symbols in ({"SPY", "QQQ"}, {"NVDA", "AAPL", "XLF"}), f"mixed two scans: {symbols}"
    assert len(snap["results"]) == len(snap["results_full"])


# --- rule 4: nothing bypasses the lock -------------------------------------------

def _api_sources():
    root = pathlib.Path(__file__).resolve().parent.parent / "api"
    return sorted(root.glob("*.py"))


def test_the_guard_can_see_the_api_package():
    """Non-vacuity: if this file can't find the sources, every rule below passes for
    the wrong reason."""
    sources = _api_sources()
    assert len(sources) >= 4
    assert any(p.name == "routes_scan.py" for p in sources)


def test_no_unlocked_scan_cache_access_anywhere():
    """_scan_cache may only be touched inside the helpers that hold _scan_lock."""
    offenders = []
    for path in _api_sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name in HELPERS:
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id == "_scan_cache":
                    offenders.append(f"{path.name}:{node.name}:{sub.lineno}")
    assert not offenders, (
        "these read or write _scan_cache outside the locked helpers: " + ", ".join(offenders)
    )


def test_every_helper_holds_the_lock():
    src = (pathlib.Path(__file__).resolve().parent.parent / "api" / "routes_scan.py").read_text()
    tree = ast.parse(src)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in HELPERS:
            found.add(node.name)
            locked = any(
                isinstance(sub, ast.With) and any(
                    isinstance(item.context_expr, ast.Name) and item.context_expr.id == "_scan_lock"
                    for item in sub.items
                )
                for sub in ast.walk(node)
            )
            assert locked, f"{node.name} touches the cache without holding _scan_lock"
    assert found == HELPERS, f"missing helpers: {HELPERS - found}"


def test_scan_entry_points_use_try_finally():
    """run_scan and the SSE generate() must release the status in a finally."""
    src = (pathlib.Path(__file__).resolve().parent.parent / "api" / "routes_scan.py").read_text()
    tree = ast.parse(src)
    checked = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in {"run_scan", "generate"}:
            has = any(
                isinstance(sub, ast.Try) and sub.finalbody
                and any("_end_scan" in ast.dump(f) for f in sub.finalbody)
                for sub in ast.walk(node)
            )
            assert has, f"{node.name} does not release the status in a finally block"
            checked.add(node.name)
    assert checked == {"run_scan", "generate"}, f"did not find both entry points: {checked}"


# --- copy semantics --------------------------------------------------------------

def test_cached_results_full_returns_a_detached_list():
    gen = rs._begin_scan()
    rs._publish_scan(gen, results_full=[{"symbol": "SPY"}], status="done")
    got = rs.cached_results_full()
    got.append({"symbol": "INJECTED"})
    assert len(rs.cached_results_full()) == 1, "caller mutated the cache through the copy"


def test_cached_results_full_is_empty_before_any_scan():
    assert rs.cached_results_full() == []


def test_snapshot_is_detached():
    snap = rs._cache_snapshot()
    snap["status"] = "tampered"
    assert _status() == "idle"
