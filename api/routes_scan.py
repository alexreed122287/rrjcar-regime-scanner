"""API routes for regime scanning."""

import os
import json
import concurrent.futures
import threading
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import time

from screener import scan_watchlist, WATCHLISTS, scan_single_ticker
from hmm_engine import REGIME_LABELS
import yfinance as yf
import logging

from api.errors import error_response

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory cache for scan results (single-user dashboard).
#
# Handlers run in the AnyIO threadpool, so several of them touch this dict at the same
# time. Never read or write it directly -- go through the helpers below, which hold
# _scan_lock. Reading it key-by-key can observe a half-published scan: status "done"
# alongside the *previous* scan's results.
_scan_lock = threading.RLock()
_scan_cache: Dict[str, Any] = {"results": [], "timestamp": None, "status": "idle"}

# Monotonic scan id. Two overlapping scans must not clobber each other: the older one
# is superseded and its results are dropped rather than published over the newer.
_scan_generation = 0

# A "scanning" status older than this is reported as idle.
#
# Scope honestly: this is a backstop, not a fix for a demonstrated bug. Disconnects were
# measured and they do NOT strand the status -- an abandoned SSE stream keeps running and
# publishes normally (killed client 3s in: "done" at +16s cold pool, +15s warm, 2 of 2).
# What justifies the deadline is a single unreproduced observation: one stream sat at
# "scanning" for 5+ minutes with its workers alive and never published. The cause was
# never identified, and nothing in the request path has a timeout, so a stalled upstream
# fetch can strand the status indefinitely. The deadline bounds that failure instead of
# guessing at its mechanism. Generous on purpose: a real scan must never trip it.
_STALE_SCAN_SECONDS = 1800


def _begin_scan() -> int:
    """Mark a scan as running and return its generation token."""
    global _scan_generation
    with _scan_lock:
        _scan_generation += 1
        _scan_cache["status"] = "scanning"
        _scan_cache["started_at"] = time.time()
        return _scan_generation


def _publish_scan(generation: int, **fields: Any) -> bool:
    """Atomically publish a finished scan. All fields land together or none do, so no
    reader can see status="done" next to stale results. Returns False if a newer scan
    started meanwhile, in which case this scan's results are discarded."""
    with _scan_lock:
        if generation != _scan_generation:
            return False
        _scan_cache.update(fields)
        return True


def _end_scan(generation: int) -> None:
    """Release the "scanning" status if this scan still owns it. Called from a finally
    block: without it, any exception out of scan_watchlist -- one bad ticker, a provider
    outage -- leaves status pinned at "scanning" for the life of the process, since the
    state is module-level and no page reload can clear it."""
    with _scan_lock:
        if generation == _scan_generation and _scan_cache.get("status") == "scanning":
            _scan_cache["status"] = "idle"


def _cache_snapshot() -> Dict[str, Any]:
    """A consistent shallow copy of the cache. Shallow is deliberate: result dicts are
    treated as read-only once published, and results_full holds DataFrames that are far
    too expensive to copy per request.

    A scan still marked "scanning" past _STALE_SCAN_SECONDS is reported as "idle" with
    stale_scan set, so an abandoned stream cannot strand the UI. The cache itself is left
    untouched -- if that scan is somehow still alive it keeps its generation token and can
    still publish."""
    with _scan_lock:
        snap = dict(_scan_cache)
    started = snap.get("started_at")
    if snap.get("status") == "scanning" and started is not None:
        if time.time() - started > _STALE_SCAN_SECONDS:
            snap["status"] = "idle"
            snap["stale_scan"] = True
    return snap


def cached_results_full() -> List[Dict[str, Any]]:
    """Full results of the last completed scan, or [] if there is none. The list is
    copied so callers can iterate it while a scan republishes; the dicts inside are
    shared and must not be mutated."""
    with _scan_lock:
        return list(_scan_cache.get("results_full") or [])

# Detect constrained environments (Render free = 0.1 CPU)
_IS_CLOUD = bool(os.environ.get("RENDER") or os.environ.get("PORT"))
_DEFAULT_WORKERS = 2 if _IS_CLOUD else 6


class ScanRequest(BaseModel):
    watchlist: str = "Technology — Mag 7"
    custom_tickers: str = ""
    strategy: str = "v2"
    n_regimes: int = 5 if _IS_CLOUD else 7
    min_confs: int = 6
    regime_confirm: int = 2
    max_workers: int = _DEFAULT_WORKERS
    bullish_only: bool = False
    period_days: int = 365 if _IS_CLOUD else 730


def _serialize_result(r: dict) -> dict:
    """Strip non-serializable fields from a scan result."""
    out = {}
    for k, v in r.items():
        if k.startswith("_"):
            continue
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        elif hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _serialize_drilldown(r: dict) -> dict:
    """Serialize a scan result including chart data for drill-down."""
    base = _serialize_result(r)
    regime_df = r.get("_regime_df")
    if regime_df is not None:
        base["chart_data"] = {
            "dates": [str(d.date()) if hasattr(d, "date") else str(d) for d in regime_df.index],
            "close": regime_df["Close"].round(2).tolist(),
            "regime_ids": regime_df["regime_id"].astype(int).tolist(),
        }
    return base


@router.get("/watchlists")
def get_watchlists():
    return {name: tickers for name, tickers in WATCHLISTS.items()
            if name not in ("ALL TICKERS", "All Stocks (no ETFs)", "All ETFs")}


@router.get("/watchlists/all")
def get_all_watchlists():
    return {name: tickers for name, tickers in WATCHLISTS.items()}


@router.post("/scan")
def run_scan(req: ScanRequest):
    generation = _begin_scan()
    try:
        # Determine tickers
        if req.custom_tickers.strip():
            symbols = [t.strip().upper() for t in req.custom_tickers.split(",") if t.strip()]
        else:
            symbols = WATCHLISTS.get(req.watchlist, [])

        if not symbols:
            return {"error": "No tickers to scan", "results": []}

        # Cap workers on constrained environments
        workers = min(req.max_workers, _DEFAULT_WORKERS) if _IS_CLOUD else req.max_workers

        start = time.time()
        results = scan_watchlist(
            symbols=symbols,
            strategy=req.strategy,
            n_regimes=req.n_regimes,
            min_confirmations=req.min_confs,
            regime_confirm_bars=req.regime_confirm,
            max_workers=workers,
            bullish_only=req.bullish_only,
            period_days=req.period_days,
        )

        # Publish in one shot (results_full keeps _regime_df for drill-down)
        serialized = [_serialize_result(r) for r in results]
        elapsed = round(time.time() - start, 1)
        _publish_scan(
            generation,
            results_full=results,
            results=serialized,
            timestamp=time.time(),
            status="done",
            elapsed=elapsed,
        )

        # Summary counts
        bulls = sum(1 for r in results if r.get("regime_id") is not None and r["regime_id"] <= 2)
        bears = sum(1 for r in results if r.get("regime_id") is not None and r["regime_id"] >= 5)
        neutrals = sum(1 for r in results if r.get("regime_id") is not None and 3 <= r["regime_id"] <= 4)
        entries = sum(1 for r in results if "ENTER" in (r.get("signal") or ""))
        exits = sum(1 for r in results if "EXIT" in (r.get("signal") or ""))
        errors = sum(1 for r in results if r.get("error") and r.get("price") is None)

        # elapsed comes from the local, not the cache: a newer scan may have superseded
        # this one, and the old code raised KeyError here if publication never happened.
        return {
            "results": serialized,
            "summary": {
                "total": len(results),
                "bullish": bulls,
                "bearish": bears,
                "neutral": neutrals,
                "entries": entries,
                "exits": exits,
                "errors": errors,
                "elapsed": elapsed,
            },
            "regime_labels": REGIME_LABELS,
        }
    finally:
        _end_scan(generation)


@router.get("/scan/status")
def scan_status():
    snap = _cache_snapshot()
    out = {
        "status": snap["status"],
        "count": len(snap.get("results", [])),
        "timestamp": snap.get("timestamp"),
    }
    if snap.get("stale_scan"):
        out["stale_scan"] = True
    return out


@router.get("/scan/cached")
def get_cached():
    snap = _cache_snapshot()
    return {
        "results": snap.get("results", []),
        "timestamp": snap.get("timestamp"),
        "status": snap["status"],
    }


def _scan_ticker_light(args):
    """Wrapper for ProcessPoolExecutor — scan one ticker, return serializable result."""
    sym, strategy, n_regimes, min_confs, regime_confirm, period_days = args
    result = scan_single_ticker(
        sym, strategy=strategy, n_regimes=n_regimes,
        min_confirmations=min_confs,
        regime_confirm_bars=regime_confirm,
        period_days=period_days,
    )
    if result is None:
        return None
    # Strip heavy non-picklable objects before crossing process boundary
    return {k: v for k, v in result.items() if not k.startswith("_")}


@router.post("/scan/stream")
def run_scan_stream(req: ScanRequest):
    """Stream scan results with concurrent process workers via SSE."""
    if req.custom_tickers.strip():
        symbols = [t.strip().upper() for t in req.custom_tickers.split(",") if t.strip()]
    else:
        symbols = WATCHLISTS.get(req.watchlist, [])

    if not symbols:
        return {"error": "No tickers to scan", "results": []}

    workers = min(req.max_workers, _DEFAULT_WORKERS) if _IS_CLOUD else req.max_workers

    def _emit(generation: int):
        all_results = []
        done_count = 0
        total = len(symbols)
        start = time.time()

        # Build args for each ticker
        args_list = [
            (sym, req.strategy, req.n_regimes, req.min_confs,
             req.regime_confirm, req.period_days)
            for sym in symbols
        ]

        # Use threads on cloud (process spawn too expensive on 0.1 CPU)
        # Use processes on desktop for true parallelism
        chunk_size = workers * 6
        PoolClass = concurrent.futures.ThreadPoolExecutor if _IS_CLOUD else concurrent.futures.ProcessPoolExecutor
        with PoolClass(max_workers=workers) as executor:
            for chunk_start in range(0, total, chunk_size):
                chunk = args_list[chunk_start:chunk_start + chunk_size]
                futures = {executor.submit(_scan_ticker_light, a): a[0] for a in chunk}

                for future in concurrent.futures.as_completed(futures):
                    done_count += 1
                    try:
                        result = future.result()
                    except Exception:
                        result = None

                    if result:
                        all_results.append(result)
                        serialized = _serialize_result(result)
                        msg = json.dumps({
                            "type": "result",
                            "data": serialized,
                            "progress": {"done": done_count, "total": total},
                        })
                        yield f"data: {msg}\n\n"
                    else:
                        msg = json.dumps({
                            "type": "progress",
                            "progress": {"done": done_count, "total": total},
                        })
                        yield f"data: {msg}\n\n"

        # Final summary -- published in one shot so no reader sees "done" beside stale
        # results, and skipped entirely if a newer scan has superseded this one.
        elapsed = round(time.time() - start, 1)
        _publish_scan(
            generation,
            results_full=all_results,
            results=[_serialize_result(r) for r in all_results],
            timestamp=time.time(),
            status="done",
            elapsed=elapsed,
        )

        bulls = sum(1 for r in all_results if r.get("regime_id") is not None and r["regime_id"] <= 2)
        bears = sum(1 for r in all_results if r.get("regime_id") is not None and r["regime_id"] >= 5)
        neutrals = sum(1 for r in all_results if r.get("regime_id") is not None and 3 <= r["regime_id"] <= 4)
        entries = sum(1 for r in all_results if "ENTER" in (r.get("signal") or ""))
        exits = sum(1 for r in all_results if "EXIT" in (r.get("signal") or ""))

        # Count hits per signal type
        signal_counts = {}
        for r in all_results:
            sig = r.get("signal") or "UNKNOWN"
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

        # Count hits per individual confirmation (across all scanned tickers)
        confirmation_counts = {}
        for r in all_results:
            detail = r.get("confirmation_detail") or {}
            for name, passed in detail.items():
                if name not in confirmation_counts:
                    confirmation_counts[name] = {"pass": 0, "fail": 0}
                if passed:
                    confirmation_counts[name]["pass"] += 1
                else:
                    confirmation_counts[name]["fail"] += 1

        # Log scanner results
        logger.info(f"Scan complete: {len(all_results)} tickers in {elapsed}s")
        logger.info(f"Signal counts: {signal_counts}")
        logger.info(f"Confirmation hit rates: { {k: v['pass'] for k, v in confirmation_counts.items()} }")

        summary = json.dumps({
            "type": "done",
            "summary": {
                "total": len(all_results), "bullish": bulls, "bearish": bears,
                "neutral": neutrals, "entries": entries, "exits": exits,
                "elapsed": elapsed,
                "signal_counts": signal_counts,
                "confirmation_counts": confirmation_counts,
            },
        })
        yield f"data: {summary}\n\n"

    def generate():
        # Same contract as run_scan: whatever unwinds this generator, the status is
        # released. Measured caveat -- a disconnecting client does NOT reliably unwind it
        # (uvicorn keeps pulling until the scan finishes, so it publishes normally), so
        # this finally covers exceptions rather than disconnects.
        generation = _begin_scan()
        try:
            yield from _emit(generation)
        finally:
            _end_scan(generation)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/vix")
def get_vix():
    """Fetch current VIX level."""
    try:
        import pandas as pd
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d", interval="1d")
        if hist is not None and not hist.empty:
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            current = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current
            change = round(current - prev, 2)
            change_pct = round((current - prev) / prev * 100, 2) if prev else 0
            return {"vix": round(current, 2), "change": change, "change_pct": change_pct}
    except Exception as e:
        logger.warning(f"VIX fetch failed: {e}")
    return {"vix": None, "change": None, "change_pct": None}


@router.get("/scan/{symbol}")
def scan_symbol(symbol: str, strategy: str = "v2"):
    """Deep scan a single ticker with full chart data."""
    # Check cache first
    full_results = cached_results_full()
    cached = next((r for r in full_results if r.get("symbol", "").upper() == symbol.upper()), None)
    if cached and cached.get("_regime_df") is not None:
        return _serialize_drilldown(cached)

    # Fresh scan
    result = scan_single_ticker(symbol, strategy=strategy)
    if result is None:
        return error_response(f"Failed to scan {symbol}", 502)
    return _serialize_drilldown(result)
