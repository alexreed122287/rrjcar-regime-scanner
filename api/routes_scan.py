"""API routes for regime scanning."""

import os
import json
import concurrent.futures
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import time

from screener import scan_watchlist, WATCHLISTS, scan_single_ticker
from hmm_engine import REGIME_LABELS

router = APIRouter()

# In-memory cache for scan results (single-user dashboard)
_scan_cache: Dict[str, Any] = {"results": [], "timestamp": None, "status": "idle"}

# Detect constrained environments (Render free = 0.1 CPU)
_IS_CLOUD = bool(os.environ.get("RENDER") or os.environ.get("PORT"))
_DEFAULT_WORKERS = 2 if _IS_CLOUD else 6


class ScanRequest(BaseModel):
    watchlist: str = "Mag 7"
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
async def get_watchlists():
    return {name: tickers for name, tickers in WATCHLISTS.items()
            if name not in ("ALL TICKERS", "All Stocks (no ETFs)", "All ETFs")}


@router.get("/watchlists/all")
async def get_all_watchlists():
    return {name: tickers for name, tickers in WATCHLISTS.items()}


@router.post("/scan")
async def run_scan(req: ScanRequest):
    global _scan_cache
    _scan_cache["status"] = "scanning"

    # Determine tickers
    if req.custom_tickers.strip():
        symbols = [t.strip().upper() for t in req.custom_tickers.split(",") if t.strip()]
    else:
        symbols = WATCHLISTS.get(req.watchlist, [])

    if not symbols:
        _scan_cache["status"] = "idle"
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

    # Cache full results (with _regime_df for drill-down)
    _scan_cache["results_full"] = results
    serialized = [_serialize_result(r) for r in results]
    _scan_cache["results"] = serialized
    _scan_cache["timestamp"] = time.time()
    _scan_cache["status"] = "done"
    _scan_cache["elapsed"] = round(time.time() - start, 1)

    # Summary counts
    bulls = sum(1 for r in results if r.get("regime_id") is not None and r["regime_id"] <= 2)
    bears = sum(1 for r in results if r.get("regime_id") is not None and r["regime_id"] >= 5)
    neutrals = sum(1 for r in results if r.get("regime_id") is not None and 3 <= r["regime_id"] <= 4)
    entries = sum(1 for r in results if "ENTER" in (r.get("signal") or ""))
    exits = sum(1 for r in results if "EXIT" in (r.get("signal") or ""))
    errors = sum(1 for r in results if r.get("error") and r.get("price") is None)

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
            "elapsed": _scan_cache["elapsed"],
        },
        "regime_labels": REGIME_LABELS,
    }


@router.get("/scan/status")
async def scan_status():
    return {
        "status": _scan_cache["status"],
        "count": len(_scan_cache.get("results", [])),
        "timestamp": _scan_cache.get("timestamp"),
    }


@router.get("/scan/cached")
async def get_cached():
    return {
        "results": _scan_cache.get("results", []),
        "timestamp": _scan_cache.get("timestamp"),
        "status": _scan_cache["status"],
    }


@router.post("/scan/stream")
async def run_scan_stream(req: ScanRequest):
    """Stream scan results with concurrent workers via SSE."""
    if req.custom_tickers.strip():
        symbols = [t.strip().upper() for t in req.custom_tickers.split(",") if t.strip()]
    else:
        symbols = WATCHLISTS.get(req.watchlist, [])

    if not symbols:
        return {"error": "No tickers to scan", "results": []}

    workers = min(req.max_workers, _DEFAULT_WORKERS) if _IS_CLOUD else req.max_workers

    def generate():
        global _scan_cache
        _scan_cache["status"] = "scanning"
        all_results = []
        done_count = 0
        total = len(symbols)
        start = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_sym = {
                executor.submit(
                    scan_single_ticker, sym,
                    strategy=req.strategy, n_regimes=req.n_regimes,
                    min_confirmations=req.min_confs,
                    regime_confirm_bars=req.regime_confirm,
                    period_days=req.period_days,
                ): sym
                for sym in symbols
            }

            for future in concurrent.futures.as_completed(future_to_sym):
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
                    # Still send progress for failed tickers
                    msg = json.dumps({
                        "type": "progress",
                        "progress": {"done": done_count, "total": total},
                    })
                    yield f"data: {msg}\n\n"

        # Final summary
        _scan_cache["results_full"] = all_results
        _scan_cache["results"] = [_serialize_result(r) for r in all_results]
        _scan_cache["timestamp"] = time.time()
        _scan_cache["status"] = "done"
        elapsed = round(time.time() - start, 1)

        bulls = sum(1 for r in all_results if r.get("regime_id") is not None and r["regime_id"] <= 2)
        bears = sum(1 for r in all_results if r.get("regime_id") is not None and r["regime_id"] >= 5)
        neutrals = sum(1 for r in all_results if r.get("regime_id") is not None and 3 <= r["regime_id"] <= 4)
        entries = sum(1 for r in all_results if "ENTER" in (r.get("signal") or ""))
        exits = sum(1 for r in all_results if "EXIT" in (r.get("signal") or ""))

        summary = json.dumps({
            "type": "done",
            "summary": {
                "total": len(all_results), "bullish": bulls, "bearish": bears,
                "neutral": neutrals, "entries": entries, "exits": exits,
                "elapsed": elapsed,
            },
        })
        yield f"data: {summary}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/scan/{symbol}")
async def scan_symbol(symbol: str, strategy: str = "v2"):
    """Deep scan a single ticker with full chart data."""
    # Check cache first
    full_results = _scan_cache.get("results_full", [])
    cached = next((r for r in full_results if r.get("symbol", "").upper() == symbol.upper()), None)
    if cached and cached.get("_regime_df") is not None:
        return _serialize_drilldown(cached)

    # Fresh scan
    result = scan_single_ticker(symbol, strategy=strategy)
    if result is None:
        return {"error": f"Failed to scan {symbol}"}
    return _serialize_drilldown(result)
