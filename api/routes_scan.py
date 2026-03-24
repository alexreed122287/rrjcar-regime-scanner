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
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

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
async def run_scan_stream(req: ScanRequest):
    """Stream scan results with concurrent process workers via SSE."""
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

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/vix")
async def get_vix():
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
