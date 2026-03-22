"""API routes for regime scanning."""

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import time

from screener import scan_watchlist, WATCHLISTS, scan_single_ticker
from hmm_engine import REGIME_LABELS

router = APIRouter()

# In-memory cache for scan results (single-user dashboard)
_scan_cache: Dict[str, Any] = {"results": [], "timestamp": None, "status": "idle"}


class ScanRequest(BaseModel):
    watchlist: str = "Mag 7"
    custom_tickers: str = ""
    strategy: str = "v2"
    n_regimes: int = 7
    min_confs: int = 6
    regime_confirm: int = 2
    max_workers: int = 6
    bullish_only: bool = False


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

    start = time.time()
    results = scan_watchlist(
        symbols=symbols,
        strategy=req.strategy,
        n_regimes=req.n_regimes,
        min_confirmations=req.min_confs,
        regime_confirm_bars=req.regime_confirm,
        max_workers=req.max_workers,
        bullish_only=req.bullish_only,
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
