"""API routes for dashboard settings."""

import os
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from settings_manager import load_settings, save_settings, DEFAULT_SETTINGS

router = APIRouter()


@router.get("/apis")
async def api_status():
    """Show which data APIs are configured and available."""
    from data_loader import _tradier_available
    return {
        "tradier": {
            "configured": _tradier_available(),
            "type": "primary",
            "limits": "Unlimited (sandbox) / 120 req/min (prod)",
            "note": "OHLCV, options chains, account data",
        },
        "yahoo_finance": {
            "configured": True,  # always available via yfinance
            "type": "fallback",
            "limits": "Soft rate limit (~2000 req/hr)",
            "note": "OHLCV, crypto, fundamentals (via yfinance)",
        },
        "alpha_vantage": {
            "configured": bool(os.environ.get("ALPHA_VANTAGE_API_KEY")),
            "type": "fallback",
            "limits": "25 calls/day (free), 75/min (premium)",
            "note": "OHLCV, technicals, fundamentals. Free key: alphavantage.co/support/#api-key",
        },
        "fmp": {
            "configured": bool(os.environ.get("FMP_API_KEY")),
            "type": "fallback",
            "limits": "250 calls/day (free)",
            "note": "OHLCV, financials, SEC filings. Free key: financialmodelingprep.com",
        },
        "twelve_data": {
            "configured": bool(os.environ.get("TWELVE_DATA_API_KEY")),
            "type": "fallback",
            "limits": "800 calls/day, 8/min (free)",
            "note": "OHLCV, technicals, forex, crypto. Free key: twelvedata.com",
        },
        "nasdaq_feeds": {
            "configured": True,  # always available
            "type": "universe",
            "limits": "Unlimited",
            "note": "NYSE + NASDAQ ticker universe (~10K symbols, refreshed daily)",
        },
    }


class SettingsUpdate(BaseModel):
    watchlist: Optional[str] = None
    custom_tickers: Optional[str] = None
    strategy: Optional[str] = None
    min_confs: Optional[int] = None
    regime_confirm: Optional[int] = None
    cooldown: Optional[int] = None
    initial_capital: Optional[float] = None
    n_regimes: Optional[int] = None
    max_workers: Optional[int] = None
    options_enabled: Optional[bool] = None
    min_dte: Optional[int] = None
    max_dte: Optional[int] = None
    top_n_options: Optional[int] = None
    auto_refresh: Optional[bool] = None
    refresh_minutes: Optional[int] = None
    risk_pct: Optional[int] = None
    bullish_only: Optional[bool] = None


@router.get("/settings")
async def get_settings():
    return load_settings()


@router.post("/settings")
async def update_settings(req: SettingsUpdate):
    current = load_settings()
    updates = req.model_dump(exclude_none=True)
    current.update(updates)
    save_settings(current)
    return current
