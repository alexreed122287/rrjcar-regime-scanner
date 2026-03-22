"""API routes for dashboard settings."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from settings_manager import load_settings, save_settings, DEFAULT_SETTINGS

router = APIRouter()


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
