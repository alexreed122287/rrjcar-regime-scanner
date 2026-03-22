"""
settings_manager.py — Persist and load dashboard settings.
Saves to a local JSON file so settings survive between sessions.
"""

import json
import os
from typing import Dict, Any

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), ".dashboard_settings.json")

DEFAULT_SETTINGS = {
    "watchlist": "All Stocks (no ETFs)",
    "custom_tickers": "",
    "strategy": "v2",
    "min_confs": 6,
    "regime_confirm": 2,
    "cooldown": 3,
    "initial_capital": 100_000,
    "n_regimes": 7,
    "max_workers": 6,
    "options_enabled": True,
    "min_dte": 21,
    "max_dte": 45,
    "top_n_options": 3,
    "auto_refresh": False,
    "refresh_minutes": 5,
    "risk_pct": 10,
    # Alert settings
    "alerts_enabled": False,
    "alert_email": "",
    "alert_smtp_server": "smtp.gmail.com",
    "alert_smtp_port": 587,
    "alert_smtp_user": "",
    "alert_smtp_password": "",
    "alert_telegram_enabled": False,
    "alert_telegram_bot_token": "",
    "alert_telegram_chat_id": "",
    "alert_on_regime_change": True,
    "alert_on_bull_entry": True,
    "alert_on_bear_entry": False,
    "alert_min_confirmations": 6,
    # Scheduled scan settings
    "scheduled_scans_enabled": False,
    "scheduled_scan_times": "09:30,12:00,15:30",
    "scheduled_scan_timezone": "America/Chicago",
}


def load_settings() -> Dict[str, Any]:
    """Load saved settings, falling back to defaults."""
    settings = dict(DEFAULT_SETTINGS)
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
            settings.update({k: v for k, v in saved.items() if k in DEFAULT_SETTINGS})
        except Exception:
            pass
    return settings


def save_settings(settings: Dict[str, Any]):
    """Save current settings to disk."""
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


def get_setting(key: str, default=None):
    """Get a single setting value."""
    settings = load_settings()
    return settings.get(key, default)
