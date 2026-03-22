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
