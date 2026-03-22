"""
tradier_broker.py — Tradier Brokerage Integration
Place orders (equity + options) through Tradier API from the dashboard.
Uses environment variables for credentials, with fallback to settings file.
"""

import os
import json
import requests
from typing import Optional, Dict, List

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), ".tradier_settings.json")

# Tradier API endpoints
SANDBOX_BASE = "https://sandbox.tradier.com/v1"
PROD_BASE = "https://api.tradier.com/v1"


def _load_config() -> Dict:
    """Load Tradier config from settings file or env vars."""
    config = {
        "access_token": os.environ.get("TRADIER_ACCESS_TOKEN", ""),
        "account_id": os.environ.get("TRADIER_ACCOUNT_ID", ""),
        "sandbox": True,
    }

    # Try settings file
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
            config.update({k: v for k, v in saved.items() if v})
        except Exception:
            pass

    return config


def save_config(access_token: str, account_id: str, sandbox: bool = True):
    """Save Tradier credentials to local settings file."""
    config = {
        "access_token": access_token,
        "account_id": account_id,
        "sandbox": sandbox,
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(config, f, indent=2)


def _get_headers(config: Dict) -> Dict:
    return {
        "Authorization": f"Bearer {config['access_token']}",
        "Accept": "application/json",
    }


def _base_url(config: Dict) -> str:
    return SANDBOX_BASE if config.get("sandbox", True) else PROD_BASE


def is_configured() -> bool:
    """Check if Tradier credentials are set."""
    config = _load_config()
    return bool(config.get("access_token")) and bool(config.get("account_id"))


def get_account_info() -> Dict:
    """Get account balances and info."""
    config = _load_config()
    if not config["access_token"]:
        return {"error": "Tradier not configured"}

    try:
        r = requests.get(
            f"{_base_url(config)}/accounts/{config['account_id']}/balances",
            headers=_get_headers(config),
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        bal = data.get("balances", {})
        return {
            "account_id": config["account_id"],
            "total_equity": bal.get("total_equity", 0),
            "total_cash": bal.get("total_cash", 0),
            "option_buying_power": bal.get("margin", {}).get("option_buying_power", 0),
            "stock_buying_power": bal.get("margin", {}).get("stock_buying_power", 0),
            "open_pl": bal.get("open_pl", 0),
            "pending_orders": bal.get("pending_orders_count", 0),
            "sandbox": config.get("sandbox", True),
        }
    except Exception as e:
        return {"error": str(e)}


def get_positions() -> List[Dict]:
    """Get current open positions."""
    config = _load_config()
    if not config["access_token"]:
        return []

    try:
        r = requests.get(
            f"{_base_url(config)}/accounts/{config['account_id']}/positions",
            headers=_get_headers(config),
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        positions = data.get("positions", {})
        if positions == "null" or not positions:
            return []
        pos_list = positions.get("position", [])
        if isinstance(pos_list, dict):
            pos_list = [pos_list]
        return pos_list
    except Exception:
        return []


def get_orders(status: str = "pending") -> List[Dict]:
    """Get orders, optionally filtered by status."""
    config = _load_config()
    if not config["access_token"]:
        return []

    try:
        r = requests.get(
            f"{_base_url(config)}/accounts/{config['account_id']}/orders",
            headers=_get_headers(config),
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        orders = data.get("orders", {})
        if orders == "null" or not orders:
            return []
        order_list = orders.get("order", [])
        if isinstance(order_list, dict):
            order_list = [order_list]
        if status != "all":
            order_list = [o for o in order_list if o.get("status") == status]
        return order_list
    except Exception:
        return []


def place_equity_order(
    symbol: str,
    side: str,
    quantity: int,
    order_type: str = "market",
    limit_price: float = None,
    duration: str = "day",
    preview: bool = True,
) -> Dict:
    """
    Place an equity (stock/ETF) order.

    side: "buy" or "sell"
    order_type: "market", "limit", "stop", "stop_limit"
    duration: "day" or "gtc"
    preview: if True, previews without executing
    """
    config = _load_config()
    if not config["access_token"]:
        return {"error": "Tradier not configured"}

    data = {
        "class": "equity",
        "symbol": symbol.upper(),
        "side": side,
        "quantity": str(quantity),
        "type": order_type,
        "duration": duration,
    }
    if order_type == "limit" and limit_price:
        data["price"] = str(limit_price)

    if preview:
        data["preview"] = "true"

    try:
        r = requests.post(
            f"{_base_url(config)}/accounts/{config['account_id']}/orders",
            headers=_get_headers(config),
            data=data,
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def place_option_order(
    symbol: str,
    option_symbol: str,
    side: str,
    quantity: int,
    order_type: str = "market",
    limit_price: float = None,
    duration: str = "day",
    preview: bool = True,
) -> Dict:
    """
    Place an options order.

    symbol: underlying (e.g. "SPY")
    option_symbol: OCC symbol (e.g. "SPY260417C00650000")
    side: "buy_to_open", "buy_to_close", "sell_to_open", "sell_to_close"
    order_type: "market", "limit"
    duration: "day" or "gtc"
    preview: if True, previews without executing
    """
    config = _load_config()
    if not config["access_token"]:
        return {"error": "Tradier not configured"}

    data = {
        "class": "option",
        "symbol": symbol.upper(),
        "option_symbol": option_symbol,
        "side": side,
        "quantity": str(quantity),
        "type": order_type,
        "duration": duration,
    }
    if order_type == "limit" and limit_price:
        data["price"] = str(limit_price)

    if preview:
        data["preview"] = "true"

    try:
        r = requests.post(
            f"{_base_url(config)}/accounts/{config['account_id']}/orders",
            headers=_get_headers(config),
            data=data,
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def cancel_order(order_id: str) -> Dict:
    """Cancel a pending order."""
    config = _load_config()
    if not config["access_token"]:
        return {"error": "Tradier not configured"}

    try:
        r = requests.delete(
            f"{_base_url(config)}/accounts/{config['account_id']}/orders/{order_id}",
            headers=_get_headers(config),
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}
