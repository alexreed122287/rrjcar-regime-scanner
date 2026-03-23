"""API routes for Tradier broker integration."""

import time
import threading
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict, Any

router = APIRouter()

# Track active ladder orders
_ladder_status: Dict[str, Any] = {}


class BrokerConnect(BaseModel):
    access_token: str
    account_id: str
    sandbox: bool = True


@router.get("/broker/status")
async def broker_status():
    try:
        from tradier_broker import is_configured, get_account_info
        configured = is_configured()
        info = None
        if configured:
            try:
                info = get_account_info()
            except Exception as e:
                info = {"error": str(e)}
        return {"configured": configured, "account_info": info}
    except Exception as e:
        return {"configured": False, "error": str(e)}


@router.post("/broker/connect")
async def broker_connect(req: BrokerConnect):
    try:
        from tradier_broker import save_config, is_configured, get_account_info
        save_config(req.access_token, req.account_id, req.sandbox)
        # Clear data_loader cache so it picks up new config
        import data_loader
        data_loader._tradier_config_cache = None
        data_loader._tradier_session = None

        configured = is_configured()
        info = None
        if configured:
            try:
                info = get_account_info()
            except Exception:
                pass
        return {"success": True, "configured": configured, "account_info": info}
    except Exception as e:
        return {"success": False, "error": str(e)}


class LadderOrder(BaseModel):
    symbol: str
    side: str              # "buy" or "sell"
    quantity: int = 1
    max_attempts: int = 15
    increment: float = 0.10


def _get_quote(symbol: str) -> dict:
    """Get current bid/ask from Tradier."""
    from tradier_broker import _load_config, _get_headers, _base_url
    config = _load_config()
    import requests
    r = requests.get(
        f"{_base_url(config)}/markets/quotes",
        headers=_get_headers(config),
        params={"symbols": symbol.upper(), "greeks": "false"},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    quote = data.get("quotes", {}).get("quote", {})
    return {
        "bid": float(quote.get("bid", 0)),
        "ask": float(quote.get("ask", 0)),
        "last": float(quote.get("last", 0)),
    }


def _run_ladder(order_id: str, symbol: str, side: str, quantity: int,
                max_attempts: int, increment: float):
    """
    Ladder order execution in background thread.
    BUY: starts at bid + increment, raises by increment each attempt
    SELL: starts at ask - increment, lowers by increment each attempt
    Cancels previous unfilled order before placing next.
    """
    from tradier_broker import place_equity_order, cancel_order, get_orders, _load_config, _get_headers, _base_url
    import requests

    status = _ladder_status[order_id]
    status["status"] = "running"

    try:
        quote = _get_quote(symbol)
        if side == "buy":
            start_price = round(quote["bid"] + increment, 2)
        else:
            start_price = round(quote["ask"] - increment, 2)

        status["start_price"] = start_price
        last_order_id = None

        for attempt in range(1, max_attempts + 1):
            if side == "buy":
                price = round(start_price + (attempt - 1) * increment, 2)
            else:
                price = round(start_price - (attempt - 1) * increment, 2)
                if price <= 0:
                    price = 0.01

            status["attempt"] = attempt
            status["current_price"] = price

            # Cancel previous unfilled order
            if last_order_id:
                try:
                    cancel_order(str(last_order_id))
                except Exception:
                    pass
                time.sleep(0.3)

            # Place new limit order
            result = place_equity_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type="limit",
                limit_price=price,
                duration="day",
                preview=False,
            )

            order_resp = result.get("order", {})
            last_order_id = order_resp.get("id")
            status["last_order_id"] = last_order_id
            status["last_result"] = result

            if not last_order_id:
                status["status"] = "error"
                status["error"] = str(result)
                return

            # Wait and check fill
            time.sleep(2.0)

            # Check if filled
            config = _load_config()
            try:
                r = requests.get(
                    f"{_base_url(config)}/accounts/{config['account_id']}/orders/{last_order_id}",
                    headers=_get_headers(config),
                    timeout=10,
                )
                r.raise_for_status()
                order_data = r.json().get("order", {})
                if order_data.get("status") == "filled":
                    status["status"] = "filled"
                    status["fill_price"] = price
                    status["filled_attempt"] = attempt
                    return
            except Exception:
                pass

        # All attempts exhausted
        if last_order_id:
            try:
                cancel_order(str(last_order_id))
            except Exception:
                pass
        status["status"] = "exhausted"

    except Exception as e:
        status["status"] = "error"
        status["error"] = str(e)


@router.post("/broker/ladder")
async def ladder_order(req: LadderOrder):
    """
    Place a ladder order: incremental limit price attempts.
    BUY: starts bid + 0.10, increments +0.10 each attempt (up to 15)
    SELL: starts ask - 0.10, increments -0.10 each attempt (up to 15)
    """
    from tradier_broker import is_configured
    if not is_configured():
        return {"error": "Tradier not configured. Go to Config tab to connect."}

    order_id = f"{req.side}_{req.symbol}_{int(time.time())}"
    _ladder_status[order_id] = {
        "order_id": order_id,
        "symbol": req.symbol.upper(),
        "side": req.side,
        "quantity": req.quantity,
        "max_attempts": req.max_attempts,
        "increment": req.increment,
        "status": "starting",
        "attempt": 0,
        "current_price": None,
        "start_price": None,
    }

    # Run in background thread
    t = threading.Thread(
        target=_run_ladder,
        args=(order_id, req.symbol, req.side, req.quantity,
              req.max_attempts, req.increment),
        daemon=True,
    )
    t.start()

    return {"order_id": order_id, "status": "started"}


@router.get("/broker/ladder/{order_id}")
async def ladder_status(order_id: str):
    """Check status of a ladder order."""
    if order_id not in _ladder_status:
        return {"error": "Order not found"}
    return _ladder_status[order_id]


@router.get("/broker/positions")
async def get_positions():
    from tradier_broker import get_positions
    return {"positions": get_positions()}


@router.get("/broker/orders")
async def get_orders_route(status: str = "all"):
    from tradier_broker import get_orders
    return {"orders": get_orders(status)}
