"""
order_executor.py — Smart Order Execution
Handles incremental limit order placement for both buying and selling.

Buy: Start at bid + 0.05, increment by 0.05 up to 50 attempts until filled.
Sell: Start at ask, decrement by 0.05 up to 50 attempts until filled.
Roll: Sell current + Buy target using same logic.
"""

import time
import threading
from datetime import datetime
from typing import Dict, Optional, Callable
from tradier_broker import (
    place_option_order, place_equity_order, cancel_order,
    get_orders, is_configured,
)


def _get_option_quote(symbol: str, option_symbol: str) -> Dict:
    """Get current bid/ask for an option via yfinance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        for exp in ticker.options:
            chain = ticker.option_chain(exp)
            for _, row in chain.calls.iterrows():
                if row["contractSymbol"] == option_symbol:
                    return {
                        "bid": float(row.get("bid", 0) or 0),
                        "ask": float(row.get("ask", 0) or 0),
                        "last": float(row.get("lastPrice", 0) or 0),
                    }
            for _, row in chain.puts.iterrows():
                if row["contractSymbol"] == option_symbol:
                    return {
                        "bid": float(row.get("bid", 0) or 0),
                        "ask": float(row.get("ask", 0) or 0),
                        "last": float(row.get("lastPrice", 0) or 0),
                    }
    except Exception:
        pass
    return {"bid": 0, "ask": 0, "last": 0}


def execute_buy_calls(
    symbol: str,
    option_symbol: str,
    quantity: int,
    starting_bid: float,
    increment: float = 0.05,
    max_attempts: int = 50,
    delay_seconds: float = 2.0,
    on_status: Callable = None,
) -> Dict:
    """
    Buy-to-open with incrementing limit orders.
    Starts at bid + 0.05, increments by 0.05 each attempt.
    Cancels and replaces until filled or max_attempts reached.
    """
    if not is_configured():
        return {"success": False, "error": "Tradier not configured"}

    current_price = round(starting_bid + increment, 2)
    result = {"success": False, "attempts": 0, "fill_price": None, "order_id": None}

    for attempt in range(1, max_attempts + 1):
        if on_status:
            on_status(f"Attempt {attempt}/{max_attempts}: limit ${current_price:.2f}")

        # Place limit order
        order = place_option_order(
            symbol=symbol,
            option_symbol=option_symbol,
            side="buy_to_open",
            quantity=quantity,
            order_type="limit",
            limit_price=current_price,
            duration="day",
            preview=False,
        )

        if "error" in order:
            result["error"] = order["error"]
            result["attempts"] = attempt
            return result

        # Extract order ID
        order_data = order.get("order", {})
        order_id = order_data.get("id") or order_data.get("order_id")
        if not order_id:
            # Try nested structure
            for key in ["id", "order_id"]:
                if key in order:
                    order_id = order[key]
                    break

        result["order_id"] = order_id

        # Wait briefly for fill
        time.sleep(delay_seconds)

        # Check if filled
        if order_id:
            try:
                orders = get_orders(status="all")
                for o in orders:
                    if str(o.get("id")) == str(order_id):
                        if o.get("status") == "filled":
                            result["success"] = True
                            result["fill_price"] = float(o.get("avg_fill_price", current_price))
                            result["attempts"] = attempt
                            result["filled_at"] = datetime.now().isoformat()
                            return result
                        break

                # Not filled — cancel and increment
                cancel_order(str(order_id))
            except Exception:
                pass

        current_price = round(current_price + increment, 2)
        result["attempts"] = attempt

    result["error"] = f"Not filled after {max_attempts} attempts (final price: ${current_price:.2f})"
    return result


def execute_sell_to_close(
    symbol: str,
    option_symbol: str,
    quantity: int,
    starting_ask: float,
    decrement: float = 0.05,
    max_attempts: int = 50,
    delay_seconds: float = 2.0,
    on_status: Callable = None,
) -> Dict:
    """
    Sell-to-close with decrementing limit orders.
    Starts at ask, decrements by 0.05 each attempt.
    """
    if not is_configured():
        return {"success": False, "error": "Tradier not configured"}

    current_price = round(starting_ask, 2)
    result = {"success": False, "attempts": 0, "fill_price": None}

    for attempt in range(1, max_attempts + 1):
        if on_status:
            on_status(f"Selling attempt {attempt}/{max_attempts}: limit ${current_price:.2f}")

        order = place_option_order(
            symbol=symbol,
            option_symbol=option_symbol,
            side="sell_to_close",
            quantity=quantity,
            order_type="limit",
            limit_price=current_price,
            duration="day",
            preview=False,
        )

        if "error" in order:
            result["error"] = order["error"]
            result["attempts"] = attempt
            return result

        order_data = order.get("order", {})
        order_id = order_data.get("id") or order_data.get("order_id")
        if not order_id:
            for key in ["id", "order_id"]:
                if key in order:
                    order_id = order[key]
                    break

        time.sleep(delay_seconds)

        if order_id:
            try:
                orders = get_orders(status="all")
                for o in orders:
                    if str(o.get("id")) == str(order_id):
                        if o.get("status") == "filled":
                            result["success"] = True
                            result["fill_price"] = float(o.get("avg_fill_price", current_price))
                            result["attempts"] = attempt
                            result["filled_at"] = datetime.now().isoformat()
                            return result
                        break
                cancel_order(str(order_id))
            except Exception:
                pass

        current_price = round(current_price - decrement, 2)
        if current_price <= 0:
            current_price = 0.05
        result["attempts"] = attempt

    result["error"] = f"Not filled after {max_attempts} attempts"
    return result


def execute_roll(
    symbol: str,
    current_contract: str,
    target_contract: str,
    quantity: int,
    current_bid: float,
    target_ask: float,
    on_status: Callable = None,
) -> Dict:
    """
    Execute a roll: sell-to-close current + buy-to-open target.
    Uses the same incremental logic for both legs.
    """
    if on_status:
        on_status("Closing current position...")

    # Sell current
    sell_result = execute_sell_to_close(
        symbol=symbol,
        option_symbol=current_contract,
        quantity=quantity,
        starting_ask=current_bid,  # start at bid (we're selling)
        on_status=on_status,
    )

    if not sell_result.get("success"):
        return {
            "success": False,
            "error": f"Failed to close: {sell_result.get('error', 'unknown')}",
            "sell": sell_result,
        }

    if on_status:
        on_status("Opening new position...")

    # Buy target
    buy_result = execute_buy_calls(
        symbol=symbol,
        option_symbol=target_contract,
        quantity=quantity,
        starting_bid=target_ask - 0.10,  # start slightly below ask
        on_status=on_status,
    )

    credit = (sell_result.get("fill_price", 0) - buy_result.get("fill_price", 0)) if buy_result.get("success") else 0

    return {
        "success": buy_result.get("success", False),
        "sell": sell_result,
        "buy": buy_result,
        "credit": round(credit, 2),
        "rolled_from": current_contract,
        "rolled_to": target_contract,
    }
