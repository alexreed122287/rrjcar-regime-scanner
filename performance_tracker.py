"""
performance_tracker.py — Trade Performance Tracking
SQLite-based trade journal that logs entries, exits, rolls, and computes performance.
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), ".trade_history.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _init_tables(conn)
    return conn


def _init_tables(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            contract TEXT,
            side TEXT DEFAULT 'buy_to_open',
            quantity INTEGER DEFAULT 1,
            entry_price REAL,
            entry_date TEXT,
            exit_price REAL,
            exit_date TEXT,
            pnl_dollars REAL,
            pnl_pct REAL,
            regime_at_entry TEXT,
            signal_at_entry TEXT,
            confidence_tier TEXT,
            risk_dollars REAL,
            roll_count INTEGER DEFAULT 0,
            total_roll_credits REAL DEFAULT 0,
            status TEXT DEFAULT 'open',
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS rolls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER REFERENCES trades(id),
            from_contract TEXT,
            to_contract TEXT,
            roll_type TEXT,
            credit REAL,
            date TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()


def log_entry(
    symbol: str,
    contract: str = None,
    quantity: int = 1,
    entry_price: float = 0,
    regime: str = "",
    signal: str = "",
    confidence_tier: str = "",
    risk_dollars: float = 0,
    notes: str = "",
) -> int:
    """Log a new trade entry. Returns trade_id."""
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO trades (symbol, contract, quantity, entry_price, entry_date,
           regime_at_entry, signal_at_entry, confidence_tier, risk_dollars, notes, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
        (symbol, contract, quantity, entry_price, datetime.now().isoformat(),
         regime, signal, confidence_tier, risk_dollars, notes),
    )
    trade_id = cur.lastrowid
    conn.commit()
    conn.close()
    return trade_id


def log_exit(trade_id: int, exit_price: float, notes: str = ""):
    """Log a trade exit."""
    conn = _get_conn()
    trade = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
    if not trade:
        conn.close()
        return

    entry_price = trade["entry_price"] or 0
    quantity = trade["quantity"] or 1
    roll_credits = trade["total_roll_credits"] or 0

    if trade["contract"]:
        # Options: PnL = (exit - entry) * 100 * qty + roll credits
        pnl_dollars = (exit_price - entry_price) * 100 * quantity + roll_credits
    else:
        # Shares: PnL = (exit - entry) * qty
        pnl_dollars = (exit_price - entry_price) * quantity

    pnl_pct = (exit_price - entry_price) / entry_price * 100 if entry_price > 0 else 0

    conn.execute(
        """UPDATE trades SET exit_price = ?, exit_date = ?, pnl_dollars = ?,
           pnl_pct = ?, status = 'closed', notes = COALESCE(notes || ' | ', '') || ?
           WHERE id = ?""",
        (exit_price, datetime.now().isoformat(), round(pnl_dollars, 2),
         round(pnl_pct, 2), notes, trade_id),
    )
    conn.commit()
    conn.close()


def log_roll(trade_id: int, from_contract: str, to_contract: str,
             roll_type: str, credit: float):
    """Log a roll event for a trade."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO rolls (trade_id, from_contract, to_contract, roll_type, credit, date)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (trade_id, from_contract, to_contract, roll_type, credit, datetime.now().isoformat()),
    )
    # Update trade roll count and credits
    conn.execute(
        """UPDATE trades SET roll_count = roll_count + 1,
           total_roll_credits = total_roll_credits + ?,
           contract = ?
           WHERE id = ?""",
        (credit, to_contract, trade_id),
    )
    conn.commit()
    conn.close()


def get_open_positions() -> List[Dict]:
    """Get all open trades."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM trades WHERE status = 'open' ORDER BY entry_date DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_closed_trades(limit: int = 100) -> List[Dict]:
    """Get closed trades, most recent first."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM trades WHERE status = 'closed' ORDER BY exit_date DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_rolls_for_trade(trade_id: int) -> List[Dict]:
    """Get all rolls for a specific trade."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM rolls WHERE trade_id = ? ORDER BY date", (trade_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_performance_summary() -> Dict:
    """Compute aggregate performance metrics."""
    conn = _get_conn()

    closed = conn.execute(
        "SELECT * FROM trades WHERE status = 'closed'"
    ).fetchall()

    open_trades = conn.execute(
        "SELECT * FROM trades WHERE status = 'open'"
    ).fetchall()

    total_rolls = conn.execute(
        "SELECT COUNT(*) as cnt, SUM(credit) as total_credit FROM rolls"
    ).fetchone()

    conn.close()

    if not closed:
        return {
            "total_trades": 0, "open_trades": len(open_trades),
            "win_rate": 0, "total_pnl": 0, "avg_pnl": 0,
            "total_rolls": 0, "total_roll_credits": 0,
            "best_trade": None, "worst_trade": None,
        }

    closed_dicts = [dict(r) for r in closed]
    wins = [t for t in closed_dicts if (t.get("pnl_dollars") or 0) > 0]
    losses = [t for t in closed_dicts if (t.get("pnl_dollars") or 0) <= 0]

    total_pnl = sum(t.get("pnl_dollars", 0) or 0 for t in closed_dicts)
    avg_pnl = total_pnl / len(closed_dicts) if closed_dicts else 0

    best = max(closed_dicts, key=lambda t: t.get("pnl_dollars", 0) or 0) if closed_dicts else None
    worst = min(closed_dicts, key=lambda t: t.get("pnl_dollars", 0) or 0) if closed_dicts else None

    # By regime
    regime_perf = {}
    for t in closed_dicts:
        regime = t.get("regime_at_entry", "Unknown")
        if regime not in regime_perf:
            regime_perf[regime] = {"trades": 0, "wins": 0, "pnl": 0}
        regime_perf[regime]["trades"] += 1
        regime_perf[regime]["pnl"] += t.get("pnl_dollars", 0) or 0
        if (t.get("pnl_dollars", 0) or 0) > 0:
            regime_perf[regime]["wins"] += 1

    return {
        "total_trades": len(closed_dicts),
        "open_trades": len(open_trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(len(wins) / len(closed_dicts) * 100, 1) if closed_dicts else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "total_rolls": total_rolls["cnt"] or 0,
        "total_roll_credits": round(total_rolls["total_credit"] or 0, 2),
        "best_trade": dict(best) if best else None,
        "worst_trade": dict(worst) if worst else None,
        "regime_performance": regime_perf,
    }
