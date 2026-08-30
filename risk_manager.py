"""
risk_manager.py — Account-level circuit breakers.

Completely independent of the HMM / strategy layer. This module does not know or care
what the model thinks; it looks only at realized account equity and vetoes trading when
loss thresholds are breached.

Thresholds
----------
    Daily   -2%   -> HALVE      new position sizes cut 50% for the rest of the session
    Daily   -3%   -> NO_ENTRY   no new entries for the rest of the session (exits allowed)
    Weekly  -5%   -> HALVE      size reduction until the weekly figure recovers
    Peak DD -10%  -> HALT       all trading blocked; writes the HALT_TRADING sentinel file

The -10% peak-drawdown breach writes a ``HALT_TRADING`` file at the repo root containing
the equity curve that triggered it. Trading stays blocked until a human **manually deletes
that file**. There is deliberately no code path in this repository that clears it.

Exits are always permitted. Reducing existing risk must never be blocked.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Optional

# ── Thresholds (see .github/copilot-instructions.md before changing these) ──
DAILY_HALVE_PCT = -2.0
DAILY_BLOCK_PCT = -3.0
WEEKLY_HALVE_PCT = -5.0
PEAK_DRAWDOWN_HALT_PCT = -10.0

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
HALT_SENTINEL_NAME = "HALT_TRADING"
EQUITY_HISTORY_NAME = ".equity_history.json"

# Risk status levels, ordered from least to most restrictive.
STATUS_OK = "OK"
STATUS_REDUCED = "REDUCED"
STATUS_NO_ENTRY = "NO_ENTRY"
STATUS_HALTED = "HALTED"

_STATUS_SEVERITY = {STATUS_OK: 0, STATUS_REDUCED: 1, STATUS_NO_ENTRY: 2, STATUS_HALTED: 3}


class TradingHalted(Exception):
    """Raised when an order is attempted while trading is halted or entries are blocked."""

    def __init__(self, message: str, status: "RiskStatus"):
        super().__init__(message)
        self.status = status


@dataclass
class RiskStatus:
    """Result of a risk evaluation."""

    status: str = STATUS_OK
    size_multiplier: float = 1.0
    allow_new_entries: bool = True
    allow_exits: bool = True  # invariant: always True
    halted: bool = False
    reasons: List[str] = field(default_factory=list)
    breaches: List[str] = field(default_factory=list)
    equity: Optional[float] = None
    peak_equity: Optional[float] = None
    daily_pct: Optional[float] = None
    weekly_pct: Optional[float] = None
    drawdown_pct: Optional[float] = None
    sentinel_path: Optional[str] = None
    evaluated_at: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def is_ok(self) -> bool:
        return self.status == STATUS_OK


def _round_or_none(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(float(value), digits)


class RiskManager:
    """
    Evaluates account-level circuit breakers against an equity history.

    Parameters
    ----------
    equity_provider : callable, optional
        Returns current account equity as a float. Defaults to reading
        ``total_equity`` from ``tradier_broker.get_account_info()``.
    state_dir : str, optional
        Directory holding the sentinel and equity-history files. Defaults to the
        repository root. Overridden in tests via ``tmp_path``.
    """

    def __init__(
        self,
        equity_provider: Optional[Callable[[], float]] = None,
        state_dir: Optional[str] = None,
    ):
        self.state_dir = state_dir or REPO_ROOT
        self._equity_provider = equity_provider or _tradier_equity_provider

    # ── file paths ──

    @property
    def sentinel_path(self) -> str:
        return os.path.join(self.state_dir, HALT_SENTINEL_NAME)

    @property
    def history_path(self) -> str:
        return os.path.join(self.state_dir, EQUITY_HISTORY_NAME)

    # ── sentinel ──

    def is_halted(self) -> bool:
        """True if the HALT_TRADING sentinel exists. Only a human removes it."""
        return os.path.exists(self.sentinel_path)

    def read_halt_reason(self) -> Optional[str]:
        if not self.is_halted():
            return None
        try:
            with open(self.sentinel_path, "r") as f:
                return f.read()
        except OSError:
            return "HALT_TRADING present but unreadable."

    def _write_halt_sentinel(self, status: "RiskStatus", history: List[Dict]) -> None:
        """
        Write the halt sentinel. Never overwrite an existing one — the first breach is
        the one worth reading, and rewriting would destroy the original evidence.
        """
        if os.path.exists(self.sentinel_path):
            return

        tail = history[-30:]
        curve = "\n".join(
            f"  {row.get('date', '?')}  equity={row.get('equity')}" for row in tail
        )
        body = (
            "HALT_TRADING\n"
            "============\n\n"
            f"Written at: {datetime.now().isoformat(timespec='seconds')}\n"
            f"Trigger:    peak drawdown {status.drawdown_pct}% "
            f"(limit {PEAK_DRAWDOWN_HALT_PCT}%)\n"
            f"Equity:     {status.equity}\n"
            f"Peak:       {status.peak_equity}\n\n"
            "All trading is blocked while this file exists.\n"
            "Review what happened, then delete this file MANUALLY to resume.\n"
            "Nothing in this repository removes it automatically.\n\n"
            "Equity curve (most recent 30 snapshots):\n"
            f"{curve}\n"
        )
        with open(self.sentinel_path, "w") as f:
            f.write(body)

    # ── equity history ──

    def _load_history(self) -> List[Dict]:
        if not os.path.exists(self.history_path):
            return []
        try:
            with open(self.history_path, "r") as f:
                data = json.load(f)
            snaps = data.get("snapshots", []) if isinstance(data, dict) else data
            return [s for s in snaps if isinstance(s, dict) and "equity" in s]
        except (OSError, json.JSONDecodeError, ValueError):
            return []

    def _save_history(self, snapshots: List[Dict], peak: float) -> None:
        payload = {
            "peak_equity": peak,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "snapshots": snapshots[-400:],  # ~18 months of trading days
        }
        tmp = self.history_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self.history_path)

    def record_equity(self, equity: float, when: Optional[date] = None) -> List[Dict]:
        """
        Append or update today's equity snapshot and return the full history.

        One snapshot per calendar day: repeated calls on the same day update the day's
        latest value rather than inflating the history.
        """
        when = when or date.today()
        stamp = when.isoformat()
        history = self._load_history()

        if history and history[-1].get("date") == stamp:
            history[-1]["equity"] = float(equity)
            history[-1]["updated_at"] = datetime.now().isoformat(timespec="seconds")
        else:
            history.append(
                {
                    "date": stamp,
                    "equity": float(equity),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
            )

        peak = max([float(s["equity"]) for s in history] or [float(equity)])
        # Peak is monotonic across the account's life, even if history is trimmed.
        prior_peak = self._stored_peak()
        if prior_peak is not None:
            peak = max(peak, prior_peak)

        self._save_history(history, peak)
        return history

    def _stored_peak(self) -> Optional[float]:
        if not os.path.exists(self.history_path):
            return None
        try:
            with open(self.history_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("peak_equity") is not None:
                return float(data["peak_equity"])
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            pass
        return None

    # ── evaluation ──

    def evaluate(
        self,
        equity: Optional[float] = None,
        when: Optional[date] = None,
        record: bool = True,
    ) -> RiskStatus:
        """
        Evaluate all circuit breakers.

        Parameters
        ----------
        equity : float, optional
            Current account equity. Fetched from the broker if omitted.
        when : date, optional
            Evaluation date, for testing.
        record : bool
            Persist this equity reading into the history before evaluating.
        """
        when = when or date.today()
        status = RiskStatus(evaluated_at=datetime.now().isoformat(timespec="seconds"))
        status.sentinel_path = self.sentinel_path

        # A pre-existing sentinel halts everything, no matter what the numbers say.
        # This also honors a manually created HALT_TRADING file.
        if self.is_halted():
            status.status = STATUS_HALTED
            status.halted = True
            status.allow_new_entries = False
            status.size_multiplier = 0.0
            status.breaches.append("sentinel")
            status.reasons.append(
                f"HALT_TRADING sentinel present at {self.sentinel_path}. "
                "Delete it manually after review to resume trading."
            )
            return status

        if equity is None:
            try:
                equity = self._equity_provider()
            except Exception as exc:  # broker unreachable
                # Fail closed on new risk, but never block exits.
                status.status = STATUS_NO_ENTRY
                status.allow_new_entries = False
                status.size_multiplier = 0.0
                status.breaches.append("equity_unavailable")
                status.reasons.append(
                    f"Could not read account equity ({exc}). Blocking new entries; "
                    "exits remain allowed."
                )
                return status

        equity = float(equity)
        status.equity = _round_or_none(equity)

        history = self.record_equity(equity, when=when) if record else self._load_history()

        # Peak equity is monotonic over the account's life: the highest of the stored
        # peak, every retained snapshot, and the current reading.
        candidates = [float(s["equity"]) for s in history] + [equity]
        stored_peak = self._stored_peak()
        if stored_peak is not None:
            candidates.append(stored_peak)
        peak = max(candidates)
        status.peak_equity = _round_or_none(peak)

        # ── peak drawdown ──
        drawdown_pct = ((equity - peak) / peak * 100.0) if peak > 0 else 0.0
        status.drawdown_pct = _round_or_none(drawdown_pct)

        # ── daily change ──
        daily_pct = self._pct_change_since(history, equity, when, days_back=1)
        status.daily_pct = _round_or_none(daily_pct)

        # ── weekly change ──
        weekly_pct = self._pct_change_since(history, equity, when, days_back=7)
        status.weekly_pct = _round_or_none(weekly_pct)

        # ── apply breakers, most severe wins ──
        if drawdown_pct <= PEAK_DRAWDOWN_HALT_PCT:
            status.breaches.append("peak_drawdown")
            status.reasons.append(
                f"Peak drawdown {drawdown_pct:.2f}% breached "
                f"{PEAK_DRAWDOWN_HALT_PCT}% limit. Trading halted."
            )
            self._escalate(status, STATUS_HALTED)
            status.halted = True
            status.allow_new_entries = False
            status.size_multiplier = 0.0
            self._write_halt_sentinel(status, history)
            return status

        if daily_pct is not None and daily_pct <= DAILY_BLOCK_PCT:
            status.breaches.append("daily_block")
            status.reasons.append(
                f"Daily loss {daily_pct:.2f}% breached {DAILY_BLOCK_PCT}% limit. "
                "No new entries for the rest of the session; exits allowed."
            )
            self._escalate(status, STATUS_NO_ENTRY)
            status.allow_new_entries = False
            status.size_multiplier = 0.0

        elif daily_pct is not None and daily_pct <= DAILY_HALVE_PCT:
            status.breaches.append("daily_halve")
            status.reasons.append(
                f"Daily loss {daily_pct:.2f}% breached {DAILY_HALVE_PCT}% limit. "
                "New position sizes cut 50% for the rest of the session."
            )
            self._escalate(status, STATUS_REDUCED)
            status.size_multiplier = min(status.size_multiplier, 0.5)

        if weekly_pct is not None and weekly_pct <= WEEKLY_HALVE_PCT:
            status.breaches.append("weekly_halve")
            status.reasons.append(
                f"Weekly loss {weekly_pct:.2f}% breached {WEEKLY_HALVE_PCT}% limit. "
                "Position sizes reduced until the weekly figure recovers."
            )
            self._escalate(status, STATUS_REDUCED)
            if status.allow_new_entries:
                status.size_multiplier = min(status.size_multiplier, 0.5)

        if not status.reasons:
            status.reasons.append("All circuit breakers clear.")

        # Invariant: exits are never blocked.
        status.allow_exits = True
        return status

    @staticmethod
    def _escalate(status: RiskStatus, new_status: str) -> None:
        if _STATUS_SEVERITY[new_status] > _STATUS_SEVERITY[status.status]:
            status.status = new_status

    @staticmethod
    def _pct_change_since(
        history: List[Dict],
        equity: float,
        when: date,
        days_back: int,
    ) -> Optional[float]:
        """
        Percent change from the most recent snapshot at least ``days_back`` days before
        ``when``. Returns None when there is no such reference point yet.
        """
        cutoff = when - timedelta(days=days_back)
        reference = None
        for snap in history:
            try:
                snap_date = date.fromisoformat(str(snap.get("date")))
            except (TypeError, ValueError):
                continue
            if snap_date <= cutoff:
                reference = float(snap["equity"])
        if reference is None or reference <= 0:
            return None
        return (equity - reference) / reference * 100.0

    # ── order gate ──

    def check_order_allowed(self, side: str, equity: Optional[float] = None) -> RiskStatus:
        """
        Evaluate whether an order may be submitted.

        ``side`` semantics: anything that reduces risk ("sell", "sell_to_close",
        "buy_to_close") is treated as an exit and permitted unless fully halted.
        """
        status = self.evaluate(equity=equity)
        return status

    def assert_order_allowed(self, side: str, equity: Optional[float] = None) -> RiskStatus:
        """Raise TradingHalted if this order must not be submitted."""
        status = self.check_order_allowed(side, equity=equity)
        if not is_order_permitted(side, status):
            raise TradingHalted("; ".join(status.reasons), status)
        return status


def is_order_permitted(side: str, status: RiskStatus) -> bool:
    """
    Exits are allowed unless trading is fully halted. Entries require
    ``allow_new_entries``.
    """
    if status.halted:
        return False
    if is_exit_side(side):
        return status.allow_exits
    return status.allow_new_entries


def is_exit_side(side: str) -> bool:
    """True when the order reduces existing exposure rather than adding risk."""
    s = (side or "").strip().lower().replace("-", "_").replace(" ", "_")
    return s in {
        "sell",
        "sell_to_close",
        "buy_to_close",
        "close",
        "exit",
        "sell_short_close",
    }


def _tradier_equity_provider() -> float:
    """Read total account equity from Tradier. Raises if unavailable."""
    from tradier_broker import get_account_info

    info = get_account_info() or {}
    if info.get("error"):
        raise RuntimeError(str(info["error"]))
    equity = info.get("total_equity")
    if equity in (None, "", 0):
        raise RuntimeError("Tradier returned no total_equity")
    return float(equity)


# ── module-level convenience API ──

_default_manager: Optional[RiskManager] = None


def get_risk_manager() -> RiskManager:
    """Shared RiskManager using the repo root for state."""
    global _default_manager
    if _default_manager is None:
        _default_manager = RiskManager()
    return _default_manager


def check_risk_status(equity: Optional[float] = None, record: bool = True) -> RiskStatus:
    """Evaluate circuit breakers with the default manager."""
    return get_risk_manager().evaluate(equity=equity, record=record)


def is_halted() -> bool:
    return get_risk_manager().is_halted()


def apply_size_multiplier(quantity: int, status: RiskStatus) -> int:
    """
    Scale an intended order quantity by the risk multiplier.

    Returns 0 when trading is blocked. A surviving position is never rounded below 1
    unless the multiplier is exactly 0.
    """
    if status.size_multiplier <= 0:
        return 0
    scaled = int(quantity * status.size_multiplier)
    return max(scaled, 1) if quantity >= 1 else 0
