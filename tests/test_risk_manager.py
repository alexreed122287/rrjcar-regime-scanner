"""
Tests for risk_manager circuit breakers.

Every test injects an isolated state_dir and a fake equity provider, so no test ever
touches the real HALT_TRADING sentinel or the real equity history.
"""

import json
from datetime import date, timedelta

import pytest

from risk_manager import (
    RiskManager,
    TradingHalted,
    apply_size_multiplier,
    is_exit_side,
    is_order_permitted,
    DAILY_HALVE_PCT,
    DAILY_BLOCK_PCT,
    WEEKLY_HALVE_PCT,
    PEAK_DRAWDOWN_HALT_PCT,
)

START_EQUITY = 100_000.0


def make_manager(tmp_path, equity):
    """RiskManager pinned to an isolated dir with a fixed equity reading."""
    return RiskManager(state_dir=tmp_path, equity_provider=lambda: equity)


def seed_history(tmp_path, points, peak=None):
    """
    Write an equity history file in the schema RiskManager actually reads.

    points = [(date, equity), ...] oldest first.
    """
    snapshots = [{"date": d.isoformat(), "equity": float(eq)} for d, eq in points]
    payload = {
        "peak_equity": float(peak) if peak is not None else max(eq for _, eq in points),
        "snapshots": snapshots,
    }
    (tmp_path / ".equity_history.json").write_text(json.dumps(payload))


# ── baseline ──

def test_clean_state_allows_trading(tmp_path):
    mgr = make_manager(tmp_path, START_EQUITY)
    status = mgr.evaluate()
    assert status.status == "OK"
    assert not status.halted
    assert is_order_permitted("buy", status)


def test_no_sentinel_file_created_on_clean_evaluate(tmp_path):
    mgr = make_manager(tmp_path, START_EQUITY)
    mgr.evaluate()
    assert not (tmp_path / "HALT_TRADING").exists()


# ── daily loss breakers ──

def test_daily_loss_beyond_halve_threshold_reduces_size(tmp_path):
    today = date.today()
    seed_history(tmp_path, [(today - timedelta(days=1), START_EQUITY)])
    # -2.5% is past the halve threshold but not the block threshold.
    mgr = make_manager(tmp_path, START_EQUITY * 0.975)
    status = mgr.evaluate(when=today)

    assert status.daily_pct < DAILY_HALVE_PCT
    assert status.status == "REDUCED"
    assert status.size_multiplier == pytest.approx(0.5)
    # Entries still allowed, just smaller.
    assert is_order_permitted("buy", status)
    assert apply_size_multiplier(10, status) == 5


def test_daily_loss_beyond_block_threshold_blocks_entries(tmp_path):
    today = date.today()
    seed_history(tmp_path, [(today - timedelta(days=1), START_EQUITY)])
    mgr = make_manager(tmp_path, START_EQUITY * 0.96)  # -4%
    status = mgr.evaluate(when=today)

    assert status.daily_pct < DAILY_BLOCK_PCT
    assert status.status == "NO_ENTRY"
    assert not is_order_permitted("buy", status)
    # Exits must always remain possible so positions are never trapped.
    assert is_order_permitted("sell", status)


# ── weekly loss breaker ──

def test_weekly_loss_reduces_size(tmp_path):
    today = date.today()
    # Reference point 8 days back so the 7-day lookback finds it; yesterday's value is
    # close to today's so the DAILY breakers stay clear and we isolate the weekly one.
    seed_history(tmp_path, [
        (today - timedelta(days=8), START_EQUITY),
        (today - timedelta(days=1), START_EQUITY * 0.941),
    ], peak=START_EQUITY)
    mgr = make_manager(tmp_path, START_EQUITY * 0.94)  # -6% on the week
    status = mgr.evaluate(when=today)

    assert status.weekly_pct is not None
    assert status.weekly_pct < WEEKLY_HALVE_PCT
    assert status.status in ("REDUCED", "NO_ENTRY", "HALTED")
    assert status.size_multiplier <= 0.5


# ── peak drawdown halt ──

def test_peak_drawdown_writes_sentinel_and_halts(tmp_path):
    today = date.today()
    seed_history(tmp_path, [(today - timedelta(days=30), START_EQUITY)])
    mgr = make_manager(tmp_path, START_EQUITY * 0.88)  # -12% from peak
    status = mgr.evaluate(when=today)

    assert status.drawdown_pct < PEAK_DRAWDOWN_HALT_PCT
    assert status.status == "HALTED"
    assert status.halted
    assert (tmp_path / "HALT_TRADING").exists()
    # Nothing gets through a hard halt — not even exits via the entry gate.
    assert not is_order_permitted("buy", status)


def test_existing_sentinel_halts_regardless_of_equity(tmp_path):
    (tmp_path / "HALT_TRADING").write_text("halted by hand\n")
    mgr = make_manager(tmp_path, START_EQUITY * 2)  # equity is great; still halted
    status = mgr.evaluate()

    assert status.halted
    assert status.status == "HALTED"
    assert "halted by hand" in (mgr.read_halt_reason() or "")


def test_sentinel_is_never_overwritten(tmp_path):
    original = "FIRST REASON — keep me\n"
    (tmp_path / "HALT_TRADING").write_text(original)
    today = date.today()
    seed_history(tmp_path, [(today - timedelta(days=30), START_EQUITY)])

    mgr = make_manager(tmp_path, START_EQUITY * 0.5)  # would also trigger a halt
    mgr.evaluate(when=today)

    assert (tmp_path / "HALT_TRADING").read_text() == original


def test_evaluate_never_clears_sentinel_after_recovery(tmp_path):
    """A recovered account must stay halted until a human clears the file."""
    (tmp_path / "HALT_TRADING").write_text("drawdown breach\n")
    mgr = make_manager(tmp_path, START_EQUITY * 1.5)
    for _ in range(3):
        status = mgr.evaluate()
        assert status.halted
    assert (tmp_path / "HALT_TRADING").exists()


# ── fail-closed behavior ──

def test_unreadable_equity_blocks_entries_but_allows_exits(tmp_path):
    def boom():
        raise RuntimeError("broker unreachable")

    mgr = RiskManager(state_dir=tmp_path, equity_provider=boom)
    status = mgr.evaluate()

    assert status.status in ("NO_ENTRY", "HALTED", "UNKNOWN")
    assert not is_order_permitted("buy", status)
    assert is_order_permitted("sell", status)


# ── order gate ──

def test_assert_order_allowed_raises_when_blocked(tmp_path):
    seed_history(tmp_path, [(date.today() - timedelta(days=1), START_EQUITY)])
    mgr = make_manager(tmp_path, START_EQUITY * 0.95)  # -5% today

    with pytest.raises(TradingHalted):
        mgr.assert_order_allowed("buy")


def test_assert_order_allowed_permits_exit_when_blocked(tmp_path):
    seed_history(tmp_path, [(date.today() - timedelta(days=1), START_EQUITY)])
    mgr = make_manager(tmp_path, START_EQUITY * 0.95)

    # Should not raise — closing risk is always allowed short of a hard halt.
    mgr.assert_order_allowed("sell")


@pytest.mark.parametrize("side,expected", [
    ("sell", True),
    ("SELL", True),
    ("sell_to_close", True),
    ("buy_to_close", True),
    ("buy", False),
    ("buy_to_open", False),
    ("sell_to_open", False),
])
def test_exit_side_detection(side, expected):
    assert is_exit_side(side) is expected


def test_equity_history_is_recorded_one_snapshot_per_day(tmp_path):
    mgr = make_manager(tmp_path, START_EQUITY)
    mgr.evaluate()
    mgr.evaluate()
    payload = json.loads((tmp_path / ".equity_history.json").read_text())
    snaps = payload["snapshots"]
    # One snapshot per calendar day — repeated calls update rather than append.
    assert len(snaps) == 1
    assert snaps[0]["equity"] == START_EQUITY
    assert payload["peak_equity"] == START_EQUITY
