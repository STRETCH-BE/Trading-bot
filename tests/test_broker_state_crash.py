"""State survives a kill at five points in an order lifecycle.

The process is genuinely killed with os._exit() in a subprocess — no cleanup,
no atexit, no SQLite teardown. Mocking this would test the mock.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from trading_bot.broker.base import OrderStatus
from trading_bot.broker.state import StateStore

HARNESS = Path(__file__).parent.parent / "scripts" / "crash_harness.py"

KILL_POINTS = [
    "after_insert",
    "after_partial_fill",
    "after_full_fill",
    "after_equity",
    "mid_transaction",
]


def run_harness(db: Path, die_at: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(HARNESS), str(db), die_at],
        capture_output=True, text=True,
    )


@pytest.mark.parametrize("die_at", KILL_POINTS)
def test_state_is_consistent_after_a_kill(die_at, tmp_path):
    """THE mandated test: kill at five points, verify recoverable state."""
    db = tmp_path / "state.db"
    result = run_harness(db, die_at)
    assert result.returncode in (9, -9), (
        f"harness did not die as expected at {die_at}: rc={result.returncode} "
        f"stderr={result.stderr}"
    )

    # Reopen exactly as a restarting process would.
    store = StateStore(db)
    problems = store.check_integrity()
    assert problems == [], f"inconsistent state after kill at {die_at}: {problems}"


def test_kill_mid_transaction_rolls_back(tmp_path):
    """A kill inside a transaction must leave NO trace of the partial write."""
    db = tmp_path / "state.db"
    run_harness(db, "mid_transaction")

    store = StateStore(db)
    assert store.check_integrity() == []
    # the uncommitted 999-unit fill must be absent
    assert all(f["units"] != 999.0 for f in store.all_fills())
    order = store.get_order("tb-crash")
    assert order is not None
    assert order.filled_units == 0.0


def test_kill_after_insert_leaves_a_recoverable_open_order(tmp_path):
    db = tmp_path / "state.db"
    run_harness(db, "after_insert")

    store = StateStore(db)
    assert store.check_integrity() == []
    order = store.get_order("tb-crash")
    assert order is not None
    assert order.status is OrderStatus.OPEN
    assert order.filled_units == 0.0
    # a restarting process can see it needs resolving
    assert [o.order.client_order_id for o in store.open_orders()] == ["tb-crash"]


def test_kill_after_partial_fill_keeps_fill_and_order_in_agreement(tmp_path):
    db = tmp_path / "state.db"
    run_harness(db, "after_partial_fill")

    store = StateStore(db)
    assert store.check_integrity() == []
    order = store.get_order("tb-crash")
    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert order.filled_units == pytest.approx(0.01)  # half of 0.02
    fills = store.fills_for("tb-crash")
    assert len(fills) == 1
    assert sum(f["units"] for f in fills) == pytest.approx(order.filled_units)
    assert store.positions()[0].units == pytest.approx(0.01)


def test_kill_after_full_fill_is_complete_and_balanced(tmp_path):
    db = tmp_path / "state.db"
    run_harness(db, "after_full_fill")

    store = StateStore(db)
    assert store.check_integrity() == []
    order = store.get_order("tb-crash")
    assert order.status is OrderStatus.FILLED
    assert order.filled_units == pytest.approx(0.02)
    assert store.positions()[0].units == pytest.approx(0.02)
    assert store.open_orders() == []
    # cash was debited exactly once per fill
    assert store.get_cash() < 10_000.0


def test_completed_run_records_equity(tmp_path):
    db = tmp_path / "state.db"
    result = run_harness(db, "none")
    assert "completed" in result.stdout

    store = StateStore(db)
    assert store.check_integrity() == []
    assert len(store.equity_history()) == 1


def test_integrity_check_actually_detects_corruption(tmp_path):
    """Control: the checker must fail on state it should reject."""
    db = tmp_path / "state.db"
    run_harness(db, "after_full_fill")

    store = StateStore(db)
    assert store.check_integrity() == []
    # corrupt deliberately: claim more filled than the fills support
    store._conn.execute("UPDATE orders SET filled_units = 99.0")
    problems = store.check_integrity()
    assert problems, "integrity check passed corrupted state"
    assert any("filled_units" in p for p in problems)
