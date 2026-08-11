"""Stage 7 operational surface: kill switch, heartbeat, backups.

These are the pieces the RUNBOOK points at. A runbook command that does not do
what the runbook says is worse than no runbook, so each documented behaviour is
asserted here.
"""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from trading_bot.backtest.config import BacktestConfig
from trading_bot.broker import PaperBroker, PaperBrokerConfig, StateStore
from trading_bot.data import schema
from trading_bot.execution import FillModel
from trading_bot.risk import RiskGate, RiskLimits
from trading_bot.scheduler import CycleContext, CycleHalted, run_cycle
from trading_bot.strategies.voltrend import VolTrendParams, voltrend

from .synthetic import from_closes

STAMP = "%Y%m%dT%H%M%SZ"


def build(tmp_path, *, signal_fn=None):
    local = StateStore(tmp_path / "local.db")
    venue = StateStore(tmp_path / "venue.db")
    local.set_cash(10_000.0)
    venue.set_cash(10_000.0)
    fm = FillModel(maker_fee_bps=16.0, taker_fee_bps=26.0, slippage_bps=5.0)
    cs = from_closes([50.0 * (1.01**i) for i in range(300)])
    broker = PaperBroker(
        venue, fm, lambda pair: float(cs[schema.CLOSE].iloc[-1]),
        config=PaperBrokerConfig.deterministic(), rng=random.Random(1),
        clock=lambda: pd.Timestamp(cs[schema.TIMESTAMP].iloc[-1]),
    )
    ctx = CycleContext(
        local=local, broker=broker,
        gate=RiskGate(RiskLimits(), fm, halt_file=tmp_path / "HALT"),
        fill_model=fm,
        config=BacktestConfig(starting_capital=10_000.0, min_rebalance_delta=0.05),
        strategy_name="voltrend",
        signal_fn=signal_fn or (lambda c: voltrend(c, VolTrendParams())),
        candles_for=lambda pair: cs, pairs=("XBTEUR",),
    )
    return ctx, cs


# --- the kill switch ---------------------------------------------------------


def test_touch_halt_stops_the_next_cycle_even_with_nothing_to_trade(tmp_path):
    """THE runbook guarantee for `touch HALT`.

    Regression: the gate checks the halt file, but the gate only runs when
    there IS an order to approve. A strategy sitting flat never produced one,
    so `touch HALT` left the process running cycle after cycle. The check now
    happens at the top of the cycle, before the strategy is even consulted.
    """
    ctx, cs = build(tmp_path, signal_fn=lambda c: pd.Series([0.0] * len(c)))
    now = pd.Timestamp(cs[schema.TIMESTAMP].iloc[-1])

    # flat strategy, no halt: a perfectly ordinary do-nothing cycle
    assert run_cycle(ctx, now=now, reconcile_first=True).halted is False

    (tmp_path / "HALT").write_text("HALTED by hand\n")
    with pytest.raises(CycleHalted, match="HALT file present"):
        run_cycle(ctx, now=now)


def test_halt_stops_the_cycle_before_any_broker_call(tmp_path):
    """Halting must not settle, reconcile, or read the strategy first."""
    ctx, cs = build(tmp_path)
    (tmp_path / "HALT").write_text("HALTED\n")

    calls = []
    ctx.candles_for = lambda pair: calls.append(pair) or cs

    with pytest.raises(CycleHalted):
        run_cycle(ctx, now=pd.Timestamp(cs[schema.TIMESTAMP].iloc[-1]),
                  reconcile_first=True)
    assert calls == [], "the strategy was consulted despite the halt"


def test_removing_the_halt_file_resumes(tmp_path):
    """Only a human removes it — but when they do, it must actually resume."""
    ctx, cs = build(tmp_path)
    now = pd.Timestamp(cs[schema.TIMESTAMP].iloc[-1])
    halt = tmp_path / "HALT"

    halt.write_text("HALTED\n")
    with pytest.raises(CycleHalted):
        run_cycle(ctx, now=now, reconcile_first=True)

    halt.unlink()
    assert run_cycle(ctx, now=now, reconcile_first=True).halted is False


def test_run_main_refuses_to_start_while_halted(tmp_path, capsys):
    """systemd will restart the process; the halt must survive that."""
    from trading_bot.run import main

    halt = tmp_path / "HALT"
    halt.write_text("HALTED 2026-08-11\nreason: max_drawdown_pct\n")

    rc = main([
        "--once", "--config", "config.yaml",
        "--state-dir", str(tmp_path / "state"),
        "--halt-file", str(halt),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "HALTED" in err and "max_drawdown_pct" in err
    assert not (tmp_path / "state").exists(), "state was opened while halted"


def test_reconcile_only_still_works_while_halted(tmp_path):
    """Diagnostics while halted are the whole point of --reconcile-only."""
    from trading_bot.run import main

    halt = tmp_path / "HALT"
    halt.write_text("HALTED\nreason: testing\n")
    rc = main([
        "--reconcile-only", "--config", "config.yaml",
        "--state-dir", str(tmp_path / "state"),
        "--halt-file", str(halt),
    ])
    assert rc == 0, "reconciliation was blocked by the halt"


# --- heartbeat ---------------------------------------------------------------


def _write_heartbeat(path: Path, *, age_hours: float = 0.0, halted: bool = False):
    written = datetime.now(UTC) - timedelta(hours=age_hours)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "written_at": written.isoformat(), "cycle_id": "abc123",
        "mode": "paper", "settled": 0, "orders_submitted": 0,
        "rejections": [], "halted": halted,
    }))


def test_heartbeat_checker_accepts_a_fresh_beat(tmp_path):
    from scripts.check_heartbeat import main

    hb = tmp_path / "heartbeat.json"
    _write_heartbeat(hb, age_hours=1.0)
    assert main(["--heartbeat-file", str(hb)]) == 0


def test_heartbeat_checker_reports_a_stale_beat(tmp_path, capsys):
    from scripts.check_heartbeat import main

    hb = tmp_path / "heartbeat.json"
    _write_heartbeat(hb, age_hours=48.0)
    assert main(["--heartbeat-file", str(hb)]) == 1
    assert "STALE" in capsys.readouterr().err


def test_heartbeat_checker_reports_a_halt_distinctly(tmp_path, capsys):
    """Exit 2 not 1: 'halted on purpose' is a different alarm from 'wedged'."""
    from scripts.check_heartbeat import main

    hb = tmp_path / "heartbeat.json"
    _write_heartbeat(hb, halted=True)
    assert main(["--heartbeat-file", str(hb)]) == 2
    assert "HALTED" in capsys.readouterr().err


def test_missing_and_corrupt_heartbeats_are_failures_not_crashes(tmp_path):
    from scripts.check_heartbeat import main

    assert main(["--heartbeat-file", str(tmp_path / "nope.json")]) == 1
    corrupt = tmp_path / "heartbeat.json"
    corrupt.write_text("{not json")
    assert main(["--heartbeat-file", str(corrupt)]) == 1


def test_bot_writes_a_heartbeat_after_each_cycle(tmp_path):
    from trading_bot.run import write_heartbeat

    ctx, cs = build(tmp_path)
    result = run_cycle(ctx, now=pd.Timestamp(cs[schema.TIMESTAMP].iloc[-1]),
                       reconcile_first=True)
    hb = tmp_path / "state" / "heartbeat.json"
    write_heartbeat(hb, result, "paper")

    payload = json.loads(hb.read_text())
    assert payload["cycle_id"] == result.cycle_id
    assert payload["mode"] == "paper"
    assert not list(hb.parent.glob("*.tmp")), "atomic write left a temp file"

    from scripts.check_heartbeat import main
    assert main(["--heartbeat-file", str(hb)]) == 0


def test_heartbeat_failure_never_takes_down_the_bot(tmp_path):
    """A disk problem in the heartbeat must not kill a trading process."""
    from trading_bot.run import write_heartbeat

    ctx, cs = build(tmp_path)
    result = run_cycle(ctx, now=pd.Timestamp(cs[schema.TIMESTAMP].iloc[-1]),
                       reconcile_first=True)
    blocked = tmp_path / "a-file" / "heartbeat.json"
    (tmp_path / "a-file").write_text("not a directory")

    write_heartbeat(blocked, result, "paper")  # must not raise


# --- backups -----------------------------------------------------------------


def test_backup_uses_the_online_api_and_verifies_integrity(tmp_path):
    """A live, WAL-mode database with an OPEN transaction must still snapshot."""
    from scripts.backup_state import backup_one

    state = tmp_path / "state"
    state.mkdir()
    store = StateStore(state / "local.db")
    store.set_cash(1234.5)

    dest = backup_one(state / "local.db", tmp_path / "backups", "20260811T000000Z")
    assert dest.exists()

    restored = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
    try:
        assert restored.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        value = restored.execute(
            "SELECT value FROM account WHERE key='cash'"
        ).fetchone()[0]
    finally:
        restored.close()
    assert value == pytest.approx(1234.5), "the snapshot lost committed data"


def test_backup_captures_wal_content_that_a_file_copy_would_miss(tmp_path):
    """The reason this is not `cp`: in WAL mode recent commits live in the -wal.

    The bot's connection stays open, so the -wal is not checkpointed. A copy of
    the .db alone would miss this write; the backup API must not.
    """
    from scripts.backup_state import backup_one

    state = tmp_path / "state"
    state.mkdir()
    store = StateStore(state / "local.db")  # deliberately left OPEN
    store.set_cash(999.0)
    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    dest = backup_one(state / "local.db", tmp_path / "backups", "20260811T000000Z")
    conn = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
    try:
        got = conn.execute("SELECT value FROM account WHERE key='cash'").fetchone()[0]
    finally:
        conn.close()
    assert got == pytest.approx(999.0)


def test_retention_prunes_only_old_backups(tmp_path):
    from scripts.backup_state import sweep

    backups = tmp_path / "backups"
    backups.mkdir()
    now = datetime(2026, 8, 11, tzinfo=UTC)
    for days in (0, 5, 29, 31, 400):
        stamp = (now - timedelta(days=days)).strftime(STAMP)
        (backups / f"local_{stamp}.db").write_bytes(b"")
    unrelated = backups / "keep-me.db"
    unrelated.write_bytes(b"")

    removed = sweep(backups, days=30, now=now)

    assert len(removed) == 2, f"expected the 31d and 400d copies to go, got {removed}"
    assert unrelated.exists(), "pruned a file it did not create"
    assert len(list(backups.glob("local_*.db"))) == 3


def test_backup_main_reports_failure_when_there_is_nothing_to_back_up(tmp_path):
    from scripts.backup_state import main

    empty = tmp_path / "state"
    empty.mkdir()
    assert main(["--state-dir", str(empty), "--backup-dir", str(tmp_path / "b")]) == 1


# --- what the runbook says is NOT automatic ----------------------------------


def test_a_halt_does_not_flatten_an_open_position(tmp_path):
    """RUNBOOK claim, pinned: HALT stops ORDERS, it does not sell anything.

    The gate's own message used to read 'FLATTEN AND HALT', which reads as a
    promise that the bot exits the market. It does not, and there is no code
    path that does. An operator who believes otherwise leaves a position on
    through a drawdown thinking it was closed.
    """
    ctx, cs = build(tmp_path)
    now = pd.Timestamp(cs[schema.TIMESTAMP].iloc[-1])

    run_cycle(ctx, now=now, reconcile_first=True)          # submits
    run_cycle(ctx, now=now + pd.Timedelta(hours=2))        # settles it
    held = ctx.broker.get_positions()
    assert held and held[0].units > 0, "expected an open position to test with"

    (tmp_path / "HALT").write_text("HALTED\nreason: max_drawdown_pct\n")
    with pytest.raises(CycleHalted):
        run_cycle(ctx, now=now + pd.Timedelta(hours=4))

    still = ctx.broker.get_positions()
    assert still and still[0].units == pytest.approx(held[0].units), (
        "the position changed during a halt — if flattening was implemented, "
        "the RUNBOOK's 'nothing flattens automatically' section is now wrong"
    )


def test_risk_breach_messages_do_not_promise_a_flatten():
    """Guard the wording itself: no message may claim an action we never take."""
    from trading_bot.risk import gate as gate_module

    source = Path(gate_module.__file__).read_text()
    assert "FLATTEN AND HALT" not in source, (
        "a risk message promises to flatten. Either implement flattening or "
        "say what actually happens."
    )
