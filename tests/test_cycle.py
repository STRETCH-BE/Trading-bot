"""Stage 6c: the cycle loop, reconciliation, and the run.py entry point."""

from __future__ import annotations

import random
from dataclasses import replace

import pandas as pd
import pytest

from trading_bot.backtest.config import BacktestConfig
from trading_bot.broker import OrderStatus, PaperBroker, PaperBrokerConfig, StateStore
from trading_bot.data import schema
from trading_bot.execution import FillModel
from trading_bot.risk import RiskGate, RiskLimits
from trading_bot.scheduler import CycleContext, CycleHalted, reconcile, run_cycle
from trading_bot.strategies.voltrend import VolTrendParams, voltrend

from .synthetic import from_closes

PRICE = 100.0

# Most cycle tests use permissive limits so a rejection never masks a loop bug.
# The interaction between the shipped RiskLimits() and the strategy's target is
# tested on its own, below, under the real defaults.
PERMISSIVE = RiskLimits(
    max_position_pct=100.0, max_total_exposure_pct=100.0, max_order_size_pct=100.0,
    # These tests simulate "the next cycle" by advancing `now` while the candle
    # fixture stays put, which is genuinely stale data — and since the account
    # checks became unconditional, the real 300s limit would fire on every
    # multi-cycle test. Freshness has its own cycle-level test in
    # tests/test_risk_cycle.py, at the shipped limit.
    stale_data_max_seconds=30 * 86400.0,
)


def candles_rising(n: int = 300, end: pd.Timestamp | None = None) -> pd.DataFrame:
    df = from_closes([50.0 * (1.01**i) for i in range(n)])
    if end is not None:
        step = pd.Timedelta(days=1)
        df[schema.TIMESTAMP] = pd.Series(
            [end - step * (n - 1 - i) for i in range(n)], dtype="datetime64[us, UTC]"
        )
    return df


def build(tmp_path, *, broker_config=None, dry_run=False, candles=None, seed=1):
    """Local store and venue store are SEPARATE — that is what makes
    reconciliation able to detect drift rather than compare a thing to itself."""
    local = StateStore(tmp_path / "local.db")
    venue = StateStore(tmp_path / "venue.db")
    local.set_cash(10_000.0)
    venue.set_cash(10_000.0)

    fm = FillModel(maker_fee_bps=16.0, taker_fee_bps=26.0, slippage_bps=5.0)
    cs = candles if candles is not None else candles_rising()
    broker = PaperBroker(
        venue, fm, lambda pair: float(cs[schema.CLOSE].iloc[-1]),
        config=broker_config or PaperBrokerConfig.deterministic(),
        rng=random.Random(seed),
        clock=lambda: pd.Timestamp(cs[schema.TIMESTAMP].iloc[-1]),
    )
    config = BacktestConfig(starting_capital=10_000.0, min_rebalance_delta=0.05)
    ctx = CycleContext(
        local=local, broker=broker,
        gate=RiskGate(PERMISSIVE, fm, halt_file=tmp_path / "HALT"),
        fill_model=fm, config=config, strategy_name="voltrend",
        signal_fn=lambda c: voltrend(c, VolTrendParams()),
        candles_for=lambda pair: cs, pairs=("XBTEUR",), dry_run=dry_run,
    )
    return ctx, local, venue, broker, cs


def now_of(cs) -> pd.Timestamp:
    return pd.Timestamp(cs[schema.TIMESTAMP].iloc[-1])


# --- ADDITION 1: settle happens first, fills land in the NEXT cycle ----------


def test_order_submitted_in_cycle_N_fills_in_cycle_N_plus_1(tmp_path):
    """Latency means a fill lands in a later cycle. Both the position state and
    the decisions table must attribute it to the correct cycle."""
    ctx, local, venue, broker, cs = build(
        tmp_path,
        broker_config=PaperBrokerConfig(
            partial_fill_probability=0.0, rejection_probability=0.0,
            no_fill_probability=0.0, fill_latency_seconds=3600.0,  # fills next cycle
        ),
    )
    t0 = now_of(cs)

    # cycle N: submits, but the latency means nothing fills yet
    c1 = run_cycle(ctx, now=t0, reconcile_first=True)
    assert c1.orders_submitted == 1
    assert c1.settled == 0
    assert broker.get_positions() == [], "filled during the submitting cycle"

    # cycle N+1, an hour later: settle() at the top applies the fill
    c2 = run_cycle(ctx, now=t0 + pd.Timedelta(hours=2))
    assert c2.settled == 1, "the pending fill did not land at the start of cycle N+1"
    assert c2.cycle_id != c1.cycle_id

    positions = broker.get_positions()
    assert len(positions) == 1 and positions[0].units > 0

    # decisions attribute correctly: the submit to N, and N+1 sees the position
    rows = local._conn.execute(
        "SELECT cycle_id, action_taken, current_position FROM decisions ORDER BY id"
    ).fetchall()
    submit_rows = [r for r in rows if r["action_taken"] == "submitted"]
    assert submit_rows[0]["cycle_id"] == c1.cycle_id
    assert submit_rows[0]["current_position"] == pytest.approx(0.0), (
        "cycle N recorded a position it did not yet hold"
    )
    later = [r for r in rows if r["cycle_id"] == c2.cycle_id]
    assert later, "cycle N+1 recorded no decision"
    assert later[0]["current_position"] > 0.0, (
        "cycle N+1 did not see the position the fill created"
    )


def test_settle_runs_before_positions_are_read(tmp_path):
    """Reading positions before settling would size against a stale book."""
    ctx, local, venue, broker, cs = build(
        tmp_path,
        broker_config=PaperBrokerConfig(
            partial_fill_probability=0.0, rejection_probability=0.0,
            no_fill_probability=0.0, fill_latency_seconds=0.0,
        ),
    )
    t0 = now_of(cs)
    run_cycle(ctx, now=t0, reconcile_first=True)
    second = run_cycle(ctx, now=t0 + pd.Timedelta(days=1))
    # by the second cycle the fill is visible and no duplicate buy is issued
    assert second.orders_submitted == 0
    assert broker.get_positions()[0].units > 0


def test_fills_are_mirrored_into_local_state(tmp_path):
    ctx, local, venue, broker, cs = build(tmp_path)
    t0 = now_of(cs)
    run_cycle(ctx, now=t0, reconcile_first=True)
    run_cycle(ctx, now=t0 + pd.Timedelta(days=1))

    assert local.positions(), "local belief never learned about the fill"
    assert reconcile(local, broker).agrees


# --- decisions: every evaluation, including inaction -------------------------


def test_every_cycle_records_a_decision_even_when_doing_nothing(tmp_path):
    flat = from_closes([100.0] * 300)  # voltrend stays flat on a constant tape
    ctx, local, venue, broker, cs = build(tmp_path, candles=flat)
    result = run_cycle(ctx, now=now_of(cs), reconcile_first=True)

    assert result.orders_submitted == 0
    assert result.decisions == 1
    row = local.recent_decisions()[0]
    assert row["action_taken"] == "none"
    assert row["cycle_id"] == result.cycle_id
    assert row["reasoning"]


def test_dry_run_submits_nothing(tmp_path):
    ctx, local, venue, broker, cs = build(tmp_path, dry_run=True)
    result = run_cycle(ctx, now=now_of(cs), reconcile_first=True)

    assert result.orders_submitted == 0
    assert venue.all_orders() == []
    assert broker.get_positions() == []
    row = local.recent_decisions()[0]
    assert row["action_taken"] == "dry_run"
    assert "WOULD SUBMIT" in row["reasoning"]


# --- risk gate is in the path ------------------------------------------------


def test_a_risk_rejection_stops_submission_and_is_recorded(tmp_path):
    ctx, local, venue, broker, cs = build(tmp_path)
    ctx.gate = RiskGate(
        RiskLimits(max_position_pct=0.5, max_order_size_pct=100.0),
        ctx.fill_model, halt_file=tmp_path / "HALT",
    )
    result = run_cycle(ctx, now=now_of(cs), reconcile_first=True)

    assert result.orders_submitted == 0
    assert "max_position_pct" in result.rejections
    assert venue.all_orders() == []
    assert local.recent_decisions()[0]["action_taken"] == "rejected"


def test_halt_file_blocks_the_cycle(tmp_path):
    ctx, local, venue, broker, cs = build(tmp_path)
    (tmp_path / "HALT").write_text("stopped")
    with pytest.raises(CycleHalted):
        run_cycle(ctx, now=now_of(cs), reconcile_first=True)
    assert venue.all_orders() == []


# --- reconciliation ----------------------------------------------------------


def test_reconciliation_agrees_on_a_clean_start(tmp_path):
    ctx, local, venue, broker, cs = build(tmp_path)
    report = reconcile(local, broker)
    assert report.agrees
    assert "AGREE" in report.render()


def test_reconciliation_detects_a_position_mismatch(tmp_path):
    ctx, local, venue, broker, cs = build(tmp_path)
    with local.transaction() as c:
        c.execute(
            "INSERT INTO positions(pair, units, avg_cost, opened_at) "
            "VALUES('XBTEUR', 5.0, 100.0, '2026-01-01T00:00:00+00:00')"
        )
    report = reconcile(local, broker)
    assert not report.agrees
    assert any(d.kind == "position" and d.key == "XBTEUR" for d in report.discrepancies)
    assert "HALTING" in report.render()


def test_reconciliation_detects_a_cash_mismatch(tmp_path):
    ctx, local, venue, broker, cs = build(tmp_path)
    local.set_cash(9_000.0)
    report = reconcile(local, broker)
    assert not report.agrees
    assert any(d.kind == "cash" for d in report.discrepancies)


def test_reconciliation_detects_an_orphan_open_order(tmp_path):
    ctx, local, venue, broker, cs = build(tmp_path)
    from trading_bot.execution import Order

    ghost = Order(
        client_order_id="tb-ghost", pair="XBTEUR", side="buy", units=0.01,
        order_type="market", limit_price=None, reason="x", timestamp=now_of(cs),
    )
    local.insert_order(ghost, OrderStatus.OPEN, now_of(cs))
    report = reconcile(local, broker)
    assert not report.agrees
    assert any(d.kind == "open_order" for d in report.discrepancies)


def test_mismatch_halts_the_cycle_and_writes_the_halt_file(tmp_path):
    ctx, local, venue, broker, cs = build(tmp_path)
    local.set_cash(1.0)
    with pytest.raises(CycleHalted):
        run_cycle(ctx, now=now_of(cs), reconcile_first=True)
    assert ctx.gate.is_halted(), "a mismatch must halt, not merely warn"


def test_reconcile_module_is_broker_agnostic():
    """ADDITION 3: no broker is special-cased. Structural, not aspirational."""
    import ast
    import importlib
    import pathlib

    mod = importlib.import_module("trading_bot.scheduler.reconcile")
    source = pathlib.Path(mod.__file__).read_text()
    tree = ast.parse(source)

    # Only real imports count; the module docstring naturally names the brokers
    # it must not depend on.
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
    for banned in ("PaperBroker", "KrakenBroker", "paper", "kraken"):
        assert not any(banned.lower() in i.lower() for i in imported), (
            f"reconcile.py imports {banned!r} — it must work against the "
            f"Broker ABC alone. Imports: {imported}"
        )

    # and no runtime type-sniffing of the broker
    code_only = "\n".join(
        ln for ln in source.splitlines() if not ln.strip().startswith("#")
    )
    assert "isinstance(broker" not in code_only


# --- run.py entry point ------------------------------------------------------


def test_live_requires_both_the_flag_and_the_env_var(monkeypatch):
    from trading_bot.run import build_parser, resolve_mode

    monkeypatch.delenv("LIVE_TRADING", raising=False)
    args = build_parser().parse_args(["--live"])
    with pytest.raises(SystemExit, match="LIVE_TRADING=yes"):
        resolve_mode(args)


def test_live_with_env_var_still_refuses_because_no_live_broker_exists(monkeypatch):
    from trading_bot.run import build_parser, resolve_mode

    monkeypatch.setenv("LIVE_TRADING", "yes")
    args = build_parser().parse_args(["--live"])
    with pytest.raises(SystemExit, match="no live broker exists"):
        resolve_mode(args)


def test_default_mode_is_paper():
    from trading_bot.run import build_parser, resolve_mode

    assert resolve_mode(build_parser().parse_args([])) == "paper"
    assert resolve_mode(build_parser().parse_args(["--paper"])) == "paper"


def test_run_module_is_executable_as_main():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "trading_bot.run", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "--reconcile-only" in out.stdout
    assert "--dry-run" in out.stdout
    assert "--once" in out.stdout


def test_RESOLVED_full_strategy_target_is_accepted_under_shipped_risk_limits(tmp_path):
    """The former FINDING: voltrend targets 1.0, the shipped limits cap a
    position at 25% of equity, so every cycle was rejected.

    Resolved by ``strategy_max_allocation``: a target of 1.0 now means 'fully
    allocated within my risk budget' = 25% of equity. The risk limits were NOT
    weakened — this runs against the shipped ``RiskLimits()`` defaults.
    """
    ctx, local, venue, broker, cs = build(tmp_path)
    ctx.gate = RiskGate(RiskLimits(), ctx.fill_model, halt_file=tmp_path / "HALT")

    result = run_cycle(ctx, now=now_of(cs), reconcile_first=True)

    assert result.signal == pytest.approx(1.0), "strategy asked for a full position"
    assert result.rejections == []
    assert result.orders_submitted == 1

    # and it asked for a QUARTER of the account, not all of it
    decision = local.recent_decisions()[0]
    assert decision["action_taken"] == "submitted"
    assert decision["target_position"] == pytest.approx(
        ctx.config.strategy_max_allocation
    )
    submitted = local.all_orders()[0].order
    notional = submitted.units * float(cs[schema.CLOSE].iloc[-1])
    assert notional == pytest.approx(10_000.0 * 0.25, rel=0.02)


def test_decision_row_logs_both_positions_in_equity_units(tmp_path):
    """current_position and target_position must be the SAME unit.

    The mapping makes it easy to log the raw signal (1.0) beside an equity
    fraction (0.25); at 3am that reads as 'it wants to quadruple' when the bot
    is in fact fully allocated and about to do nothing. The raw signal belongs
    in the reasoning text, where it is labelled.
    """
    ctx, local, venue, broker, cs = build(tmp_path)
    t0 = now_of(cs)
    run_cycle(ctx, now=t0, reconcile_first=True)
    run_cycle(ctx, now=t0 + pd.Timedelta(hours=2))  # now holding: expect 'none'

    row = local.recent_decisions()[0]
    assert row["action_taken"] == "none", "expected the second cycle to sit still"
    assert row["target_position"] == pytest.approx(0.25, abs=0.01)
    assert row["current_position"] == pytest.approx(row["target_position"], abs=0.02), (
        "the two columns disagree — they are not in the same unit"
    )
    assert "signal 1.0000 x budget 0.2500 = 0.2500 of equity" in row["reasoning"]


def test_the_gate_still_bites_if_the_allocation_mapping_is_removed(tmp_path):
    """Teeth check for the test above: the conflict was resolved by the mapping,
    not by the gate having gone quiet. Put the target back to 1.0-of-equity and
    the shipped limits must reject it exactly as they did before."""
    ctx, local, venue, broker, cs = build(tmp_path)
    ctx.gate = RiskGate(RiskLimits(), ctx.fill_model, halt_file=tmp_path / "HALT")
    ctx.config = replace(ctx.config, strategy_max_allocation=1.0)

    result = run_cycle(ctx, now=now_of(cs), reconcile_first=True)

    assert result.orders_submitted == 0
    assert result.rejections == ["max_order_size_pct"]
    assert local.recent_decisions()[0]["action_taken"] == "rejected"
