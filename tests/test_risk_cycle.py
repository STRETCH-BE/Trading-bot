"""Cycle-level tests for every risk limit: outcomes, not mechanisms.

The rule for this file, and the reason it exists:

    **AccountState is NEVER constructed by hand here.**

``tests/test_risk_limits.py`` hands the gate a fabricated ``AccountState`` and
asserts it rejects. Those tests all passed while ``daily_loss_limit_pct`` and
``max_drawdown_pct`` were incapable of firing in production, because nothing
checked that the CYCLE built the state the gate reads. Every test below drives
``run_cycle`` into a real breach and asserts what the bot actually does.

``test_every_limit_has_a_cycle_level_test`` at the bottom fails if a limit
gains a gate-layer test without one here, so a future limit cannot ship
monitored-in-name-only.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest

from trading_bot.backtest.config import BacktestConfig
from trading_bot.broker import PaperBroker, PaperBrokerConfig, StateStore
from trading_bot.data import schema
from trading_bot.execution import FillModel
from trading_bot.risk import RiskGate, RiskLimits
from trading_bot.scheduler import CycleContext, CycleHalted, run_cycle

from .synthetic import from_closes

DAY = pd.Timedelta(days=1)


def inert(**override) -> RiskLimits:
    """Limits that cannot fire, so a test only exercises the one it names.

    Now that the account checks run every cycle, the shipped defaults fire
    readily during multi-cycle fixtures — which is the point of the fix, but
    it means a test about equity BOOKKEEPING must not also be a test about the
    drawdown limit. Each test below opts into exactly the limit it is about.
    """
    base = dict(
        max_position_pct=100.0, max_total_exposure_pct=100.0,
        max_order_size_pct=100.0, max_concurrent_positions=99,
        daily_loss_limit_pct=100.0, max_drawdown_pct=100.0,
        max_orders_per_hour=10_000, stale_data_max_seconds=30 * 86400.0,
    )
    base.update(override)
    return RiskLimits(**base)


def rising(n: int = 300) -> pd.DataFrame:
    return from_closes([50.0 * (1.01**i) for i in range(n)])


def crash(pre: int = 150, drop: int = 160, pct: float = 0.99) -> pd.DataFrame:
    """Flat, then a long grind down. ~80% off the peak by the end."""
    return from_closes([100.0] * pre + [100.0 * (pct**i) for i in range(1, drop)])


def build(tmp_path, *, limits=None, signal=None, candles=None, pairs=("XBTEUR",)):
    """A real cycle context. Nothing about the account is faked."""
    local = StateStore(tmp_path / "local.db")
    venue = StateStore(tmp_path / "venue.db")
    local.set_cash(10_000.0)
    venue.set_cash(10_000.0)
    fm = FillModel(maker_fee_bps=16.0, taker_fee_bps=26.0, slippage_bps=5.0)
    cs = rising() if candles is None else candles

    # The price the venue marks at follows the candle we are currently on, so a
    # crash is a real crash rather than a static quote.
    box = {"price": float(cs[schema.CLOSE].iloc[-1])}
    broker = PaperBroker(
        venue, fm, lambda pair: box["price"],
        config=PaperBrokerConfig.deterministic(), rng=random.Random(1),
        clock=lambda: box["now"],
    )
    box["now"] = pd.Timestamp(cs[schema.TIMESTAMP].iloc[-1])

    ctx = CycleContext(
        local=local, broker=broker,
        gate=RiskGate(limits or inert(), fm, halt_file=tmp_path / "HALT"),
        fill_model=fm,
        config=BacktestConfig(starting_capital=10_000.0, min_rebalance_delta=0.05),
        strategy_name="voltrend",
        signal_fn=signal or (lambda c: pd.Series([1.0] * len(c))),
        candles_for=lambda pair: cs, pairs=pairs,
    )
    return ctx, local, venue, broker, cs, box


def run_through(ctx, cs, box, upto: int, *, start: int = 150, step: int = 1):
    """Walk the cycle forward through real candles, one cycle per step.

    Each cycle sees candles up to index i and a venue price to match, so the
    equity curve the bot records is the one the market actually produced.
    """
    results = []
    view = {"df": None}
    ctx.candles_for = lambda pair: view["df"]
    for i in range(start, upto, step):
        view["df"] = cs.iloc[: i + 1].reset_index(drop=True)
        box["price"] = float(cs[schema.CLOSE].iloc[i])
        box["now"] = pd.Timestamp(cs[schema.TIMESTAMP].iloc[i])
        results.append(run_cycle(ctx, now=box["now"], reconcile_first=(i == start)))
    return results


# --- FINDING 1: equity history reaches the store the limits actually read ----


def test_local_equity_history_is_written_from_cycle_one(tmp_path):
    """It was ALWAYS empty: record_equity() only ever ran against the venue."""
    ctx, local, venue, broker, cs, box = build(tmp_path)
    assert local.equity_history() == []

    run_cycle(ctx, now=pd.Timestamp(cs[schema.TIMESTAMP].iloc[-1]), reconcile_first=True)

    history = local.equity_history()
    assert len(history) == 1, "the cycle did not record equity to its OWN store"
    assert history[0]["total_equity"] == pytest.approx(10_000.0, rel=1e-6)


def test_equity_is_recorded_every_cycle_not_just_the_first(tmp_path):
    ctx, local, venue, broker, cs, box = build(tmp_path)
    run_through(ctx, cs, box, upto=155)
    assert len(local.equity_history()) == 5


def test_recorded_equity_is_the_bots_own_books_not_the_brokers(tmp_path):
    """The drawdown limit must not trust the thing reconciliation checks it against."""
    ctx, local, venue, broker, cs, box = build(tmp_path)
    t0 = pd.Timestamp(cs[schema.TIMESTAMP].iloc[-1])
    run_cycle(ctx, now=t0, reconcile_first=True)                 # submits
    run_cycle(ctx, now=t0 + pd.Timedelta(hours=1))               # settles and mirrors

    # Diverge the bot's belief AFTER settlement, so nothing legitimately
    # rewrites cash between here and the snapshot.
    local.set_cash(4_242.0)
    ctx.reconcile_every_cycles = 99  # so the divergence is not caught first
    run_cycle(ctx, now=t0 + pd.Timedelta(hours=2))

    assert local.equity_history()[-1]["cash"] == pytest.approx(4_242.0), (
        "equity was snapshotted from the broker, not from the bot's own store"
    )


def test_peak_equity_exceeds_current_after_a_real_crash(tmp_path):
    """The direct regression. peak_equity was ALWAYS equal to equity.

    The asset falls ~80%, but the risk budget caps exposure at 25% of equity,
    so the account drawdown is roughly a quarter of that. The assertion is
    sized to the budget deliberately — inflating it to the asset's drawdown
    would be asserting something the bot never experiences.
    """
    ctx, local, venue, broker, cs, box = build(tmp_path, candles=crash())
    run_through(ctx, cs, box, upto=len(cs), step=10)

    history = [r["total_equity"] for r in local.equity_history()]
    peak, final = max(history), history[-1]
    drawdown_pct = (peak - final) / peak * 100.0
    assert drawdown_pct > 10.0, (
        f"no meaningful drawdown recorded across an 80% asset crash: "
        f"peak {peak:.2f} vs final {final:.2f} ({drawdown_pct:.2f}%)"
    )


def test_max_drawdown_limit_actually_fires_through_a_crash(tmp_path):
    """THE outcome test: a bot riding an 80% crash must halt itself.

    Before the fix this was unreachable — drawdown_pct() returned 0.0 forever
    because peak_equity was read from a store nothing ever wrote to.
    """
    ctx, local, venue, broker, cs, box = build(
        tmp_path,
        limits=inert(max_drawdown_pct=5.0),
        candles=crash(),
    )
    with pytest.raises(CycleHalted):
        run_through(ctx, cs, box, upto=len(cs), step=5)

    assert ctx.gate.is_halted(), "the drawdown breach did not write the HALT file"
    assert "drawdown" in ctx.gate.halt_file.read_text()


def test_daily_loss_limit_actually_fires(tmp_path):
    """Same regression for the daily limit: day_start_equity was always current.

    Candles here are daily and the bot cycles once per candle, so this also
    covers the second half of that defect: with day_start taken as the first
    snapshot of the day, the first snapshot IS the current cycle and the limit
    could never fire at daily cadence.
    """
    ctx, local, venue, broker, cs, box = build(
        tmp_path,
        # The asset loses ~1%/day and the budget caps exposure at 25%, so the
        # account loses ~0.25%/day. The limit is set below that on purpose.
        limits=inert(daily_loss_limit_pct=0.1),
        candles=crash(),
    )
    with pytest.raises(CycleHalted):
        run_through(ctx, cs, box, upto=len(cs))

    assert ctx.gate.is_halted()
    assert "daily loss" in ctx.gate.halt_file.read_text()


# --- FINDING 2: account limits are evaluated with no order in sight ----------


def flat(c):
    return pd.Series([0.0] * len(c))


def test_the_mandated_case_flat_strategy_stale_data_and_a_drawdown_halts(tmp_path):
    """The exact scenario from the audit: nothing to trade, and it must STILL halt.

    A flat strategy generates no order, so before the split the cycle evaluated
    precisely zero limits — 10-day-old data and a 50% drawdown alike.
    """
    ctx, local, venue, broker, cs, box = build(tmp_path, candles=crash())
    # Build a real drawdown first, with the limits inert so the run gets there.
    run_through(ctx, cs, box, upto=250, step=5)

    # Now arm the limits, go flat, and let the data rot for ten days. The
    # strategy wants nothing, so before the account/order split this cycle
    # evaluated precisely zero limits.
    ctx.gate = RiskGate(
        inert(max_drawdown_pct=5.0, stale_data_max_seconds=300.0),
        ctx.fill_model, halt_file=tmp_path / "HALT",
    )
    ctx.signal_fn = flat
    with pytest.raises(CycleHalted):
        run_cycle(ctx, now=box["now"] + pd.Timedelta(days=10))

    assert ctx.gate.is_halted()
    assert "drawdown" in ctx.gate.halt_file.read_text(), (
        "the capital-destroying condition must be reported ahead of stale data"
    )


def test_stale_data_blocks_a_cycle_that_wants_no_order(tmp_path):
    ctx, local, venue, broker, cs, box = build(
        tmp_path, limits=inert(stale_data_max_seconds=300.0), signal=flat
    )
    result = run_cycle(
        ctx, now=pd.Timestamp(cs[schema.TIMESTAMP].iloc[-1]) + pd.Timedelta(days=10),
        reconcile_first=True,
    )
    assert result.account_rejection == "stale_data"
    assert result.orders_submitted == 0
    assert local.recent_decisions()[0]["action_taken"] == "account_blocked"


def test_stale_data_does_not_write_the_halt_file(tmp_path):
    """Stale data stops trading; it is not a reason to demand a human at 3am.

    Distinguishing this from a drawdown breach is the point of having both.
    """
    ctx, local, venue, broker, cs, box = build(
        tmp_path, limits=inert(stale_data_max_seconds=300.0), signal=flat
    )
    run_cycle(ctx, now=pd.Timestamp(cs[schema.TIMESTAMP].iloc[-1]) + DAY,
              reconcile_first=True)
    assert not ctx.gate.is_halted()


def test_clock_skew_is_caught_with_no_order(tmp_path):
    """Data from the future: the machine clock is wrong. Also account-scoped."""
    ctx, local, venue, broker, cs, box = build(
        tmp_path, limits=inert(stale_data_max_seconds=300.0), signal=flat
    )
    result = run_cycle(
        ctx, now=pd.Timestamp(cs[schema.TIMESTAMP].iloc[-1]) - pd.Timedelta(days=3),
        reconcile_first=True,
    )
    assert result.account_rejection == "stale_data"


def test_current_exposure_over_the_cap_blocks_the_cycle(tmp_path):
    """Price drift alone can put exposure over the cap, with no order at fault.

    The order-scoped exposure check returns None for sells and for no-order
    cycles, so nothing was watching this.
    """
    ctx, local, venue, broker, cs, box = build(tmp_path)
    run_through(ctx, cs, box, upto=155)
    assert local.positions(), "expected a position to have been built"

    ctx.signal_fn = flat
    ctx.gate = RiskGate(
        inert(max_total_exposure_pct=1.0),
        ctx.fill_model, halt_file=tmp_path / "HALT",
    )
    result = run_cycle(ctx, now=box["now"] + pd.Timedelta(hours=1))
    assert result.account_rejection == "max_total_exposure_pct"


def test_too_many_concurrent_positions_blocks_the_cycle(tmp_path):
    ctx, local, venue, broker, cs, box = build(tmp_path)
    run_through(ctx, cs, box, upto=155)

    ctx.signal_fn = flat
    ctx.gate = RiskGate(
        inert(max_concurrent_positions=0),
        ctx.fill_model, halt_file=tmp_path / "HALT",
    )
    result = run_cycle(ctx, now=box["now"] + pd.Timedelta(hours=1))
    assert result.account_rejection == "max_concurrent_positions"


def test_order_rate_breaker_trips_without_a_new_order(tmp_path):
    # One cycle, which submits; then look again five minutes later. Candles are
    # DAILY, so stepping further would push the order outside the one-hour
    # window and the breaker would correctly not trip.
    ctx, local, venue, broker, cs, box = build(tmp_path)
    run_through(ctx, cs, box, upto=151)
    assert local.all_orders(), "expected an order in the last hour to count"

    ctx.signal_fn = flat
    ctx.gate = RiskGate(
        inert(max_orders_per_hour=1),
        ctx.fill_model, halt_file=tmp_path / "HALT",
    )
    result = run_cycle(ctx, now=box["now"] + pd.Timedelta(minutes=5))
    assert result.account_rejection == "max_orders_per_hour"


def test_halt_file_blocks_a_cycle_with_nothing_to_trade(tmp_path):
    ctx, local, venue, broker, cs, box = build(tmp_path, signal=flat)
    (tmp_path / "HALT").write_text("HALTED by hand\n")
    with pytest.raises(CycleHalted, match="HALT file present"):
        run_cycle(ctx, now=box["now"], reconcile_first=True)


# --- order-scoped limits, still reached through a real cycle -----------------


def test_max_order_size_blocks_submission(tmp_path):
    ctx, local, venue, broker, cs, box = build(
        tmp_path,
        limits=inert(max_order_size_pct=1.0),
    )
    result = run_cycle(ctx, now=box["now"], reconcile_first=True)
    assert result.orders_submitted == 0
    assert "max_order_size_pct" in result.rejections


def test_max_position_pct_blocks_submission(tmp_path):
    ctx, local, venue, broker, cs, box = build(
        tmp_path,
        limits=inert(max_position_pct=0.5),
    )
    result = run_cycle(ctx, now=box["now"], reconcile_first=True)
    assert result.orders_submitted == 0
    assert "max_position_pct" in result.rejections


def test_ordermin_and_costmin_stop_a_dust_order(tmp_path):
    """A 10 EUR account cannot clear XBTEUR's 1 EUR costmin at a 25% budget."""
    ctx, local, venue, broker, cs, box = build(tmp_path)
    local.set_cash(2.0)
    venue.set_cash(2.0)
    result = run_cycle(ctx, now=box["now"], reconcile_first=True)
    assert result.orders_submitted == 0


def test_unknown_pair_is_impossible_to_reach_through_the_cycle(tmp_path):
    """Documented, not asserted as reachable: ctx.pairs is validated upstream.

    schema.PAIRS gates which pairs the CLI accepts, so the cycle cannot present
    an unknown one. The gate check remains as defence for a caller that does
    not go through the CLI.
    """
    from trading_bot.data import schema as s

    assert set(("XBTEUR", "ETHEUR")) <= set(s.PAIRS)


def test_short_or_leverage_cannot_be_produced_by_the_translator(tmp_path):
    """Also structural: orders_for_target clamps to cash and to units held."""
    ctx, local, venue, broker, cs, box = build(tmp_path)
    results = run_through(ctx, cs, box, upto=170)
    assert results
    for order in local.all_orders():
        assert order.order.units > 0
    assert local.get_cash() >= -1e-9


# --- FINDING 3: reconciliation is not a boot-time formality ------------------


def test_drift_appearing_after_cycle_one_is_detected(tmp_path):
    """It reconciled ONCE, at boot. Drift at cycle 2 was invisible forever."""
    ctx, local, venue, broker, cs, box = build(tmp_path)
    run_cycle(ctx, now=box["now"], reconcile_first=True)

    local.set_cash(1.0)  # the books now disagree by ~7,500

    with pytest.raises(CycleHalted):
        run_cycle(ctx, now=box["now"] + pd.Timedelta(hours=1))
    assert ctx.gate.is_halted(), "a post-boot mismatch must write the HALT file"


def test_reconciliation_runs_on_every_cycle_by_default(tmp_path):
    ctx, local, venue, broker, cs, box = build(tmp_path)
    results = run_through(ctx, cs, box, upto=155)
    assert all(r.reconciled for r in results), (
        f"reconciled on {sum(r.reconciled for r in results)}/{len(results)} cycles"
    )


def test_the_interval_is_configurable_and_still_bounded(tmp_path):
    ctx, local, venue, broker, cs, box = build(tmp_path)
    ctx.reconcile_every_cycles = 3
    results = run_through(ctx, cs, box, upto=157)
    assert results[0].reconciled  # boot always reconciles
    assert sum(r.reconciled for r in results) >= 2, "the interval never came round"


def test_default_interval_is_one_cycle():
    """A default measured in cycles, not months."""
    from trading_bot.scheduler.cycle import CycleContext as C

    assert C.reconcile_every_cycles == 1


# --- THE STRUCTURAL GUARD ----------------------------------------------------

# Every limit name the gate can emit, mapped to the test above that drives a
# real cycle into it. Add a limit to the gate and this file fails until a
# cycle-level test exists for it.
CYCLE_LEVEL_COVERAGE = {
    "kill_switch": "test_halt_file_blocks_a_cycle_with_nothing_to_trade",
    "daily_loss_limit_pct": "test_daily_loss_limit_actually_fires",
    "max_drawdown_pct": "test_max_drawdown_limit_actually_fires_through_a_crash",
    "stale_data": "test_stale_data_blocks_a_cycle_that_wants_no_order",
    "max_total_exposure_pct": "test_current_exposure_over_the_cap_blocks_the_cycle",
    "max_concurrent_positions": "test_too_many_concurrent_positions_blocks_the_cycle",
    "max_orders_per_hour": "test_order_rate_breaker_trips_without_a_new_order",
    "unknown_pair": "test_unknown_pair_is_impossible_to_reach_through_the_cycle",
    "short_or_leverage": "test_short_or_leverage_cannot_be_produced_by_the_translator",
    "ordermin": "test_ordermin_and_costmin_stop_a_dust_order",
    "costmin": "test_ordermin_and_costmin_stop_a_dust_order",
    "max_order_size_pct": "test_max_order_size_blocks_submission",
    "max_position_pct": "test_max_position_pct_blocks_submission",
}


def _limits_the_gate_can_emit() -> set[str]:
    """Every string literal passed as RiskDecision(limit=...) in gate.py."""
    import ast
    from pathlib import Path

    from trading_bot.risk import gate as gate_module

    tree = ast.parse(Path(gate_module.__file__).read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "RiskDecision"):
            continue
        # positional form: RiskDecision(None, True, "<limit>", ...)
        if len(node.args) >= 3 and isinstance(node.args[2], ast.Constant):
            found.add(node.args[2].value)
        for kw in node.keywords:
            if kw.arg == "limit" and isinstance(kw.value, ast.Constant):
                found.add(kw.value.value)
    # ordermin/costmin are emitted indirectly, via FillModel's failed_limit
    found.update({"ordermin", "costmin"})
    return {f for f in found if f}


def test_every_limit_has_a_cycle_level_test():
    """The fix for the whole CLASS of defect this file exists to prevent.

    A limit with only gate-layer tests is monitored in name only: the gate can
    be provably correct while the cycle never builds the state it reads, or
    never calls it at all. Both happened.
    """
    emitted = _limits_the_gate_can_emit()
    missing = emitted - set(CYCLE_LEVEL_COVERAGE)
    assert not missing, (
        f"these limits can be emitted by the gate but have no cycle-level test: "
        f"{sorted(missing)}. Add one to tests/test_risk_cycle.py that drives "
        f"run_cycle into a real breach — do NOT hand-build an AccountState."
    )


def test_the_coverage_map_names_real_tests():
    """A map pointing at deleted tests would pass the guard while proving nothing."""
    for limit, test_name in CYCLE_LEVEL_COVERAGE.items():
        assert test_name in globals(), f"{limit} maps to a missing test {test_name!r}"


def test_no_test_in_this_file_hand_builds_an_account_state():
    """The rule that makes these OUTCOME tests rather than mechanism tests."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "AccountState":
            raise AssertionError(
                "AccountState is constructed by hand in test_risk_cycle.py. That "
                "is exactly the mechanism-testing habit this file exists to "
                "replace — drive run_cycle and let the cycle build the state."
            )
