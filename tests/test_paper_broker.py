"""PaperBroker: interface, imperfection, idempotency, and fill equivalence.

The central test here is ``test_paper_broker_fills_are_byte_identical_to_the_engine``
— the standing regression guard for audit finding 3. If anyone ever gives the
broker its own copy of the cost arithmetic, that test fails.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest

from trading_bot.backtest import BacktestConfig, backtest
from trading_bot.broker import (
    OrderStatus,
    PaperBroker,
    PaperBrokerConfig,
    StateStore,
    UnknownOrderError,
)
from trading_bot.broker.state import Decision
from trading_bot.execution import FillModel, Order
from trading_bot.risk import ForgedApprovalError
from trading_bot.risk.approval import _mint_approved
from trading_bot.strategies.voltrend import VolTrendParams, voltrend

from .synthetic import from_closes

TS = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
PRICE = 50_000.0


def fill_model() -> FillModel:
    return FillModel(maker_fee_bps=16.0, taker_fee_bps=26.0, slippage_bps=5.0)


def make_broker(tmp_path, *, config=None, seed=1, price=PRICE, fm=None, clock=None):
    store = StateStore(tmp_path / "state.db")
    store.set_cash(10_000.0)
    return PaperBroker(
        store,
        fm or fill_model(),
        lambda pair: price,
        config=config or PaperBrokerConfig.deterministic(),
        rng=random.Random(seed),
        clock=clock or (lambda: TS),
    )


def order(**kw) -> Order:
    base = dict(
        client_order_id="tb-1", pair="XBTEUR", side="buy", units=0.01,
        order_type="market", limit_price=None, reason="test", timestamp=TS,
    )
    base.update(kw)
    return Order(**base)


def approve(o: Order):
    return _mint_approved(o, TS, 10_000.0, ("test",))


# --- THE finding-3 regression guard -----------------------------------------


def test_paper_broker_fills_are_byte_identical_to_the_engine(tmp_path):
    """Same FillModel object, same orders -> the same prices and fees, exactly.

    Not approximately: `==` on floats. Any reimplementation of fee or slippage
    arithmetic inside the broker shows up here immediately.
    """
    shared = fill_model()  # ONE instance, handed to both

    candles = from_closes([100.0 * (1.004**i) for i in range(200)])
    config = BacktestConfig(
        starting_capital=10_000.0, maker_fee_bps=16.0, taker_fee_bps=26.0,
        fee_mode="taker", slippage_bps=5.0, min_rebalance_delta=0.05,
    )
    engine_result = backtest(
        candles, lambda c: voltrend(c, VolTrendParams()), config, pair="XBTEUR"
    )
    assert engine_result.fills, "no engine fills to compare against"

    for i, engine_fill in enumerate(engine_result.fills):
        broker = make_broker(
            tmp_path / f"b{i}", fm=shared, price=engine_fill.reference_price
        )
        o = order(
            client_order_id=f"tb-eq-{i}", side=engine_fill.side,
            units=engine_fill.units,
        )
        broker.submit_order(approve(o))
        broker.settle(TS)

        rows = broker.store.fills_for(o.client_order_id)
        assert len(rows) == 1
        assert rows[0]["price"] == engine_fill.price, (
            f"fill {i}: broker price {rows[0]['price']!r} != engine "
            f"{engine_fill.price!r} — the cost arithmetic has diverged"
        )
        assert rows[0]["fee"] == engine_fill.fee, (
            f"fill {i}: broker fee {rows[0]['fee']!r} != engine {engine_fill.fee!r}"
        )


def test_broker_and_engine_share_the_object_not_a_copy(tmp_path):
    shared = fill_model()
    broker = make_broker(tmp_path, fm=shared)
    assert broker.fill_model is shared


# --- approval enforcement ----------------------------------------------------


def test_raw_order_cannot_be_submitted(tmp_path):
    broker = make_broker(tmp_path)
    with pytest.raises(ForgedApprovalError, match="Raw orders may not reach"):
        broker.submit_order(order())  # type: ignore[arg-type]


def test_forged_approval_cannot_be_submitted(tmp_path):
    from trading_bot.risk import ApprovedOrder

    broker = make_broker(tmp_path)
    forged = object.__new__(ApprovedOrder)
    for name, value in [
        ("order", order()), ("approved_at", TS), ("equity_at_approval", 10_000.0),
        ("approval_signature", "0" * 64), ("checks_passed", ()),
    ]:
        object.__setattr__(forged, name, value)
    with pytest.raises(ForgedApprovalError):
        broker.submit_order(forged)


# --- idempotency -------------------------------------------------------------


def test_resubmitting_the_same_decision_adopts_rather_than_duplicates(tmp_path):
    """The crash-recovery contract: same client_order_id, one order."""
    broker = make_broker(tmp_path)
    o = order()
    first = broker.submit_order(approve(o))
    second = broker.submit_order(approve(o))

    assert first.order.client_order_id == second.order.client_order_id
    assert len(broker.store.all_orders()) == 1
    broker.settle(TS)
    assert len(broker.store.fills_for(o.client_order_id)) == 1


def test_adoption_reports_the_current_state_not_a_fresh_one(tmp_path):
    broker = make_broker(tmp_path)
    o = order()
    broker.submit_order(approve(o))
    broker.settle(TS)
    adopted = broker.submit_order(approve(o))
    assert adopted.status is OrderStatus.FILLED
    assert adopted.filled_units == pytest.approx(0.01)


# --- interface ---------------------------------------------------------------


def test_balance_positions_and_open_orders(tmp_path):
    broker = make_broker(tmp_path)
    assert broker.get_balance().cash == 10_000.0
    assert broker.get_positions() == []
    assert broker.get_open_orders() == []

    broker.submit_order(approve(order()))
    assert len(broker.get_open_orders()) == 1

    broker.settle(TS)
    assert broker.get_open_orders() == []
    positions = broker.get_positions()
    assert len(positions) == 1
    assert positions[0].pair == "XBTEUR"
    assert positions[0].units == pytest.approx(0.01)
    assert broker.get_balance().cash < 10_000.0


def test_unknown_order_status_raises(tmp_path):
    broker = make_broker(tmp_path)
    with pytest.raises(UnknownOrderError):
        broker.get_order_status("tb-nope")


def test_cancel_order(tmp_path):
    broker = make_broker(tmp_path, config=PaperBrokerConfig(fill_latency_seconds=60))
    broker.submit_order(approve(order()))
    broker.cancel_order("tb-1")
    assert broker.get_order_status("tb-1").status is OrderStatus.CANCELLED
    broker.settle(TS + pd.Timedelta(minutes=5))
    assert broker.store.fills_for("tb-1") == []


def test_cancelling_a_filled_order_is_a_noop(tmp_path):
    broker = make_broker(tmp_path)
    broker.submit_order(approve(order()))
    broker.settle(TS)
    broker.cancel_order("tb-1")
    assert broker.get_order_status("tb-1").status is OrderStatus.FILLED


def test_mark_to_market_records_equity(tmp_path):
    broker = make_broker(tmp_path)
    broker.submit_order(approve(order()))
    broker.settle(TS)
    equity = broker.mark_to_market(TS)
    history = broker.store.equity_history()
    assert len(history) == 1
    assert history[0]["total_equity"] == pytest.approx(equity)


# --- simulated imperfection --------------------------------------------------


def test_defaults_are_deliberately_imperfect():
    """A broker that always fills perfectly teaches nothing."""
    c = PaperBrokerConfig()
    assert c.partial_fill_probability == 0.1
    assert c.rejection_probability == 0.02
    assert c.fill_latency_seconds == 2.0
    assert c.no_fill_probability == 0.15


def test_fill_latency_delays_the_fill(tmp_path):
    broker = make_broker(tmp_path, config=PaperBrokerConfig(
        partial_fill_probability=0.0, rejection_probability=0.0,
        fill_latency_seconds=2.0, no_fill_probability=0.0,
    ))
    broker.submit_order(approve(order()))

    broker.settle(TS)  # too early
    assert broker.get_order_status("tb-1").status is OrderStatus.OPEN

    broker.settle(TS + pd.Timedelta(seconds=2))
    assert broker.get_order_status("tb-1").status is OrderStatus.FILLED


def test_partial_fill_leaves_the_remainder_working(tmp_path):
    broker = make_broker(tmp_path, config=PaperBrokerConfig(
        partial_fill_probability=1.0, rejection_probability=0.0,
        fill_latency_seconds=0.0, no_fill_probability=0.0,
        partial_fill_fraction=0.5,
    ))
    broker.submit_order(approve(order(units=0.02)))
    state = broker.settle(TS)[0]

    assert state.status is OrderStatus.PARTIALLY_FILLED
    assert state.filled_units == pytest.approx(0.01)
    assert state.remaining_units == pytest.approx(0.01)
    assert len(broker.get_open_orders()) == 1


def test_rejection_is_recorded_and_never_fills(tmp_path):
    broker = make_broker(tmp_path, config=PaperBrokerConfig(
        rejection_probability=1.0, fill_latency_seconds=0.0,
    ))
    state = broker.submit_order(approve(order()))
    assert state.status is OrderStatus.REJECTED
    assert "rejection" in state.rejection_reason
    broker.settle(TS)
    assert broker.store.fills_for("tb-1") == []


def test_limit_order_may_never_fill(tmp_path):
    broker = make_broker(tmp_path, config=PaperBrokerConfig(
        no_fill_probability=1.0, rejection_probability=0.0, fill_latency_seconds=0.0,
    ))
    o = order(order_type="limit", limit_price=PRICE)
    broker.submit_order(approve(o))
    broker.settle(TS + pd.Timedelta(hours=1))
    assert broker.get_order_status("tb-1").status is OrderStatus.OPEN
    assert broker.store.fills_for("tb-1") == []


def test_limit_orders_are_charged_maker_fees(tmp_path):
    broker = make_broker(tmp_path, config=PaperBrokerConfig.deterministic())
    o = order(order_type="limit", limit_price=PRICE)
    broker.submit_order(approve(o))
    broker.settle(TS)
    assert broker.store.fills_for("tb-1")[0]["liquidity"] == "maker"


def test_imperfection_is_reproducible_with_a_seed(tmp_path):
    def run(path):
        b = make_broker(path, config=PaperBrokerConfig(), seed=99)
        results = []
        for i in range(20):
            o = order(client_order_id=f"tb-{i}")
            results.append(b.submit_order(approve(o)).status)
        return results

    assert run(tmp_path / "a") == run(tmp_path / "b")


# --- decisions log -----------------------------------------------------------


def test_decisions_table_records_do_nothing(tmp_path):
    """"Why did it do that in March" must be answerable, including inaction."""
    store = StateStore(tmp_path / "s.db")
    store.record_decision(Decision(
        timestamp=TS, strategy="voltrend", pair="XBTEUR",
        current_position=0.4, target_position=0.42,
        action_taken="none",
        reasoning="target move 0.02 inside dead-band 0.05",
        cycle_id="cycle-1",
    ))
    rows = store.recent_decisions()
    assert len(rows) == 1
    assert rows[0]["action_taken"] == "none"
    assert "dead-band" in rows[0]["reasoning"]
    assert rows[0]["cycle_id"] == "cycle-1"


def test_recent_decisions_returns_newest_first(tmp_path):
    store = StateStore(tmp_path / "s.db")
    for i in range(10):
        store.record_decision(Decision(
            timestamp=TS + pd.Timedelta(days=i), strategy="voltrend", pair="XBTEUR",
            current_position=0.0, target_position=0.0, action_taken="none",
            reasoning=f"r{i}", cycle_id=f"c{i}",
        ))
    rows = store.recent_decisions(limit=5)
    assert [r["reasoning"] for r in rows] == ["r9", "r8", "r7", "r6", "r5"]


# --- durability --------------------------------------------------------------


def test_state_survives_a_broker_restart(tmp_path):
    broker = make_broker(tmp_path)
    broker.submit_order(approve(order()))
    broker.settle(TS)
    cash, units = broker.get_balance().cash, broker.get_positions()[0].units
    broker.store.close()

    reopened = PaperBroker(
        StateStore(tmp_path / "state.db"), fill_model(), lambda p: PRICE,
        config=PaperBrokerConfig.deterministic(), rng=random.Random(1),
        clock=lambda: TS,
    )
    assert reopened.get_balance().cash == pytest.approx(cash)
    assert reopened.get_positions()[0].units == pytest.approx(units)
    assert reopened.get_order_status("tb-1").status is OrderStatus.FILLED


def test_transaction_helper_must_be_used_as_a_context_manager(tmp_path):
    """Landmine guard, found while writing the crash harness.

    `store.transaction().__enter__()` leaves the context manager unreferenced;
    CPython collects it at once, GeneratorExit fires the rollback path, and
    the connection silently returns to AUTOCOMMIT — so the next write commits
    when the caller believes it is staged. Always `with store.transaction():`.
    """
    store = StateStore(tmp_path / "s.db")
    store.set_cash(1.0)

    cm = store.transaction()  # held -> genuinely inside a transaction
    cm.__enter__()
    assert store._conn.in_transaction
    cm.__exit__(None, None, None)

    store.transaction().__enter__()  # unreferenced -> collected, rolled back
    assert not store._conn.in_transaction
