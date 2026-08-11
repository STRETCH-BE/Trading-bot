#!/usr/bin/env python3
"""Drive an order lifecycle and die at a named point. Used by the crash tests.

Invoked as a SUBPROCESS and killed with os._exit(), which runs no cleanup, no
atexit hooks and no SQLite teardown — the closest reproduction of a real
SIGKILL or power loss available in-process. The parent then reopens the
database and asserts the invariants in StateStore.check_integrity().

Usage: crash_harness.py <db_path> <die_at>
  die_at ∈ {none, after_insert, after_partial_fill, after_full_fill,
            after_equity, mid_transaction}
"""

from __future__ import annotations

import os
import random
import sys

import pandas as pd

from trading_bot.broker.paper import PaperBroker, PaperBrokerConfig
from trading_bot.broker.state import StateStore
from trading_bot.execution.fill_model import FillModel
from trading_bot.execution.order import Order
from trading_bot.risk.approval import _mint_approved

TS = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
PRICE = 50_000.0


def die(point: str, target: str) -> None:
    if point == target:
        os._exit(9)  # no cleanup whatsoever


def main() -> int:
    db_path, die_at = sys.argv[1], sys.argv[2]
    store = StateStore(db_path)
    store.set_cash(10_000.0)

    fill_model = FillModel(maker_fee_bps=16.0, taker_fee_bps=26.0, slippage_bps=5.0)
    broker = PaperBroker(
        store,
        fill_model,
        lambda pair: PRICE,
        config=PaperBrokerConfig(
            partial_fill_probability=1.0,  # force a partial first
            rejection_probability=0.0,
            fill_latency_seconds=0.0,
            no_fill_probability=0.0,
            partial_fill_fraction=0.5,
        ),
        rng=random.Random(7),
        clock=lambda: TS,
    )

    order = Order(
        client_order_id="tb-crash", pair="XBTEUR", side="buy", units=0.02,
        order_type="market", limit_price=None, reason="crash test", timestamp=TS,
    )
    approved = _mint_approved(order, TS, 10_000.0, ("test",))

    broker.submit_order(approved)
    die("after_insert", die_at)

    if die_at == "mid_transaction":
        # Enter a transaction, write, and die BEFORE commit. SQLite must roll
        # this back on the next open — the row must not survive.
        #
        # `cm` MUST stay referenced: an unreferenced _GeneratorContextManager
        # is collected immediately, GeneratorExit fires the rollback path, and
        # the connection silently returns to autocommit — which would commit
        # the write this test is trying to strand.
        cm = store.transaction()
        conn = cm.__enter__()
        conn.execute(
            "INSERT INTO fills(order_id, timestamp, units, price, fee, liquidity) "
            "VALUES('tb-crash', ?, 999.0, 1.0, 0.0, 'taker')",
            (TS.isoformat(),),
        )
        assert store._conn.in_transaction, "not actually inside a transaction"
        os._exit(9)

    broker.settle(TS)  # first fill: forced partial
    die("after_partial_fill", die_at)

    # Complete the remainder: switch off partials so the order can reach FILLED.
    broker.config = PaperBrokerConfig(
        partial_fill_probability=0.0, rejection_probability=0.0,
        fill_latency_seconds=0.0, no_fill_probability=0.0,
    )
    broker.settle(TS)
    die("after_full_fill", die_at)

    broker.mark_to_market(TS)
    die("after_equity", die_at)

    print("completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
