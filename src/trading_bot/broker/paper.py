"""PaperBroker: simulated fills, deliberately imperfect, durably recorded.

Two design commitments worth stating plainly.

**It shares the engine's FillModel object.** Not a copy, not a reimplementation
— the same instance is passed to both. ``tests/test_paper_broker.py`` asserts
byte-identical fills for the same order sequence, which is the standing
regression guard for audit finding 3.

**The defaults are annoying on purpose.** A paper broker that fills instantly
and completely teaches nothing about the failure paths that appear live:
partial fills, outright rejections, latency between submit and fill, and limit
orders that simply never fill. Those are the paths that hurt in production, so
paper trading must exercise them. Turn them off only for equivalence testing.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import pandas as pd

from trading_bot.broker.base import (
    Balance,
    Broker,
    OrderState,
    OrderStatus,
    Position,
    UnknownOrderError,
)
from trading_bot.broker.state import StateStore
from trading_bot.execution.fill_model import FillModel
from trading_bot.risk.approval import ApprovedOrder, verify_approval

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaperBrokerConfig:
    """Simulated imperfection. Defaults are intentionally hostile."""

    partial_fill_probability: float = 0.1
    rejection_probability: float = 0.02
    fill_latency_seconds: float = 2.0
    no_fill_probability: float = 0.15  # limit orders only
    partial_fill_fraction: float = 0.5  # how much of a partial fill lands

    @classmethod
    def deterministic(cls) -> PaperBrokerConfig:
        """No imperfection at all — for the engine-equivalence proof only."""
        return cls(
            partial_fill_probability=0.0,
            rejection_probability=0.0,
            fill_latency_seconds=0.0,
            no_fill_probability=0.0,
        )


class PaperBroker(Broker):
    def __init__(
        self,
        store: StateStore,
        fill_model: FillModel,
        price_source,
        *,
        config: PaperBrokerConfig | None = None,
        rng: random.Random | None = None,
        clock=None,
    ) -> None:
        """``price_source(pair) -> float`` supplies the current market price.

        ``rng`` is injected so simulated imperfection is reproducible; tests
        that assert exact behaviour must seed it.
        """
        self.store = store
        self.fill_model = fill_model
        self.price_source = price_source
        self.config = config or PaperBrokerConfig()
        self.rng = rng or random.Random()
        self._clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
        self._pending_fills: dict[str, pd.Timestamp] = {}

    def now(self) -> pd.Timestamp:
        return pd.Timestamp(self._clock())

    # --- Broker interface ----------------------------------------------------

    def get_balance(self) -> Balance:
        return Balance(cash=self.store.get_cash())

    def get_positions(self) -> list[Position]:
        return self.store.positions()

    def get_open_orders(self) -> list[OrderState]:
        return self.store.open_orders()

    def get_order_status(self, client_order_id: str) -> OrderState:
        state = self.store.get_order(client_order_id)
        if state is None:
            raise UnknownOrderError(f"no order {client_order_id!r} known to this broker")
        return state

    def submit_order(self, approved: ApprovedOrder) -> OrderState:
        """Verify provenance, then accept. Idempotent on client_order_id."""
        verify_approval(approved)  # type is not enough; prove it was minted
        order = approved.order

        existing = self.store.get_order(order.client_order_id)
        if existing is not None:
            # Idempotency: the same logical decision resubmitted after a crash
            # adopts the existing order rather than duplicating it.
            log.info(
                "order %s already known (status=%s) — adopting, not resubmitting",
                order.client_order_id, existing.status,
            )
            return existing

        now = self.now()
        if self.rng.random() < self.config.rejection_probability:
            reason = "simulated venue rejection"
            self.store.insert_order(order, OrderStatus.REJECTED, now, reason)
            log.warning("order %s REJECTED: %s", order.client_order_id, reason)
            return self.get_order_status(order.client_order_id)

        if (
            order.order_type == "limit"
            and self.rng.random() < self.config.no_fill_probability
        ):
            self.store.insert_order(order, OrderStatus.OPEN, now)
            log.info("order %s working, simulated never-fills", order.client_order_id)
            return self.get_order_status(order.client_order_id)

        self.store.insert_order(order, OrderStatus.OPEN, now)
        self._pending_fills[order.client_order_id] = now + pd.Timedelta(
            seconds=self.config.fill_latency_seconds
        )
        return self.get_order_status(order.client_order_id)

    def cancel_order(self, client_order_id: str) -> None:
        state = self.get_order_status(client_order_id)
        if state.status.is_terminal:
            log.info("cancel ignored: %s already %s", client_order_id, state.status)
            return
        self._pending_fills.pop(client_order_id, None)
        self.store.set_order_status(client_order_id, OrderStatus.CANCELLED)

    # --- simulation ----------------------------------------------------------

    def settle(self, now: pd.Timestamp | None = None) -> list[OrderState]:
        """Apply any fills whose latency has elapsed. Call once per cycle.

        Real venues fill asynchronously; this models that rather than
        pretending submission and fill are the same instant.
        """
        now = pd.Timestamp(now) if now is not None else self.now()
        settled: list[OrderState] = []
        for coid, due in list(self._pending_fills.items()):
            if now < due:
                continue
            state = self.store.get_order(coid)
            if state is None or state.status.is_terminal:
                self._pending_fills.pop(coid, None)
                continue
            settled.append(self._apply_fill(state, now))
        return settled

    def _apply_fill(self, state: OrderState, now: pd.Timestamp) -> OrderState:
        order = state.order
        remaining = state.remaining_units
        partial = self.rng.random() < self.config.partial_fill_probability
        units = (
            remaining * self.config.partial_fill_fraction if partial else remaining
        )
        units = max(units, 0.0)

        reference = float(self.price_source(order.pair))
        price = self.fill_model.fill_price(order.side, reference)
        notional = units * price
        liquidity = "taker" if order.order_type == "market" else "maker"
        fee = self.fill_model.fee(notional, liquidity)
        cash_delta = -(notional + fee) if order.side == "buy" else (notional - fee)

        self.store.record_fill(
            order.client_order_id, now, units, price, fee, liquidity,
            cash_delta=cash_delta, pair=order.pair, side=order.side,
        )

        updated = self.get_order_status(order.client_order_id)
        if updated.status is OrderStatus.FILLED:
            self._pending_fills.pop(order.client_order_id, None)
        else:
            # a partial leaves the remainder working, to be filled next cycle
            self._pending_fills[order.client_order_id] = now + pd.Timedelta(
                seconds=self.config.fill_latency_seconds
            )
            log.info(
                "order %s partially filled %.10f of %.10f",
                order.client_order_id, updated.filled_units, order.units,
            )
        return updated

    def mark_to_market(self, now: pd.Timestamp | None = None) -> float:
        """Snapshot equity into the durable store; returns total equity."""
        now = pd.Timestamp(now) if now is not None else self.now()
        cash = self.store.get_cash()
        value = sum(p.value(float(self.price_source(p.pair))) for p in self.get_positions())
        self.store.record_equity(now, cash, value)
        return cash + value
