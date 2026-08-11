"""The broker interface. ``submit_order`` accepts ``ApprovedOrder`` only.

That signature is the Stage 5b guarantee expressed in the type system: there
is no overload, no keyword escape hatch, and no ``Order`` variant. Every
implementation must additionally call ``verify_approval`` before acting,
because the type alone proves shape rather than provenance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from trading_bot.execution.order import Order
from trading_bot.risk.approval import ApprovedOrder


class OrderStatus(StrEnum):
    PENDING = "pending"  # accepted locally, not yet working at the venue
    OPEN = "open"  # working, unfilled
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)


@dataclass(frozen=True)
class Balance:
    cash: float
    currency: str = "EUR"


@dataclass(frozen=True)
class Position:
    pair: str
    units: float
    avg_cost: float
    opened_at: pd.Timestamp

    def value(self, price: float) -> float:
        return self.units * price


@dataclass(frozen=True)
class OrderState:
    """A broker's view of an order: the intent plus what has happened to it."""

    order: Order
    status: OrderStatus
    filled_units: float
    avg_fill_price: float
    fees: float
    submitted_at: pd.Timestamp
    rejection_reason: str = ""

    @property
    def remaining_units(self) -> float:
        return max(0.0, self.order.units - self.filled_units)


class Broker(ABC):
    """Every broker — paper and live — implements exactly this."""

    @abstractmethod
    def get_balance(self) -> Balance: ...

    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    @abstractmethod
    def get_open_orders(self) -> list[OrderState]: ...

    @abstractmethod
    def submit_order(self, approved: ApprovedOrder) -> OrderState:
        """Submit a RISK-APPROVED order. Implementations must call
        ``verify_approval(approved)`` first and must be idempotent on
        ``client_order_id`` — resubmitting the same logical decision after a
        crash must adopt the existing order, never duplicate it."""

    @abstractmethod
    def cancel_order(self, client_order_id: str) -> None: ...

    @abstractmethod
    def get_order_status(self, client_order_id: str) -> OrderState: ...


class UnknownOrderError(KeyError):
    """No order with that client_order_id is known to this broker."""
