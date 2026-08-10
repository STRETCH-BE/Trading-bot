"""The order abstraction — the submission boundary the audit found missing.

Before this, ``backtest()`` computed target positions and filled them inside
one loop, so there was no object for a risk gate to sit in front of. An
``Order`` is that object: a concrete intention to trade, produced by
translation and consumed by execution, with a gate between them.

``client_order_id`` is DETERMINISTIC by construction. The same logical
decision — same strategy, pair, signal timestamp and intent — always yields
the same id, in this process or any future one. That is what makes
idempotent submission possible: after a crash mid-submit, the recovering
process can ask the exchange "did order <id> land?" rather than guessing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import pandas as pd

Side = Literal["buy", "sell"]
OrderType = Literal["limit", "market"]

ID_PREFIX = "tb"
_ID_HEX_CHARS = 24


def make_client_order_id(
    strategy: str, pair: str, signal_timestamp: pd.Timestamp, intent: str
) -> str:
    """A stable id for one logical trading decision.

    Uses SHA-256, NOT Python's ``hash()``: the builtin is salted per process
    (PYTHONHASHSEED), so ids derived from it would silently change across
    restarts and every idempotency check downstream would break exactly when
    it mattered — after a crash.

    ``signal_timestamp`` is normalised to UTC ISO-8601 so an equivalent
    timestamp in another tz representation maps to the same id.
    """
    ts = pd.Timestamp(signal_timestamp)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    payload = "|".join([strategy, pair, ts.isoformat(), intent])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_ID_HEX_CHARS]
    return f"{ID_PREFIX}-{digest}"


@dataclass(frozen=True)
class Order:
    """An intention to trade. Immutable; nothing may mutate an order in flight."""

    client_order_id: str
    pair: str
    side: Side
    units: float
    order_type: OrderType
    limit_price: float | None
    reason: str  # why the strategy wants this, for the decision log
    timestamp: pd.Timestamp

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {self.side!r}")
        if self.order_type not in ("limit", "market"):
            raise ValueError(f"order_type must be 'limit' or 'market', got {self.order_type!r}")
        if self.units <= 0:
            raise ValueError(
                f"units must be positive, got {self.units!r}. Direction is carried "
                f"by `side`; a negative size would be an implicit short."
            )
        if self.order_type == "limit" and self.limit_price is None:
            raise ValueError("a limit order requires a limit_price")

    def notional(self, price: float) -> float:
        return self.units * price


@dataclass(frozen=True)
class PortfolioState:
    """Cash and units held. Equity is derived, never stored, so it cannot drift."""

    cash: float
    units: float

    def equity(self, price: float) -> float:
        return self.cash + self.units * price

    def position_fraction(self, price: float, *, eps: float = 1e-12) -> float:
        eq = self.equity(price)
        return (self.units * price) / eq if eq > eps else 0.0
