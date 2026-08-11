"""Risk limits and the account state they are evaluated against."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from trading_bot.backtest.config import ConfigNotFoundError

DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass(frozen=True)
class RiskLimits:
    """All limits, from config.yaml. Frozen: a run cannot relax its own limits."""

    max_position_pct: float = 25.0  # max % of equity in a single position
    max_total_exposure_pct: float = 75.0  # max % of equity deployed at once
    max_concurrent_positions: int = 2
    daily_loss_limit_pct: float = 5.0  # breach -> flatten, halt, manual restart
    max_drawdown_pct: float = 20.0  # from peak equity; same response
    max_order_size_pct: float = 30.0  # fat-finger / runaway-loop guard
    max_orders_per_hour: int = 10  # circuit breaker
    stale_data_max_seconds: float = 300.0

    def __post_init__(self) -> None:
        for name in (
            "max_position_pct", "max_total_exposure_pct", "daily_loss_limit_pct",
            "max_drawdown_pct", "max_order_size_pct", "stale_data_max_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        for name in ("max_concurrent_positions", "max_orders_per_hour"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RiskLimits:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown risk limits: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> RiskLimits:
        path = Path(path)
        if not path.exists():
            # FINDING 6: a mistyped path must never masquerade as success.
            raise ConfigNotFoundError(
                f"config file not found: {path.resolve()}. Refusing to fall "
                f"back to defaults — that would silently run different "
                f"parameters from the ones you validated."
            )
        data = yaml.safe_load(path.read_text()) or {}
        return cls.from_dict(data.get("risk", {}) or {})


@dataclass
class AccountState:
    """Everything the gate needs to judge an order. Supplied by the caller.

    ``peak_equity`` and ``day_start_equity`` are carried by the caller across
    cycles (and across restarts, via SQLite in Stage 6a) because a drawdown
    limit measured from a peak the process forgot on restart is no limit.
    """

    equity: float
    cash: float
    positions: dict[str, float] = field(default_factory=dict)  # pair -> units
    prices: dict[str, float] = field(default_factory=dict)  # pair -> latest price
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    latest_data_time: pd.Timestamp | None = None
    recent_order_times: list[pd.Timestamp] = field(default_factory=list)

    def position_value(self, pair: str) -> float:
        return self.positions.get(pair, 0.0) * self.prices.get(pair, 0.0)

    def total_exposure(self) -> float:
        return sum(
            units * self.prices.get(pair, 0.0) for pair, units in self.positions.items()
        )

    def open_position_count(self, *, eps: float = 1e-12) -> int:
        return sum(1 for units in self.positions.values() if abs(units) > eps)

    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity * 100.0)

    def daily_loss_pct(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return max(0.0, (self.day_start_equity - self.equity) / self.day_start_equity * 100.0)
