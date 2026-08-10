"""Backtest configuration, loadable from config.yaml."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass(frozen=True)
class BacktestConfig:
    """All economics of the simulation. Frozen so a run cannot mutate its own terms."""

    starting_capital: float = 10_000.0

    # Kraken Pro schedule by default (bps = basis points = 1/100 of a percent).
    maker_fee_bps: float = 16.0
    taker_fee_bps: float = 26.0
    # Backtest fills happen at the open of the next candle, which is a
    # marketable order — taker is the honest default. Set "maker" only if you
    # actually model resting limit orders.
    fee_mode: str = "taker"

    slippage_bps: float = 5.0

    # Exchange minimum order size is a PER-PAIR property and normally comes
    # from schema.min_order_units(pair). This is an explicit override for
    # tests and what-if runs only; leaving it None is the correct production
    # setting, and backtest() then requires a pair so nothing is guessed.
    min_order_units: float | None = None

    # Kraken cost minimum (minimum order VALUE in quote currency). Per-pair
    # via schema.cost_minimum(pair); this is a test/what-if override only.
    costmin: float | None = None

    # Rebalance dead-band: a target-position change smaller than this is a
    # no-op. Kills fee-drift chatter on fractional targets; full exits
    # (target 0 while holding) are exempt.
    min_rebalance_delta: float = 0.05

    # Candles on/after this date are HOLDOUT: backtest() raises rather than
    # touch them, unless handed a one-shot HoldoutUnlock minted by an
    # explicit --unlock-holdout flag.
    holdout_start: str = "2026-01-01"

    # Annualisation factor for Sharpe/Sortino/CAGR. None -> inferred from the
    # median candle spacing (crypto trades every day, so 365 for daily).
    periods_per_year: float | None = None
    risk_free_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.fee_mode not in ("maker", "taker"):
            raise ValueError(f"fee_mode must be 'maker' or 'taker', got {self.fee_mode!r}")
        if self.starting_capital <= 0:
            raise ValueError("starting_capital must be positive")
        for name in ("maker_fee_bps", "taker_fee_bps", "slippage_bps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if not 0.0 <= self.min_rebalance_delta < 1.0:
            raise ValueError(
                f"min_rebalance_delta must be in [0, 1), got {self.min_rebalance_delta!r}"
            )
        if self.costmin is not None and self.costmin < 0:
            raise ValueError("costmin must not be negative")
        _ = self.holdout_ts  # fail fast on an unparseable holdout_start

    @property
    def holdout_ts(self) -> pd.Timestamp:
        ts = pd.Timestamp(self.holdout_start)
        return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")

    @property
    def fill_model(self):
        """The one object allowed to turn these bps into rates.

        Config carries the SETTINGS; FillModel owns the ARITHMETIC. Keeping a
        second bps->rate conversion here is exactly the duplication audit
        finding 3 warned about, so `fee_rate` and `slippage_rate` delegate
        rather than recompute.
        """
        from trading_bot.execution.fill_model import FillModel

        return FillModel.from_config(self)

    @property
    def fee_rate(self) -> float:
        """Fractional fee charged on each fill's notional."""
        return self.fill_model.fee_rate()

    @property
    def slippage_rate(self) -> float:
        return self.fill_model.slippage_rate

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BacktestConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown backtest config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> BacktestConfig:
        """Load the ``backtest:`` section of config.yaml (defaults if absent)."""
        path = Path(path)
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text()) or {}
        return cls.from_dict(data.get("backtest", {}) or {})
