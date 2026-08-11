"""Backtest configuration, loadable from config.yaml."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")


class ConfigNotFoundError(FileNotFoundError):
    """A config path was given but does not exist. Never silently defaulted."""


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

    # FINDING 11: warn when this fraction of intended orders is skipped for
    # failing an exchange minimum. A strategy that can never afford ordermin
    # sits flat forever while the report shows only a counter.
    max_skipped_order_fraction: float = 0.25

    # RISK BUDGET. A strategy target of 1.0 means "fully allocated within my
    # budget", NOT "all of equity": the equity fraction actually held is
    # target * strategy_max_allocation. Defaults to 0.25 to match the shipped
    # max_position_pct, so a full strategy signal is exactly at the risk cap
    # rather than being rejected by it.
    strategy_max_allocation: float = 0.25

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
        if not 0.0 < self.strategy_max_allocation <= 1.0:
            raise ValueError(
                f"strategy_max_allocation must be in (0, 1]; above 1.0 would "
                f"imply leverage, got {self.strategy_max_allocation!r}"
            )
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
        """Load the ``backtest:`` section of config.yaml.

        FINDING 6: this used to return defaults when the path was missing, so a
        typo'd path was indistinguishable from success and the bot could run
        different economics from the backtest with no signal at all. It now
        RAISES. Use ``defaults()`` if you genuinely want unconfigured values.
        """
        path = Path(path)
        if not path.exists():
            raise ConfigNotFoundError(
                f"config file not found: {path.resolve()}. Refusing to fall back "
                f"to defaults — a mistyped path would silently run different fee "
                f"and slippage assumptions from the ones you backtested. Pass a "
                f"real path, or call BacktestConfig.defaults() deliberately."
            )
        data = yaml.safe_load(path.read_text()) or {}
        return cls.from_dict(data.get("backtest", {}) or {})

    @classmethod
    def defaults(cls) -> BacktestConfig:
        """Explicitly unconfigured values. Never reached by accident."""
        return cls()

    def banner(self, path: str | Path = DEFAULT_CONFIG_PATH) -> str:
        """Printed on every start so the numbers in use are visible in the log."""
        fm = self.fill_model
        return "\n".join([
            "=" * 68,
            "RESOLVED CONFIGURATION",
            "=" * 68,
            f"  config path        : {Path(path).resolve()}",
            f"  starting capital   : {self.starting_capital:,.2f}",
            f"  maker / taker fees : {self.maker_fee_bps} / {self.taker_fee_bps} bps"
            f"  (mode: {self.fee_mode})",
            f"  slippage           : {self.slippage_bps} bps",
            f"  effective fee rate : {fm.fee_rate():.6f}",
            f"  effective slippage : {fm.slippage_rate:.6f}",
            f"  min rebalance delta: {self.min_rebalance_delta}",
            f"  ordermin / costmin : per-pair from schema.PAIRS "
            f"(override: {self.min_order_units} / {self.costmin})",
            f"  lot/tick rounding  : {'ON' if fm.apply_rounding else 'OFF'}",
            f"  holdout start      : {self.holdout_start}",
            "=" * 68,
        ])
