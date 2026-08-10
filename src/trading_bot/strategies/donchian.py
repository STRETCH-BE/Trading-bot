"""Donchian channel breakout, long-only, with an ATR trailing stop.

Pure function: no I/O, no network, no logging, no module state. Same candles
in, same signals out — which is what makes the shuffle-the-future test in
``tests/test_donchian.py`` meaningful.

Causality rule enforced throughout: the signal for candle N is computed only
from candles <= N. Every lookback window is built with ``.shift(1)`` so it
excludes the current candle, and the position state machine is a forward loop
that never reads ahead.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from trading_bot.data import schema

DEFAULT_CONFIG_PATH = Path("config.yaml")

LONG = 1.0
FLAT = 0.0


@dataclass(frozen=True)
class DonchianParams:
    """Strategy parameters. Frozen: a run cannot retune itself mid-flight."""

    entry_lookback: int = 20  # N: break above the highest high of the prior N
    exit_lookback: int = 10  # M: break below the lowest low of the prior M
    atr_period: int = 14
    atr_multiple: float = 2.5

    def __post_init__(self) -> None:
        for name in ("entry_lookback", "exit_lookback", "atr_period"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.atr_multiple <= 0:
            raise ValueError(f"atr_multiple must be positive, got {self.atr_multiple!r}")

    @property
    def warmup(self) -> int:
        """Candles that cannot produce a signal because a window is incomplete."""
        return max(self.entry_lookback, self.exit_lookback, self.atr_period)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DonchianParams:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown donchian params: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> DonchianParams:
        """Load ``strategies.donchian`` from config.yaml (defaults if absent)."""
        path = Path(path)
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text()) or {}
        section = (data.get("strategies") or {}).get("donchian") or {}
        return cls.from_dict(section)


def donchian_breakout(
    candles: pd.DataFrame, params: DonchianParams | None = None
) -> pd.Series:
    """Target position per candle: 1.0 fully long, 0.0 flat. Never anything else.

    Entry  : close > highest high of the previous ``entry_lookback`` candles
    Exit   : close < highest close since entry - ``atr_multiple`` * ATR
             or close < lowest low of the previous ``exit_lookback`` candles
    """
    params = params or DonchianParams()
    _validate(candles)

    df = candles.reset_index(drop=True)
    high = df[schema.HIGH].astype(float)
    low = df[schema.LOW].astype(float)
    close = df[schema.CLOSE].astype(float)

    # `.shift(1)` is what excludes the current candle from its own lookback.
    entry_band = high.rolling(params.entry_lookback).max().shift(1).to_numpy()
    exit_band = low.rolling(params.exit_lookback).min().shift(1).to_numpy()
    atr = _wilder_atr(high, low, close, params.atr_period).to_numpy()
    close_arr = close.to_numpy()

    n = len(df)
    signals = np.zeros(n, dtype=float)
    warmup = params.warmup

    in_position = False
    highest_close = 0.0

    for i in range(n):
        if i < warmup:
            continue  # stays FLAT; never NaN, never a spurious entry

        px = close_arr[i]

        if not in_position:
            band = entry_band[i]
            if np.isfinite(band) and px > band:
                in_position = True
                highest_close = px  # trailing stop anchors at the entry close
                signals[i] = LONG
            continue

        # already long: update the trail, then test both exits
        highest_close = max(highest_close, px)
        stop = highest_close - params.atr_multiple * atr[i]
        broke_trail = np.isfinite(stop) and px < stop
        broke_channel = np.isfinite(exit_band[i]) and px < exit_band[i]

        if broke_trail or broke_channel:
            in_position = False
            highest_close = 0.0
            signals[i] = FLAT
        else:
            signals[i] = LONG

    return pd.Series(signals, index=candles.index, name="donchian_breakout")


def _wilder_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    """Average True Range with Wilder's smoothing (the conventional ATR).

    True range uses the PREVIOUS close, so it never reads a future bar.
    """
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # first bar has no previous close; its true range is simply high - low
    true_range.iloc[0] = high.iloc[0] - low.iloc[0]
    return true_range.ewm(alpha=1.0 / period, adjust=False).mean()


def _validate(candles: pd.DataFrame) -> None:
    required = [schema.TIMESTAMP, schema.HIGH, schema.LOW, schema.CLOSE]
    missing = [c for c in required if c not in candles.columns]
    if missing:
        raise ValueError(f"candles missing required columns: {missing}")
    if not candles[schema.TIMESTAMP].is_monotonic_increasing:
        raise ValueError("candle timestamps must be monotonically increasing")
