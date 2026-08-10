"""Volatility-targeted trend, long-only. Strategy #2 (preregistration amendment 2).

Direction from a trend filter, SIZE from inverse realized volatility:

    target[i] = 0.0                                   if close[i] <= SMA(trend_lookback)[i]
    target[i] = min(target_vol / realized_vol[i], 1)  otherwise

Rationale on record before any result: strategy #1 sized binary, so it paid a
full round trip at every whipsaw and carried full exposure through the
high-volatility stretches where its drawdowns were made.

Pure function: no I/O, no state, no logging. Causality — the signal for
candle N uses only candles <= N. Both the SMA and the volatility estimate are
trailing windows ending at N; neither is shifted forward, and no reversal or
centred window appears anywhere. Enforced by the registry causality gate.
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

TRADING_DAYS_PER_YEAR = 365.0  # crypto trades every day
_MIN_VOL = 1e-8  # guard against a zero-variance window


@dataclass(frozen=True)
class VolTrendParams:
    """Frozen: a run cannot retune itself mid-flight."""

    trend_lookback: int = 100  # SMA length defining the trend regime
    vol_lookback: int = 20  # realized-volatility estimation window
    target_vol: float = 0.40  # annualised volatility target for the position
    max_position: float = 1.0  # long-only, unlevered: never above 1.0

    def __post_init__(self) -> None:
        for name in ("trend_lookback", "vol_lookback"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2, got {value!r}")
        if self.target_vol <= 0:
            raise ValueError(f"target_vol must be positive, got {self.target_vol!r}")
        if not 0 < self.max_position <= 1.0:
            raise ValueError(
                f"max_position must be in (0, 1]; above 1.0 implies leverage, "
                f"got {self.max_position!r}"
            )

    @property
    def warmup(self) -> int:
        """Candles that cannot produce a signal because a window is incomplete.

        vol_lookback + 1 because the volatility window is over RETURNS, which
        cost one candle to difference.
        """
        return max(self.trend_lookback, self.vol_lookback + 1)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VolTrendParams:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown voltrend params: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> VolTrendParams:
        path = Path(path)
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text()) or {}
        section = (data.get("strategies") or {}).get("voltrend") or {}
        return cls.from_dict(section)


def voltrend(candles: pd.DataFrame, params: VolTrendParams | None = None) -> pd.Series:
    """Target position per candle: 0.0 when flat, else the vol-scaled size."""
    params = params or VolTrendParams()
    _validate(candles)

    df = candles.reset_index(drop=True)
    close = df[schema.CLOSE].astype(float)

    # Trailing SMA ending at the current candle: causal, uses closes <= N.
    sma = close.rolling(params.trend_lookback).mean()
    trend_on = (close > sma).to_numpy()

    # Realized volatility of log returns, annualised. rolling() ends at the
    # current candle, and the return at N uses closes N-1 and N — both <= N.
    log_ret = np.log(close / close.shift(1))
    realized = log_ret.rolling(params.vol_lookback).std(ddof=1) * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    realized_arr = realized.to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = params.target_vol / np.maximum(realized_arr, _MIN_VOL)
    size = np.clip(scaled, 0.0, params.max_position)

    signals = np.where(trend_on, size, 0.0)
    signals[: params.warmup] = 0.0  # never NaN, never a spurious entry
    signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

    return pd.Series(signals, index=candles.index, name="voltrend")


@dataclass(frozen=True)
class VolTrendSpan:
    """``params`` in force for candles at index >= ``start``. Data, not state."""

    start: int
    params: VolTrendParams


def voltrend_schedule(
    candles: pd.DataFrame,
    spans: list[VolTrendSpan],
    *,
    trade_start: int = 0,
) -> pd.Series:
    """Walk-forward variant: parameters swap at fixed candle indexes.

    This strategy is stateless — the signal at N is a function of trailing
    data only, with no position flag or trailing anchor to carry — so a
    parameter swap changes the sizing rule and nothing else. Position
    continuity is preserved by the engine, which holds units across the
    boundary because the signal never forces a flat.
    """
    _validate(candles)
    if not spans:
        raise ValueError("empty parameter schedule")
    spans = sorted(spans, key=lambda s: s.start)
    if spans[0].start > trade_start:
        raise ValueError(
            f"first span starts at {spans[0].start} but trading starts at "
            f"{trade_start}; every traded candle needs parameters in force"
        )

    n = len(candles)
    # Per unique params, the full-history signal; then pick per candle by span.
    # Cached: walk-forward reselects the same parameters often.
    computed: dict[VolTrendParams, np.ndarray] = {}
    for span in spans:
        if span.params not in computed:
            computed[span.params] = voltrend(candles, span.params).to_numpy()

    out = np.zeros(n, dtype=float)
    for k, span in enumerate(spans):
        lo = max(span.start, trade_start)
        hi = spans[k + 1].start if k + 1 < len(spans) else n
        if hi > lo:
            out[lo:hi] = computed[span.params][lo:hi]

    return pd.Series(out, index=candles.index, name="voltrend")


def _validate(candles: pd.DataFrame) -> None:
    required = [schema.TIMESTAMP, schema.CLOSE]
    missing = [c for c in required if c not in candles.columns]
    if missing:
        raise ValueError(f"candles missing required columns: {missing}")
    if not candles[schema.TIMESTAMP].is_monotonic_increasing:
        raise ValueError("candle timestamps must be monotonically increasing")
