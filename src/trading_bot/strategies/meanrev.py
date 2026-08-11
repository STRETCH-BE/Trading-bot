"""Mean reversion, long-or-flat. Strategy #3 (preregistration Amendment 6).

The "buy dips, sell recovery" hypothesis. Mechanism on record before any
result: liquidation cascades and panic selling push price transiently below
short-horizon fair value; a buyer of K-sigma dips who exits on reversion to
the mean harvests the rebound.

    ENTRY (flat) : close(t) <  MA_N(t) - K * sigma_N(t)   -> target 1.0
    EXIT  (long) : close(t) >= MA_N(t) - E * sigma_N(t)   -> target 0.0
    otherwise    : hold the previous target (hysteresis)

K and E are both in STANDARD DEVIATIONS (fixed by Amendment 6). MA and sigma
are computed over the trailing N closes INCLUDING bar t, sigma with ddof=1.
E < K always in the preregistered grid, so the exit band sits above the
entry band and the two states cannot chatter within one bar's information.

Pure function: no I/O, no state between calls, no logging. Causality — the
signal at candle N uses closes <= N only; both rolling windows end at N.
Enforced by the registry causality gate (prefix-vs-full).

The strategy is a two-state machine (position flag carries), so the
walk-forward variant is a schedule over ONE state machine, exactly like
Donchian's: at a span boundary only the parameters swap; the position flag
carries straight across. No forced close, no forced re-entry.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from trading_bot.backtest.config import ConfigNotFoundError
from trading_bot.data import schema

DEFAULT_CONFIG_PATH = Path("config.yaml")

LONG = 1.0
FLAT = 0.0


@dataclass(frozen=True)
class MeanRevParams:
    """Frozen: a run cannot retune itself mid-flight."""

    lookback: int = 20  # N: window for both the MA and sigma
    entry_k: float = 1.5  # K: entry when close < MA - K*sigma
    exit_e: float = 0.0  # E: exit when close >= MA - E*sigma

    def __post_init__(self) -> None:
        if not isinstance(self.lookback, int) or self.lookback < 2:
            raise ValueError(f"lookback must be an integer >= 2, got {self.lookback!r}")
        if self.entry_k <= 0:
            raise ValueError(f"entry_k must be positive, got {self.entry_k!r}")
        if self.exit_e < 0:
            raise ValueError(f"exit_e must be >= 0, got {self.exit_e!r}")
        if self.exit_e >= self.entry_k:
            raise ValueError(
                f"exit_e ({self.exit_e!r}) must be below entry_k ({self.entry_k!r}) "
                f"or the exit band sits at/below the entry band and the state "
                f"machine enters and exits on the same information"
            )

    @property
    def warmup(self) -> int:
        """Candles that cannot produce a signal: the rolling window needs N
        closes, and sigma with ddof=1 needs at least 2 — covered by N >= 2."""
        return self.lookback

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MeanRevParams:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown meanrev params: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> MeanRevParams:
        path = Path(path)
        if not path.exists():
            raise ConfigNotFoundError(
                f"config file not found: {path.resolve()}. Refusing to fall "
                f"back to defaults — that would silently run different "
                f"parameters from the ones you validated."
            )
        data = yaml.safe_load(path.read_text()) or {}
        section = (data.get("strategies") or {}).get("meanrev") or {}
        return cls.from_dict(section)


def meanrev(candles: pd.DataFrame, params: MeanRevParams | None = None) -> pd.Series:
    """Target position per candle: 1.0 long, 0.0 flat. Never anything else."""
    params = params or MeanRevParams()
    return meanrev_schedule(candles, [MeanRevSpan(0, params)])


@dataclass(frozen=True)
class MeanRevSpan:
    """``params`` in force for candles at index >= ``start``. Data, not state."""

    start: int
    params: MeanRevParams


def meanrev_schedule(
    candles: pd.DataFrame,
    spans: list[MeanRevSpan],
    *,
    trade_start: int = 0,
) -> pd.Series:
    """Mean reversion with parameters that swap at fixed candle indexes.

    THE walk-forward continuity mechanism, same shape as Donchian's: one
    state machine over the whole series; at a span boundary only the
    parameters change — the position flag carries straight across.

    Candles before ``trade_start`` emit 0.0 and the state machine does not
    run there. Still a pure function of (candles, spans, trade_start).
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

    df = candles.reset_index(drop=True)
    close = df[schema.CLOSE].astype(float)
    close_arr = close.to_numpy()
    n = len(df)

    # Per unique params: MA and sigma over the trailing N closes INCLUDING
    # the current candle (Amendment 6), sigma with ddof=1. Cached — the
    # walk-forward reselects the same params often.
    bands: dict[MeanRevParams, tuple[np.ndarray, np.ndarray]] = {}
    for span in spans:
        p = span.params
        if p not in bands:
            ma = close.rolling(p.lookback).mean().to_numpy()
            sigma = close.rolling(p.lookback).std(ddof=1).to_numpy()
            bands[p] = (ma, sigma)

    signals = np.zeros(n, dtype=float)
    in_position = False
    span_idx = 0
    while span_idx + 1 < len(spans) and spans[span_idx + 1].start <= trade_start:
        span_idx += 1

    for i in range(max(trade_start, 0), n):
        # advance the schedule; ONLY the parameters change here, never
        # in_position
        while span_idx + 1 < len(spans) and spans[span_idx + 1].start <= i:
            span_idx += 1
        params = spans[span_idx].params
        ma, sigma = bands[params]

        if i < params.warmup:
            continue  # stays FLAT; never NaN, never a spurious entry

        m, s = ma[i], sigma[i]
        if not (np.isfinite(m) and np.isfinite(s)):
            # sigma can be NaN on a constant window even past warmup only if
            # the window is degenerate; treat as no-signal, hold state
            signals[i] = LONG if in_position else FLAT
            continue

        px = close_arr[i]
        if not in_position:
            if px < m - params.entry_k * s:
                in_position = True
                signals[i] = LONG
            continue

        if px >= m - params.exit_e * s:
            in_position = False
            signals[i] = FLAT
        else:
            signals[i] = LONG

    return pd.Series(signals, index=candles.index, name="meanrev")


def _validate(candles: pd.DataFrame) -> None:
    required = [schema.TIMESTAMP, schema.CLOSE]
    missing = [c for c in required if c not in candles.columns]
    if missing:
        raise ValueError(f"candles missing required columns: {missing}")
    if not candles[schema.TIMESTAMP].is_monotonic_increasing:
        raise ValueError("candle timestamps must be monotonically increasing")
