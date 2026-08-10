"""Synthetic candle builders for engine and strategy tests.

Deliberately hand-constructed rather than random: every test that uses these
asserts an analytically known answer.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from trading_bot.data import schema

START = pd.Timestamp("2024-01-01", tz="UTC")
ONE_DAY = pd.Timedelta(days=1)


def make_candles(
    rows: Sequence[tuple[float, float, float, float, float]],
    *,
    step: pd.Timedelta = ONE_DAY,
    start: pd.Timestamp = START,
) -> pd.DataFrame:
    """Build a canonical candle frame from (open, high, low, close, volume) rows."""
    ts = [start + i * step for i in range(len(rows))]
    return pd.DataFrame(
        {
            schema.TIMESTAMP: pd.Series(ts, dtype="datetime64[us, UTC]"),
            schema.OPEN: [r[0] for r in rows],
            schema.HIGH: [r[1] for r in rows],
            schema.LOW: [r[2] for r in rows],
            schema.CLOSE: [r[3] for r in rows],
            schema.VOLUME: [r[4] for r in rows],
            schema.TRADES: pd.Series([1] * len(rows), dtype="Int64"),
            schema.SOURCE: pd.Series([schema.SOURCE_DUMP] * len(rows), dtype="string"),
        }
    )


def flat_candles(n: int, price: float = 100.0) -> pd.DataFrame:
    """Constant price — any P&L is therefore pure cost."""
    return make_candles([(price, price, price, price, 10.0)] * n)


def from_closes(closes: Sequence[float], *, spread: float = 0.0) -> pd.DataFrame:
    """Candles whose open equals the previous close (a continuous tape)."""
    rows = []
    prev = closes[0]
    for c in closes:
        o = prev
        rows.append((o, max(o, c) + spread, min(o, c) - spread, c, 10.0))
        prev = c
    return make_candles(rows)
