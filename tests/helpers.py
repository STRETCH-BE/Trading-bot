"""Test doubles. FakeExchange replays recorded rows instead of calling Kraken."""

from __future__ import annotations

from typing import Any


class FakeExchange:
    """Replays recorded ``fetch_ohlcv`` rows with Kraken-like semantics.

    ``depth`` emulates Kraken's OHLC endpoint only serving the most recent N
    candles: a ``since`` older than the window is silently ignored and the
    window is returned anyway, exactly like the real endpoint.
    """

    def __init__(self, rows: list[list[Any]], *, depth: int | None = None) -> None:
        self.rows = sorted((list(r) for r in rows), key=lambda r: r[0])
        self.depth = depth
        self.calls: list[tuple[str, str, int | None, int | None]] = []

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
    ) -> list[list[Any]]:
        self.calls.append((symbol, timeframe, since, limit))
        rows = self.rows
        if self.depth is not None:
            rows = rows[-self.depth :]
        if since is not None:
            rows = [r for r in rows if r[0] >= since]
        if limit is not None:
            rows = rows[:limit]
        return [list(r) for r in rows]


class IgnoresSinceExchange(FakeExchange):
    """Returns its whole window regardless of ``since`` (Kraken-style)."""

    def fetch_ohlcv(self, symbol, timeframe="1m", since=None, limit=None):
        self.calls.append((symbol, timeframe, since, limit))
        rows = self.rows
        if self.depth is not None:
            rows = rows[-self.depth :]
        if limit is not None:
            rows = rows[:limit]
        return [list(r) for r in rows]
