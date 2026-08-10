"""Incremental top-up of the parquet store from the ccxt REST API.

Fetches candles strictly newer than the last stored one and appends them.
Kraken's OHLC endpoint only serves roughly the most recent 720 candles per
timeframe (~30 days of 1h, ~2 years of 1d), so if the store has fallen
further behind than that, the hole cannot be bridged over REST — a newer
dump must be ingested first. That situation raises ``GapError`` unless
``allow_gap=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

import pandas as pd

from trading_bot.data import schema
from trading_bot.data.normalize import from_ccxt
from trading_bot.data.store import ParquetStore

# Kraken's documented OHLC depth limit (candles per request AND total depth).
KRAKEN_OHLC_LIMIT = 720

_MAX_PAGES = 1000  # hard stop against a misbehaving exchange paginator


class Exchange(Protocol):
    """The slice of the ccxt exchange interface the updater needs."""

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
    ) -> list[list[Any]]: ...


class GapError(RuntimeError):
    """The API's history window no longer overlaps the stored data."""


@dataclass
class UpdateResult:
    pair: str
    timeframe: str
    fetched: int
    appended: int
    last_before: pd.Timestamp | None
    last_after: pd.Timestamp | None
    warnings: list[str] = field(default_factory=list)


def make_kraken_exchange() -> Exchange:
    """A rate-limited live ccxt Kraken client (not used by tests)."""
    import ccxt

    return ccxt.kraken({"enableRateLimit": True})


def update(
    store: ParquetStore,
    exchange: Exchange,
    pair: schema.Pair,
    timeframe: schema.Timeframe,
    *,
    now: datetime | None = None,
    allow_gap: bool = False,
) -> UpdateResult:
    """Fetch candles newer than the store's last timestamp and append them."""
    now = now or datetime.now(UTC)
    last_before = store.last_timestamp(pair, timeframe)
    result = UpdateResult(
        pair=pair.kraken_name,
        timeframe=timeframe.name,
        fetched=0,
        appended=0,
        last_before=last_before,
        last_after=last_before,
    )

    if last_before is None:
        result.warnings.append(
            f"{pair.kraken_name}/{timeframe.name}: store is empty — ingest a Kraken "
            f"dump first; fetching only the API window (~{KRAKEN_OHLC_LIMIT} candles)"
        )
        since_ms = None
    else:
        since_ms = int(last_before.timestamp() * 1000) + timeframe.step_ms

    rows = _fetch_all(exchange, pair, timeframe, since_ms)
    result.fetched = len(rows)
    if not rows:
        return result

    df = from_ccxt(rows)
    if since_ms is not None:
        # Kraken ignores a `since` older than its depth window and returns the
        # window anyway; also guards any exchange that rounds `since` down.
        df = df[df[schema.TIMESTAMP] >= pd.Timestamp(since_ms, unit="ms", tz="UTC")]

    # The newest candle is still forming until open-time + step has passed.
    cutoff = pd.Timestamp(now) - timeframe.step
    df = df[df[schema.TIMESTAMP] <= cutoff]
    if df.empty:
        return result

    gap_msgs: list[str] = []
    if last_before is not None:
        expected_next = last_before + timeframe.step
        first_fetched = df[schema.TIMESTAMP].iloc[0]
        if first_fetched > expected_next:
            gap_msgs.append(
                f"{pair.kraken_name}/{timeframe.name}: API history starts at "
                f"{first_fetched} but the store ends at {last_before}; candles in "
                f"between are no longer served over REST (depth ~{KRAKEN_OHLC_LIMIT} "
                f"candles). Ingest a newer Kraken dump to bridge the hole."
            )

    hole_diffs = df[schema.TIMESTAMP].diff()
    holes = int((hole_diffs > pd.Timedelta(timeframe.step)).sum())
    if holes:
        gap_msgs.append(
            f"{pair.kraken_name}/{timeframe.name}: fetched candles contain {holes} "
            f"internal hole(s); the exchange skipped intervals inside its own window."
        )

    if gap_msgs:
        if not allow_gap:
            raise GapError(" ".join(gap_msgs))
        result.warnings.extend(gap_msgs)

    result.appended = store.upsert(pair, timeframe, df)
    result.last_after = store.last_timestamp(pair, timeframe)
    return result


def _fetch_all(
    exchange: Exchange,
    pair: schema.Pair,
    timeframe: schema.Timeframe,
    since_ms: int | None,
) -> list[list[Any]]:
    """Page through fetch_ohlcv until the exchange stops making progress."""
    out: list[list[Any]] = []
    cursor = since_ms
    for _ in range(_MAX_PAGES):
        page = exchange.fetch_ohlcv(
            pair.ccxt_symbol,
            timeframe=timeframe.ccxt_code,
            since=cursor,
            limit=KRAKEN_OHLC_LIMIT,
        )
        if not page:
            break
        out.extend(page)
        if len(page) < KRAKEN_OHLC_LIMIT:
            break
        next_cursor = int(page[-1][0]) + timeframe.step_ms
        if cursor is not None and next_cursor <= cursor:
            break  # no forward progress; avoid spinning
        cursor = next_cursor
    return out
