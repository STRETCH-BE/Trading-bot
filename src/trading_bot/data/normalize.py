"""Normalise raw candle data (Kraken dump CSV rows, ccxt OHLCV rows) into the canonical schema."""

from __future__ import annotations

from collections.abc import Sequence
from typing import IO

import pandas as pd

from trading_bot.data import schema

# Kraken OHLCVT dump columns, in file order. Files have no header row.
# 7 columns: timestamp(s), open, high, low, close, volume, trades.
# Very old exports had 6 columns (no trade count); we accept both.
_DUMP_COLUMNS_7 = ["timestamp_s", "open", "high", "low", "close", "volume", "trades"]
_DUMP_COLUMNS_6 = _DUMP_COLUMNS_7[:-1]


class NormalizeError(ValueError):
    """Raw input does not look like the expected wire format."""


def read_dump_csv(source: str | IO[bytes] | IO[str]) -> pd.DataFrame:
    """Parse one Kraken OHLCVT dump CSV into the canonical schema.

    ``source`` is a path or an open file object (e.g. a zip member).
    """
    raw = pd.read_csv(source, header=None)
    if raw.shape[1] == 7:
        raw.columns = _DUMP_COLUMNS_7
    elif raw.shape[1] == 6:
        raw.columns = _DUMP_COLUMNS_6
        raw["trades"] = pd.NA
    else:
        raise NormalizeError(
            f"expected 6 or 7 columns in Kraken OHLCVT csv, got {raw.shape[1]}"
        )
    ts = pd.to_datetime(raw["timestamp_s"], unit="s", utc=True)
    return _assemble(ts, raw, source_label=schema.SOURCE_DUMP)


def from_ccxt(rows: Sequence[Sequence[float]]) -> pd.DataFrame:
    """Convert ccxt ``fetch_ohlcv`` rows into the canonical schema.

    Each row is ``[timestamp_ms, open, high, low, close, volume]``. ccxt does
    not return a trade count, so ``trades`` is null for API-sourced candles.
    """
    if len(rows) == 0:
        return schema.empty_frame()
    raw = pd.DataFrame(list(rows))
    if raw.shape[1] < 6:
        raise NormalizeError(
            f"expected >=6 fields per ccxt OHLCV row, got {raw.shape[1]}"
        )
    raw = raw.iloc[:, :6]
    raw.columns = ["timestamp_ms", "open", "high", "low", "close", "volume"]
    raw["trades"] = pd.NA
    ts = pd.to_datetime(raw["timestamp_ms"], unit="ms", utc=True)
    return _assemble(ts, raw, source_label=schema.SOURCE_API)


def _assemble(ts: pd.Series, raw: pd.DataFrame, *, source_label: str) -> pd.DataFrame:
    if ts.isna().any():
        raise NormalizeError(
            f"{int(ts.isna().sum())} row(s) with unparseable timestamps — refusing to "
            f"ingest silently corrupt data"
        )
    df = pd.DataFrame(
        {
            schema.TIMESTAMP: ts.astype("datetime64[us, UTC]"),
            schema.OPEN: raw["open"].astype("float64"),
            schema.HIGH: raw["high"].astype("float64"),
            schema.LOW: raw["low"].astype("float64"),
            schema.CLOSE: raw["close"].astype("float64"),
            schema.VOLUME: raw["volume"].astype("float64"),
            schema.TRADES: raw["trades"].astype("Int64"),
            schema.SOURCE: pd.Series([source_label] * len(raw), dtype="string"),
        }
    )
    return finalize(df)


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by timestamp and drop repeated timestamps (keeping the first row).

    Exact duplicates within a single raw file are a wire-format artifact, not
    signal; cross-source conflicts are handled by ``ParquetStore.upsert``.
    """
    df = df.sort_values(schema.TIMESTAMP, kind="stable")
    df = df.drop_duplicates(subset=schema.TIMESTAMP, keep="first")
    return df.reset_index(drop=True)
