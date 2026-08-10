"""Normalise raw candle data into the canonical schema.

Two wire formats, and they are NOT positionally compatible:

    dump (7 cols):  timestamp, open, high, low, close, volume, trades
    REST (8 cols):  timestamp, open, high, low, close, vwap, volume, count
                                                       ^^^^
Field 5 is *volume* in a dump row and *vwap* in a REST row. A positional
mapping therefore writes vwap into the volume column and silently corrupts
every REST-sourced candle — a bug that would never raise, only produce wrong
backtests.

So: the headerless wire rows are given names once, guarded by an exact
column-count check, and every mapping into the canonical schema after that
is BY NAME. ``_from_named`` is the single assembler both paths go through.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import IO

import numpy as np
import pandas as pd

from trading_bot.data import schema

# Wire-format field names, in file/response order. Used ONLY to label the
# columns of a headerless source; nothing downstream indexes by position.
DUMP_FIELDS = ["timestamp_s", "open", "high", "low", "close", "volume", "trades"]
DUMP_FIELDS_LEGACY = ["timestamp_s", "open", "high", "low", "close", "volume"]
REST_FIELDS = ["timestamp_s", "open", "high", "low", "close", "vwap", "volume", "trades"]
CCXT_FIELDS = ["timestamp_ms", "open", "high", "low", "close", "volume"]


class NormalizeError(ValueError):
    """Raw input does not look like the expected wire format."""


def read_dump_csv(source: str | IO[bytes] | IO[str]) -> pd.DataFrame:
    """Parse one Kraken OHLCVT dump CSV into the canonical schema.

    ``source`` is a path or an open file object (e.g. a zip member). The dump
    has no header row and no vwap field, so ``vwap`` is null for these rows.
    """
    raw = pd.read_csv(source, header=None)
    if raw.shape[1] == len(DUMP_FIELDS):
        raw.columns = DUMP_FIELDS
    elif raw.shape[1] == len(DUMP_FIELDS_LEGACY):
        # very old exports omit the trade count
        raw.columns = DUMP_FIELDS_LEGACY
        raw["trades"] = pd.NA
    else:
        raise NormalizeError(
            f"expected {len(DUMP_FIELDS_LEGACY)} or {len(DUMP_FIELDS)} columns in a "
            f"Kraken OHLCVT dump csv, got {raw.shape[1]}. If this file has "
            f"{len(REST_FIELDS)} columns it is REST output, not a dump — use "
            f"from_kraken_rest() so vwap is not read as volume."
        )
    raw["timestamp"] = pd.to_datetime(raw["timestamp_s"], unit="s", utc=True)
    return _from_named(raw, source_label=schema.SOURCE_DUMP)


def from_kraken_rest(rows: Sequence[Sequence]) -> pd.DataFrame:
    """Convert Kraken's native REST OHLC rows into the canonical schema.

    Each row is ``[time, open, high, low, close, vwap, volume, count]``.
    Note field 5 is vwap, NOT volume — see the module docstring.
    """
    if len(rows) == 0:
        return schema.empty_frame()
    raw = pd.DataFrame([list(r) for r in rows])
    if raw.shape[1] != len(REST_FIELDS):
        raise NormalizeError(
            f"expected {len(REST_FIELDS)} fields per Kraken REST OHLC row "
            f"(time,open,high,low,close,vwap,volume,count), got {raw.shape[1]}"
        )
    raw.columns = REST_FIELDS
    raw["timestamp"] = pd.to_datetime(raw["timestamp_s"], unit="s", utc=True)
    return _from_named(raw, source_label=schema.SOURCE_REST)


def from_ccxt(rows: Sequence[Sequence[float]]) -> pd.DataFrame:
    """Convert ccxt ``fetch_ohlcv`` rows into the canonical schema.

    ccxt normalises away both vwap and the trade count, so each row is
    ``[timestamp_ms, open, high, low, close, volume]`` and both of those
    canonical columns end up null. Still REST-sourced.
    """
    if len(rows) == 0:
        return schema.empty_frame()
    raw = pd.DataFrame([list(r) for r in rows])
    if raw.shape[1] < len(CCXT_FIELDS):
        raise NormalizeError(
            f"expected >={len(CCXT_FIELDS)} fields per ccxt OHLCV row, got {raw.shape[1]}"
        )
    if raw.shape[1] > len(CCXT_FIELDS):
        raise NormalizeError(
            f"ccxt OHLCV rows should have {len(CCXT_FIELDS)} fields, got "
            f"{raw.shape[1]}. Refusing to guess which to drop — if this is raw "
            f"Kraken REST output use from_kraken_rest()."
        )
    raw.columns = CCXT_FIELDS
    raw["timestamp"] = pd.to_datetime(raw["timestamp_ms"], unit="ms", utc=True)
    raw["trades"] = pd.NA
    return _from_named(raw, source_label=schema.SOURCE_REST)


def _from_named(raw: pd.DataFrame, *, source_label: str) -> pd.DataFrame:
    """Assemble the canonical frame, pulling every field BY NAME.

    A field absent from this source (vwap in a dump, trades in ccxt) becomes
    null rather than being back-filled from a neighbouring position.
    """
    ts = raw["timestamp"]
    if ts.isna().any():
        raise NormalizeError(
            f"{int(ts.isna().sum())} row(s) with unparseable timestamps — refusing to "
            f"ingest silently corrupt data"
        )

    n = len(raw)
    vwap = raw["vwap"].astype("float64") if "vwap" in raw.columns else pd.Series(
        np.full(n, np.nan), dtype="float64"
    )
    trades = raw["trades"] if "trades" in raw.columns else pd.Series([pd.NA] * n)

    df = pd.DataFrame(
        {
            schema.TIMESTAMP: ts.astype("datetime64[us, UTC]"),
            schema.OPEN: raw["open"].astype("float64"),
            schema.HIGH: raw["high"].astype("float64"),
            schema.LOW: raw["low"].astype("float64"),
            schema.CLOSE: raw["close"].astype("float64"),
            schema.VWAP: vwap.to_numpy(dtype="float64"),
            schema.VOLUME: raw["volume"].astype("float64"),
            schema.TRADES: pd.Series(trades.to_numpy(), dtype="object").astype("Int64"),
            schema.SOURCE: pd.Series([source_label] * n, dtype="string"),
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
