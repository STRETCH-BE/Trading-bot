"""Canonical OHLCV schema and the pair/timeframe registry.

The full schema contract (column meanings, units, invariants) is documented
in docs/data-schema.md. This module is the single source of truth in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

# --- canonical columns -------------------------------------------------------

TIMESTAMP = "timestamp"  # candle OPEN time, UTC, tz-aware
OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close"
VOLUME = "volume"  # base-asset volume traded during the candle
TRADES = "trades"  # trade count; nullable (Kraken dumps have it, ccxt OHLCV does not)
SOURCE = "source"  # provenance: SOURCE_DUMP or SOURCE_API

COLUMNS = [TIMESTAMP, OPEN, HIGH, LOW, CLOSE, VOLUME, TRADES, SOURCE]

SOURCE_DUMP = "kraken_dump"
SOURCE_API = "ccxt_api"

DTYPES: dict[str, str] = {
    OPEN: "float64",
    HIGH: "float64",
    LOW: "float64",
    CLOSE: "float64",
    VOLUME: "float64",
    TRADES: "Int64",  # nullable integer
    SOURCE: "string",
}


def empty_frame() -> pd.DataFrame:
    """An empty DataFrame with the canonical columns and dtypes."""
    df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in DTYPES.items()})
    df.insert(0, TIMESTAMP, pd.Series(dtype="datetime64[us, UTC]"))
    return df


# --- registry ----------------------------------------------------------------


@dataclass(frozen=True)
class Timeframe:
    """A supported candle interval."""

    name: str  # canonical name used in file paths ("1h", "1d")
    minutes: int  # interval length; also Kraken's dump-file suffix
    ccxt_code: str  # timeframe string passed to ccxt.fetch_ohlcv

    @property
    def step(self) -> timedelta:
        return timedelta(minutes=self.minutes)

    @property
    def step_ms(self) -> int:
        return self.minutes * 60_000

    def dump_filename(self, pair: Pair) -> str:
        """CSV member name inside Kraken's OHLCVT dump, e.g. ``XBTEUR_60.csv``."""
        return f"{pair.kraken_name}_{self.minutes}.csv"


@dataclass(frozen=True)
class Pair:
    """A supported trading pair."""

    kraken_name: str  # name used in Kraken dump files ("XBTEUR")
    ccxt_symbol: str  # unified ccxt symbol ("BTC/EUR")


TIMEFRAMES: dict[str, Timeframe] = {
    "1h": Timeframe(name="1h", minutes=60, ccxt_code="1h"),
    "1d": Timeframe(name="1d", minutes=1440, ccxt_code="1d"),
}

PAIRS: dict[str, Pair] = {
    "XBTEUR": Pair(kraken_name="XBTEUR", ccxt_symbol="BTC/EUR"),
    "ETHEUR": Pair(kraken_name="ETHEUR", ccxt_symbol="ETH/EUR"),
}
