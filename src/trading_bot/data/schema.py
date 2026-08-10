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
VWAP = "vwap"  # volume-weighted average price; REST only, NULL for dump rows
VOLUME = "volume"  # base-asset volume traded during the candle
TRADES = "trades"  # trade count
SOURCE = "source"  # provenance: SOURCE_DUMP or SOURCE_REST

COLUMNS = [TIMESTAMP, OPEN, HIGH, LOW, CLOSE, VWAP, VOLUME, TRADES, SOURCE]

SOURCE_DUMP = "dump"
SOURCE_REST = "rest"

DTYPES: dict[str, str] = {
    OPEN: "float64",
    HIGH: "float64",
    LOW: "float64",
    CLOSE: "float64",
    VWAP: "float64",  # NaN for dump-sourced rows
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
    """A supported trading pair.

    ``ordermin`` is the exchange's minimum order size in BASE units. It is a
    per-pair property — Kraken sets a different floor for every asset — so it
    lives here next to the pair rather than as one global setting.
    """

    kraken_name: str  # name used in Kraken dump files ("XBTEUR")
    ccxt_symbol: str  # unified ccxt symbol ("BTC/EUR")
    ordermin: float  # minimum order size, base units
    costmin: float  # minimum order VALUE, quote currency (EUR)


TIMEFRAMES: dict[str, Timeframe] = {
    "1h": Timeframe(name="1h", minutes=60, ccxt_code="1h"),
    "1d": Timeframe(name="1d", minutes=1440, ccxt_code="1d"),
}

# ordermin values: Kraken's published minimum order sizes
# (https://support.kraken.com/articles/205893708 "Minimum order size ...").
#
# NOT verified against the live API in this environment — api.kraken.com is
# unreachable behind the sandbox proxy. Kraken revises these, so treat them as
# a starting point and re-check before trading real money. The authoritative
# runtime source is ccxt's `exchange.load_markets()[symbol]["limits"]["amount"]
# ["min"]`, which mirrors Kraken's AssetPairs `ordermin` field; wire that in
# during Stage 6 and assert it matches these constants at startup.
# costmin: Kraken's cost minimum — the minimum order VALUE in the quote
# currency (1 EUR for EUR-quoted pairs, per Kraken support article
# 12425041458708). Same provisional caveat as ordermin: confirmed via
# Kraken's published support documentation, NOT against the live API from
# this environment. Stage 6 must assert both against ccxt
# load_markets()[symbol]["limits"] ("amount.min" and "cost.min") at startup.
PAIRS: dict[str, Pair] = {
    "XBTEUR": Pair(kraken_name="XBTEUR", ccxt_symbol="BTC/EUR", ordermin=0.0001, costmin=1.0),
    "ETHEUR": Pair(kraken_name="ETHEUR", ccxt_symbol="ETH/EUR", ordermin=0.01, costmin=1.0),
}


class UnknownPairError(KeyError):
    """A pair with no registered ordermin. Never silently defaulted."""


def min_order_units(pair: Pair | str) -> float:
    """Exchange minimum order size for ``pair``, in base units.

    Raises ``UnknownPairError`` for an unregistered pair. Guessing a minimum
    means the backtest fills orders a real exchange would reject, so there is
    deliberately no fallback.
    """
    if isinstance(pair, Pair):
        if pair.kraken_name not in PAIRS:
            raise UnknownPairError(
                f"pair {pair.kraken_name!r} is not in the registry; add it to "
                f"PAIRS with its Kraken ordermin"
            )
        return pair.ordermin
    try:
        return PAIRS[pair].ordermin
    except KeyError:
        raise UnknownPairError(
            f"no registered ordermin for pair {pair!r}; known pairs: "
            f"{sorted(PAIRS)}. Add it to PAIRS rather than assuming a default."
        ) from None


def cost_minimum(pair: Pair | str) -> float:
    """Exchange minimum order value for ``pair``, in quote currency (EUR).

    Raises ``UnknownPairError`` for an unregistered pair — same no-fallback
    policy as ``min_order_units``.
    """
    if isinstance(pair, Pair):
        if pair.kraken_name not in PAIRS:
            raise UnknownPairError(
                f"pair {pair.kraken_name!r} is not in the registry; add it to "
                f"PAIRS with its Kraken costmin"
            )
        return pair.costmin
    try:
        return PAIRS[pair].costmin
    except KeyError:
        raise UnknownPairError(
            f"no registered costmin for pair {pair!r}; known pairs: "
            f"{sorted(PAIRS)}. Add it to PAIRS rather than assuming a default."
        ) from None
