"""Dump-shaped and REST-shaped input for the SAME candle must normalise alike.

Field 5 is volume in a dump row and vwap in a REST row. These tests are built
so that a positional mapping cannot pass them: the vwap and volume values are
deliberately far apart, so swapping them is loud rather than subtle.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from trading_bot.data import schema
from trading_bot.data.normalize import (
    NormalizeError,
    from_ccxt,
    from_kraken_rest,
    read_dump_csv,
)

# One candle, described twice.
TS = 1704067200  # 2024-01-01 00:00:00 UTC
OPEN, HIGH, LOW, CLOSE = 40000.0, 40500.0, 39800.0, 40250.0
VOLUME = 123.456  # field 5 in the dump row
VWAP = 40133.77  # field 5 in the REST row — must NOT land in volume
TRADES = 987

DUMP_ROW = f"{TS},{OPEN},{HIGH},{LOW},{CLOSE},{VOLUME},{TRADES}\n"
REST_ROW = [[TS, OPEN, HIGH, LOW, CLOSE, VWAP, VOLUME, TRADES]]

SHARED = [
    schema.TIMESTAMP,
    schema.OPEN,
    schema.HIGH,
    schema.LOW,
    schema.CLOSE,
    schema.VOLUME,
    schema.TRADES,
]


@pytest.fixture
def dump_row():
    return read_dump_csv(io.StringIO(DUMP_ROW))


@pytest.fixture
def rest_row():
    return from_kraken_rest(REST_ROW)


def test_same_candle_from_both_sources_is_identical_on_shared_fields(dump_row, rest_row):
    """The headline requirement: one candle, two wire formats, one result."""
    pd.testing.assert_frame_equal(dump_row[SHARED], rest_row[SHARED])


def test_rest_vwap_does_not_leak_into_volume(rest_row):
    """The exact bug a positional mapper would introduce."""
    assert rest_row[schema.VOLUME].iloc[0] == VOLUME
    assert rest_row[schema.VOLUME].iloc[0] != VWAP
    assert rest_row[schema.VWAP].iloc[0] == VWAP


def test_dump_vwap_is_null_not_volume(dump_row):
    """A dump has no vwap; it must be null, not silently filled from volume."""
    assert pd.isna(dump_row[schema.VWAP].iloc[0])
    assert dump_row[schema.VOLUME].iloc[0] == VOLUME


def test_source_column_distinguishes_the_two(dump_row, rest_row):
    assert dump_row[schema.SOURCE].iloc[0] == "dump"
    assert rest_row[schema.SOURCE].iloc[0] == "rest"


def test_the_only_differences_are_vwap_and_source(dump_row, rest_row):
    differing = [
        col
        for col in schema.COLUMNS
        if not dump_row[col].equals(rest_row[col])
    ]
    assert set(differing) == {schema.VWAP, schema.SOURCE}


def test_trade_count_survives_both_paths(dump_row, rest_row):
    assert dump_row[schema.TRADES].iloc[0] == TRADES
    assert rest_row[schema.TRADES].iloc[0] == TRADES


def test_ccxt_path_agrees_on_ohlcv_but_has_no_vwap_or_trades():
    """ccxt normalises vwap and count away; both must be null, never guessed."""
    ccxt_row = from_ccxt([[TS * 1000, OPEN, HIGH, LOW, CLOSE, VOLUME]])
    dump = read_dump_csv(io.StringIO(DUMP_ROW))

    ohlcv = [schema.TIMESTAMP, schema.OPEN, schema.HIGH, schema.LOW,
             schema.CLOSE, schema.VOLUME]
    pd.testing.assert_frame_equal(ccxt_row[ohlcv], dump[ohlcv])

    assert pd.isna(ccxt_row[schema.VWAP].iloc[0])
    assert pd.isna(ccxt_row[schema.TRADES].iloc[0])
    assert ccxt_row[schema.SOURCE].iloc[0] == "rest"


# --- format confusion must raise, never silently misparse --------------------


def test_rest_row_fed_to_the_dump_parser_is_rejected():
    eight_col = ",".join(str(v) for v in REST_ROW[0]) + "\n"
    with pytest.raises(NormalizeError, match="REST output, not a dump"):
        read_dump_csv(io.StringIO(eight_col))


def test_dump_row_fed_to_the_rest_parser_is_rejected():
    with pytest.raises(NormalizeError, match="expected 8 fields"):
        from_kraken_rest([[TS, OPEN, HIGH, LOW, CLOSE, VOLUME, TRADES]])


def test_kraken_rest_rows_through_ccxt_parser_keep_vwap_and_count():
    """FINDING 13: 8-column native rows must PRESERVE vwap and count,
    not be truncated to ccxt's six and silently lose both."""
    out = from_ccxt(REST_ROW)
    assert out[schema.VWAP].iloc[0] == VWAP
    assert out[schema.VOLUME].iloc[0] == VOLUME
    assert out[schema.TRADES].iloc[0] == TRADES
    pd.testing.assert_frame_equal(out, from_kraken_rest(REST_ROW))


def test_multi_row_equivalence_holds_across_a_batch():
    rows = []
    csv_lines = []
    for i in range(50):
        ts = TS + i * 3600
        o, h, low_, c = 100.0 + i, 110.0 + i, 90.0 + i, 105.0 + i
        vol, vw, tr = 10.0 + i, 999.0 + i, 7 + i
        rows.append([ts, o, h, low_, c, vw, vol, tr])
        csv_lines.append(f"{ts},{o},{h},{low_},{c},{vol},{tr}\n")

    rest = from_kraken_rest(rows)
    dump = read_dump_csv(io.StringIO("".join(csv_lines)))
    pd.testing.assert_frame_equal(dump[SHARED], rest[SHARED])
    assert (rest[schema.VWAP] > 900).all()  # vwap preserved, not confused
    assert (rest[schema.VOLUME] < 100).all()  # volume preserved
