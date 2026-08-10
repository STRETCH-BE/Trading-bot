from __future__ import annotations

import io

import pandas as pd
import pytest

from tests.conftest import FIXTURES
from trading_bot.data import schema
from trading_bot.data.normalize import NormalizeError, finalize, from_ccxt, read_dump_csv


def test_read_dump_csv_canonical_shape():
    df = read_dump_csv(str(FIXTURES / "XBTEUR_60.csv"))

    assert list(df.columns) == schema.COLUMNS
    assert len(df) == 48
    assert str(df[schema.TIMESTAMP].dtype) == "datetime64[us, UTC]"
    assert df[schema.TIMESTAMP].is_monotonic_increasing
    assert df[schema.TIMESTAMP].iloc[0] == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    assert (df[schema.SOURCE] == schema.SOURCE_DUMP).all()
    assert df[schema.TRADES].notna().all()
    assert str(df[schema.TRADES].dtype) == "Int64"
    for col in (schema.OPEN, schema.HIGH, schema.LOW, schema.CLOSE, schema.VOLUME):
        assert str(df[col].dtype) == "float64"
    # OHLC coherence of the fixture itself
    assert (df[schema.HIGH] >= df[[schema.OPEN, schema.CLOSE]].max(axis=1)).all()
    assert (df[schema.LOW] <= df[[schema.OPEN, schema.CLOSE]].min(axis=1)).all()


def test_read_dump_csv_six_column_variant():
    df = read_dump_csv(io.StringIO("1704067200,100,110,90,105,3.5\n"))
    assert len(df) == 1
    assert df[schema.TRADES].isna().all()


def test_read_dump_csv_rejects_unknown_width():
    with pytest.raises(NormalizeError):
        read_dump_csv(io.StringIO("1,2,3\n"))


def test_read_dump_csv_rejects_unparseable_timestamp():
    # a truncated dump line with an empty timestamp must not become NaT in the store
    with pytest.raises(NormalizeError, match="unparseable timestamps"):
        read_dump_csv(io.StringIO("1704067200,1,2,0.5,1.5,3,7\n,1,2,0.5,1.5,3,7\n"))


def test_from_ccxt(api_rows):
    df = from_ccxt(api_rows)
    assert list(df.columns) == schema.COLUMNS
    assert len(df) == 12
    assert df[schema.TIMESTAMP].iloc[0] == pd.Timestamp("2024-01-03 00:00:00", tz="UTC")
    assert (df[schema.SOURCE] == schema.SOURCE_REST).all()
    assert df[schema.TRADES].isna().all()


def test_from_ccxt_empty():
    df = from_ccxt([])
    assert df.empty
    assert list(df.columns) == schema.COLUMNS


def test_from_ccxt_rejects_short_rows():
    with pytest.raises(NormalizeError):
        from_ccxt([[1704240000000, 1.0, 2.0]])


def test_finalize_sorts_and_dedupes():
    df = read_dump_csv(str(FIXTURES / "XBTEUR_60.csv"))
    shuffled = pd.concat([df.iloc[::-1], df.head(3)], ignore_index=True)
    out = finalize(shuffled)
    assert len(out) == len(df)
    assert out[schema.TIMESTAMP].is_monotonic_increasing
    pd.testing.assert_frame_equal(out, df)
