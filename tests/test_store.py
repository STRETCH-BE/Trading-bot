from __future__ import annotations

import io

import pandas as pd

from tests.conftest import D1, FIXTURES, H1, XBTEUR
from trading_bot.data import schema
from trading_bot.data.normalize import from_ccxt, read_dump_csv


def _dump_df():
    return read_dump_csv(str(FIXTURES / "XBTEUR_60.csv"))


def test_roundtrip_preserves_frame(store):
    df = _dump_df()
    store.write(XBTEUR, H1, df)
    out = store.read(XBTEUR, H1)
    pd.testing.assert_frame_equal(out, df)


def test_read_missing_returns_empty_canonical(store):
    out = store.read(XBTEUR, H1)
    assert out.empty
    assert list(out.columns) == schema.COLUMNS


def test_upsert_api_row_never_replaces_dump_row(store, api_rows):
    df = _dump_df()
    store.write(XBTEUR, H1, df)

    conflict = from_ccxt(
        [[int(df[schema.TIMESTAMP].iloc[-1].timestamp() * 1000), 1.0, 2.0, 0.5, 1.5, 9.9]]
    )
    added = store.upsert(XBTEUR, H1, conflict)
    assert added == 0

    out = store.read(XBTEUR, H1)
    last = out.iloc[-1]
    assert last[schema.CLOSE] == df[schema.CLOSE].iloc[-1]  # not 1.5
    assert last[schema.SOURCE] == schema.SOURCE_DUMP


def test_upsert_dump_row_replaces_api_row(store, api_rows):
    """A newer quarterly dump enriches candles the updater fetched earlier."""
    store.write(XBTEUR, H1, from_ccxt(api_rows))
    first_api_ts = int(api_rows[0][0]) // 1000

    dump_row = read_dump_csv(
        io.StringIO(f"{first_api_ts},41000.0,41100.0,40900.0,41050.0,7.5,321\n")
    )
    added = store.upsert(XBTEUR, H1, dump_row)
    assert added == 0  # replacement, not growth

    out = store.read(XBTEUR, H1)
    row = out.iloc[0]
    assert row[schema.SOURCE] == schema.SOURCE_DUMP
    assert row[schema.CLOSE] == 41050.0
    assert row[schema.TRADES] == 321
    # re-running the same API upsert stays idempotent and does not undo it
    assert store.upsert(XBTEUR, H1, from_ccxt(api_rows)) == 0
    assert store.read(XBTEUR, H1).iloc[0][schema.SOURCE] == schema.SOURCE_DUMP


def test_write_failure_preserves_previous_data(store, monkeypatch):
    df = _dump_df()
    store.write(XBTEUR, H1, df)

    import pytest

    from trading_bot.data import store as store_mod

    def boom(table, path):
        raise OSError("disk full")

    monkeypatch.setattr(store_mod.pq, "write_table", boom)
    with pytest.raises(OSError, match="disk full"):
        store.write(XBTEUR, H1, df.head(3))
    monkeypatch.undo()

    out = store.read(XBTEUR, H1)
    pd.testing.assert_frame_equal(out, df)  # old contents intact
    leftovers = list(store.path(XBTEUR, H1).parent.glob("*.tmp"))
    assert leftovers == []


def test_upsert_appends_new_rows(store, api_rows):
    store.write(XBTEUR, H1, _dump_df())
    added = store.upsert(XBTEUR, H1, from_ccxt(api_rows))
    assert added == 12
    out = store.read(XBTEUR, H1)
    assert len(out) == 60
    assert out[schema.TIMESTAMP].is_monotonic_increasing
    assert out[schema.SOURCE].iloc[-1] == schema.SOURCE_API


def test_last_timestamp(store):
    assert store.last_timestamp(XBTEUR, H1) is None
    store.write(XBTEUR, H1, _dump_df())
    assert store.last_timestamp(XBTEUR, H1) == pd.Timestamp("2024-01-02 23:00:00", tz="UTC")


def test_datasets_lists_only_existing(store):
    assert list(store.datasets()) == []
    store.write(XBTEUR, D1, read_dump_csv(str(FIXTURES / "XBTEUR_1440.csv")))
    assert list(store.datasets()) == [(XBTEUR, D1)]


def test_write_rejects_missing_columns(store):
    import pytest

    with pytest.raises(ValueError, match="missing canonical columns"):
        store.write(XBTEUR, H1, pd.DataFrame({"timestamp": []}))
