from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from tests.conftest import FIXTURES, H1, XBTEUR
from tests.helpers import FakeExchange, IgnoresSinceExchange
from trading_bot.data import schema
from trading_bot.data import update as update_mod
from trading_bot.data.normalize import read_dump_csv
from trading_bot.data.update import GapError, update

# The recorded API rows span 2024-01-03 00:00 .. 11:00. With "now" mid-candle
# at 11:30, the 11:00 candle is still open and must not be stored.
NOW = datetime(2024, 1, 3, 11, 30, tzinfo=UTC)


@pytest.fixture
def seeded(store):
    store.write(XBTEUR, H1, read_dump_csv(str(FIXTURES / "XBTEUR_60.csv")))
    return store


def test_update_appends_only_closed_candles(seeded, api_rows):
    exchange = FakeExchange(api_rows)
    result = update(seeded, exchange, XBTEUR, H1, now=NOW)

    assert result.fetched == 12
    assert result.appended == 11  # the 11:00 candle is still open
    assert result.last_after == pd.Timestamp("2024-01-03 10:00:00", tz="UTC")
    assert result.warnings == []

    df = seeded.read(XBTEUR, H1)
    assert len(df) == 48 + 11
    assert df[schema.TIMESTAMP].is_monotonic_increasing
    # fetch started strictly after the stored tail
    symbol, timeframe, since, _ = exchange.calls[0]
    assert symbol == "BTC/EUR"
    assert timeframe == "1h"
    assert since == int(pd.Timestamp("2024-01-03 00:00:00", tz="UTC").timestamp() * 1000)


def test_update_is_idempotent(seeded, api_rows):
    update(seeded, FakeExchange(api_rows), XBTEUR, H1, now=NOW)
    result = update(seeded, FakeExchange(api_rows), XBTEUR, H1, now=NOW)
    assert result.appended == 0
    assert len(seeded.read(XBTEUR, H1)) == 59


def test_update_filters_rows_older_than_since(seeded, api_rows):
    """A Kraken-style exchange ignores `since` and returns its whole window."""
    dump_tail_and_more = [
        [int(pd.Timestamp("2024-01-02 22:00:00", tz="UTC").timestamp() * 1000), 1, 2, 0.5, 1, 3]
    ] + api_rows
    result = update(seeded, IgnoresSinceExchange(dump_tail_and_more), XBTEUR, H1, now=NOW)
    assert result.appended == 11
    # the pre-`since` row must not clobber or duplicate stored history
    df = seeded.read(XBTEUR, H1)
    assert len(df) == 59
    assert not df[schema.TIMESTAMP].duplicated().any()


def test_update_raises_on_unbridgeable_gap(seeded, api_rows):
    exchange = FakeExchange(api_rows, depth=6)  # API window starts at 06:00
    with pytest.raises(GapError, match="Ingest a newer Kraken dump"):
        update(seeded, exchange, XBTEUR, H1, now=NOW)
    # nothing was written
    assert len(seeded.read(XBTEUR, H1)) == 48


def test_update_allow_gap_appends_with_warning(seeded, api_rows):
    exchange = FakeExchange(api_rows, depth=6)
    result = update(seeded, exchange, XBTEUR, H1, now=NOW, allow_gap=True)
    assert result.appended == 5  # 06:00..10:00 (11:00 still open)
    assert any("no longer served over REST" in w for w in result.warnings)


def test_update_empty_store_warns_and_fetches_window(store, api_rows):
    result = update(store, FakeExchange(api_rows), XBTEUR, H1, now=NOW)
    assert any("store is empty" in w for w in result.warnings)
    assert result.appended == 11
    assert store.last_timestamp(XBTEUR, H1) == pd.Timestamp("2024-01-03 10:00:00", tz="UTC")


def test_update_no_new_candles(seeded, api_rows):
    """Right after the stored tail closed, the API has nothing new to add."""
    now = datetime(2024, 1, 3, 0, 30, tzinfo=UTC)  # 00:00 candle still open
    # A correct exchange never returns candles opening after `now`; giving
    # the fake the full future fixture would trip the clock-drift guard.
    available = [r for r in api_rows if r[0] <= now.timestamp() * 1000]
    result = update(seeded, FakeExchange(available), XBTEUR, H1, now=now)
    assert result.appended == 0
    assert result.last_after == pd.Timestamp("2024-01-02 23:00:00", tz="UTC")


def test_update_rejects_internal_hole_in_fetched_window(seeded, api_rows):
    holey = api_rows[:4] + api_rows[6:]  # drop 04:00 and 05:00 inside the window
    with pytest.raises(GapError, match="internal hole"):
        update(seeded, FakeExchange(holey), XBTEUR, H1, now=NOW)
    assert len(seeded.read(XBTEUR, H1)) == 48

    result = update(seeded, FakeExchange(holey), XBTEUR, H1, now=NOW, allow_gap=True)
    assert result.appended == 9  # 11 closed candles minus the 2 missing ones
    assert any("internal hole" in w for w in result.warnings)


def test_update_paginates(seeded, api_rows, monkeypatch):
    monkeypatch.setattr(update_mod, "KRAKEN_OHLC_LIMIT", 4)
    exchange = FakeExchange(api_rows)
    result = update(seeded, exchange, XBTEUR, H1, now=NOW)
    assert result.fetched == 12
    assert result.appended == 11
    assert len(exchange.calls) == 4  # 3 full pages of 4, then an empty page
    df = seeded.read(XBTEUR, H1)
    assert not df[schema.TIMESTAMP].duplicated().any()
