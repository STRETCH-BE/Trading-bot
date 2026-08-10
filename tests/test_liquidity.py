"""Liquidity floor: data-characteristics only, sustained, never filled."""

from __future__ import annotations

import pandas as pd

from trading_bot.data import schema
from trading_bot.data.liquidity import (
    FloorRule,
    count_distribution,
    daily_metrics,
    sustained_floor_date,
)

from .synthetic import make_candles

START = pd.Timestamp("2024-01-01", tz="UTC")


def _daily_frame(counts: list[int | None]) -> pd.DataFrame:
    """None = that calendar day has NO candle (absent, not zero)."""
    rows = [(100.0, 101.0, 99.0, 100.0, 5.0)] * sum(c is not None for c in counts)
    df = make_candles(rows, step=pd.Timedelta(days=1))
    ts = [START + pd.Timedelta(days=i) for i, c in enumerate(counts) if c is not None]
    df[schema.TIMESTAMP] = pd.Series(ts, dtype="datetime64[us, UTC]")
    df[schema.TRADES] = pd.array(
        [c for c in counts if c is not None], dtype="Int64"
    )
    return df


RULE = FloorRule(min_median_trades=500.0, max_missing=1, window_days=30)


def test_floor_date_when_liquidity_arrives():
    # 40 illiquid days (5 trades), then 60 liquid days (1000 trades)
    df = _daily_frame([5] * 40 + [1000] * 60)
    floor = sustained_floor_date(df, RULE)
    # window ending day i covers i-29..i. With 15 liquid days of 30 the median
    # is (5+1000)/2 = 502.5 >= 500, so the rule first holds at i = 54
    assert floor == START + pd.Timedelta(days=54)


def test_floor_is_none_if_rule_fails_at_the_end():
    df = _daily_frame([1000] * 60 + [5] * 40)  # dies at the end
    assert sustained_floor_date(df, RULE) is None


def test_floor_requires_sustained_not_first_touch():
    # liquid, then a mid-life collapse, then liquid again: the floor must be
    # AFTER the collapse, not at the first touch
    df = _daily_frame([1000] * 60 + [5] * 40 + [1000] * 80)
    floor = sustained_floor_date(df, RULE)
    assert floor is not None
    assert floor > START + pd.Timedelta(days=100)  # after the collapse ended


def test_missing_days_block_the_floor():
    counts: list[int | None] = [1000] * 100
    counts[70] = None
    counts[71] = None  # two absent days inside one 30-day window
    df = _daily_frame(counts)
    # windows ending on days 71..99 all contain both missing days -> >1 missing
    assert sustained_floor_date(df, RULE) is None
    relaxed = FloorRule(min_median_trades=500.0, max_missing=2)
    assert sustained_floor_date(df, relaxed) is not None


def test_one_missing_day_is_tolerated():
    counts: list[int | None] = [1000] * 100
    counts[70] = None
    df = _daily_frame(counts)
    floor = sustained_floor_date(df, RULE)
    assert floor == START + pd.Timedelta(days=29)  # earliest complete window


def test_missing_days_count_as_zero_for_the_median():
    metrics = daily_metrics(_daily_frame([1000] * 100), RULE)
    assert metrics["count"].min() == 1000.0
    counts: list[int | None] = [1000] * 100
    for i in range(40, 56):  # 16 of 30 days absent -> median 0
        counts[i] = None
    metrics = daily_metrics(_daily_frame(counts), RULE)
    assert metrics.loc[START + pd.Timedelta(days=55), "roll_median"] == 0.0


def test_metrics_do_not_touch_the_input_frame():
    df = _daily_frame([1000] * 50)
    before = df.copy(deep=True)
    daily_metrics(df, RULE)
    pd.testing.assert_frame_equal(df, before)


def test_count_distribution_percentiles():
    dist = count_distribution(_daily_frame([100] * 50 + [1000] * 50))
    assert dist.loc[0.01] == 100.0
    assert dist.loc[0.99] == 1000.0
