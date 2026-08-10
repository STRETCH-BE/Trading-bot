"""Liquidity floor: which part of history is liquid enough to backtest at all.

Defined on DATA CHARACTERISTICS ONLY — trade counts and candle presence.
Nothing here looks at returns, volatility, or any strategy output, so the
floor cannot be tuned toward a flattering backtest.

Rule: a calendar day D satisfies the floor when, over the trailing
``window_days`` days ending at D,

    median(daily trade count)      >= min_median_trades, and
    number of missing daily candles <= max_missing;

and the FLOOR DATE is the first day from which the rule holds on every later
day through the end of the data ("sustained" — a floor that lapses is no
floor).

The metrics are computed on an in-memory analysis calendar. Days with no
candle count as zero trades for the median and as missing for the presence
rule. The stored data is not touched: nothing is filled, nothing is written.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_bot.data import schema


@dataclass(frozen=True)
class FloorRule:
    min_median_trades: float
    max_missing: int = 1
    window_days: int = 30


def daily_metrics(df: pd.DataFrame, rule: FloorRule) -> pd.DataFrame:
    """Per-calendar-day trailing metrics for a DAILY candle frame.

    Returns a frame indexed by calendar day with columns:
    ``count`` (trades that day, 0 if no candle), ``present`` (0/1),
    ``roll_median`` and ``roll_missing`` over the trailing window, and ``ok``.
    The first ``window_days - 1`` days have incomplete windows and are never ok.
    """
    if df.empty:
        raise ValueError("empty candle frame")
    ts = df[schema.TIMESTAMP]
    calendar = pd.date_range(ts.iloc[0], ts.iloc[-1], freq="D", tz="UTC")

    counts = pd.Series(0.0, index=calendar)
    present = pd.Series(0.0, index=calendar)
    observed = pd.Series(
        df[schema.TRADES].astype("float64").fillna(0.0).to_numpy(), index=ts.to_numpy()
    )
    counts.update(observed)
    present.update(pd.Series(1.0, index=observed.index))

    roll_median = counts.rolling(rule.window_days, min_periods=rule.window_days).median()
    roll_missing = (
        rule.window_days
        - present.rolling(rule.window_days, min_periods=rule.window_days).sum()
    )
    ok = (roll_median >= rule.min_median_trades) & (roll_missing <= rule.max_missing)

    return pd.DataFrame(
        {
            "count": counts,
            "present": present,
            "roll_median": roll_median,
            "roll_missing": roll_missing,
            "ok": ok.fillna(False),
        }
    )


def sustained_floor_date(df: pd.DataFrame, rule: FloorRule) -> pd.Timestamp | None:
    """First calendar day from which the floor holds through the end of data.

    None if the rule is violated on the final day (nothing is sustained).
    """
    metrics = daily_metrics(df, rule)
    ok = metrics["ok"]
    if not bool(ok.iloc[-1]):
        return None
    failures = ok[~ok]
    if failures.empty:
        # never failed — the earliest day with a complete window qualifies
        return ok.index[rule.window_days - 1]
    return failures.index[-1] + pd.Timedelta(days=1)


def count_distribution(df: pd.DataFrame) -> pd.Series:
    """Percentiles of the daily trade count over PRESENT candles only."""
    counts = df[schema.TRADES].astype("float64").fillna(0.0)
    return counts.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
