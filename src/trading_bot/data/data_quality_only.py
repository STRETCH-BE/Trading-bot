"""FORWARD-LOOKING helpers. **Never import this from a signal path.**

Everything in this module deliberately reads candles AFTER the one being
examined. That is correct for judging data quality after the fact — deciding
whether a zero-volume candle sat inside an active trading period requires
knowing what happened next — and catastrophic for generating a trading
signal, where it is textbook look-ahead bias.

The rule this module exists to enforce:

    trading_bot.strategies.*  and  trading_bot.backtest.*
    MUST NOT import this module, directly or transitively.

``tests/test_no_lookahead_imports.py`` asserts exactly that and fails CI if
the boundary is ever crossed. If you need something from here inside a
strategy, you do not need it — you need a causal reformulation of it.
"""

from __future__ import annotations

import pandas as pd

from trading_bot.data import schema


def zero_volume_in_active_period(
    clean: pd.DataFrame, *, window: int, min_active_fraction: float
) -> list[pd.Timestamp]:
    """Timestamps of zero-volume candles with active trading on BOTH sides.

    LOOK-AHEAD BY DESIGN: the ``after`` term reads candles later than the row
    being judged. Data-quality reporting only.
    """
    volume = clean[schema.VOLUME].reset_index(drop=True)
    ts = clean[schema.TIMESTAMP].reset_index(drop=True)
    nonzero = (volume > 0).astype("float64")

    # Fraction of the up-to-`window` candles strictly before / after each row
    # that traded. Rows at the edges see partial windows (min_periods=1); the
    # first row has no "before" and the last no "after", so neither can be
    # "in the middle" and both stay unflagged via fillna(False).
    before = nonzero.shift(1).rolling(window, min_periods=1).mean()
    # <-- this reversal is the forward-looking step that must never reach a strategy
    after = nonzero.iloc[::-1].shift(1).rolling(window, min_periods=1).mean().iloc[::-1]

    flagged = (volume == 0) & (before >= min_active_fraction) & (after >= min_active_fraction)
    return list(ts[flagged.fillna(False)])
