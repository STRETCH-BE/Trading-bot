"""The registry of every strategy shipped by this project.

Registration is not bookkeeping — it is the gate. ``tests/test_causality.py``
parametrises over ``STRATEGIES`` and subjects each entry to a prefix-vs-full
causality proof: the signal at candle N recomputed from ``candles[0..N]``
alone must equal the signal at N computed from the whole series.

A strategy added here without being causal fails CI. A strategy NOT added
here is not shipped — the backtest CLI resolves strategies through this
registry, so bypassing the registry means bypassing the entry point too.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from trading_bot.strategies.donchian import DonchianParams, donchian_breakout
from trading_bot.strategies.voltrend import VolTrendParams, voltrend

# A registry entry is a zero-argument-configured signal function: it takes
# candles and returns target positions, with parameters already bound.
StrategyFn = Callable[[pd.DataFrame], pd.Series]

STRATEGIES: dict[str, StrategyFn] = {
    "donchian_breakout": lambda candles: donchian_breakout(candles, DonchianParams()),
    "voltrend": lambda candles: voltrend(candles, VolTrendParams()),
}


def get(name: str) -> StrategyFn:
    try:
        return STRATEGIES[name]
    except KeyError:
        raise KeyError(
            f"unknown strategy {name!r}; registered: {sorted(STRATEGIES)}"
        ) from None
