"""Causality proof applied to EVERY registered strategy.

This is the gate described in the audit: a strategy cannot be shipped without
demonstrating that its signal at candle N depends only on candles <= N.

The check is prefix-vs-full. For each index i, recompute the signal from
``candles[0..i]`` alone and compare it to the signal at i computed from the
entire series. Any difference means information from the future leaked
backwards.
"""

from __future__ import annotations

import numpy as np
import pytest

from trading_bot.data import schema
from trading_bot.strategies.registry import STRATEGIES

from .synthetic import from_closes

# Markets chosen to exercise different regimes; a peeking strategy will fail
# on at least one of them, usually all.
MARKETS = {
    "trending_up": [100.0 * (1.006**i) for i in range(180)],
    "trending_down": [300.0 * (0.994**i) for i in range(180)],
    "choppy": [100.0 + 25.0 * np.sin(i / 3.0) for i in range(180)],
    "random_walk": list(
        100 * np.cumprod(1 + np.random.default_rng(4).normal(0.0005, 0.025, 180))
    ),
    "step": [100.0] * 60 + [180.0] * 60 + [140.0] * 60,
}


def assert_causal(strategy_fn, candles, name: str) -> None:
    """Prefix-vs-full: no future candle may alter a past signal."""
    full = strategy_fn(candles).to_numpy()
    mismatches: list[int] = []
    for i in range(len(candles)):
        prefix = strategy_fn(candles.iloc[: i + 1].copy()).to_numpy()
        if not np.array_equal(prefix[i], full[i]):
            mismatches.append(i)
    assert not mismatches, (
        f"LOOK-AHEAD in {name!r}: signal at candle(s) {mismatches[:10]} changed when "
        f"future candles were removed. The signal at N must depend only on "
        f"candles <= N."
    )


@pytest.mark.parametrize("strategy_name", sorted(STRATEGIES))
@pytest.mark.parametrize("market_name", sorted(MARKETS))
def test_registered_strategy_is_causal(strategy_name, market_name):
    candles = from_closes(MARKETS[market_name])
    assert_causal(STRATEGIES[strategy_name], candles, strategy_name)


@pytest.mark.parametrize("strategy_name", sorted(STRATEGIES))
def test_registered_strategy_output_contract(strategy_name):
    """Long-only, unlevered, never NaN — checked for every registered strategy."""
    candles = from_closes(MARKETS["random_walk"])
    signals = STRATEGIES[strategy_name](candles)

    assert len(signals) == len(candles)
    assert not signals.isna().any(), "warmup must emit 0.0, never NaN"
    assert (signals >= 0.0).all(), "negative target implies a short"
    assert (signals <= 1.0).all(), "target above 1.0 implies leverage"
    assert list(signals.index) == list(candles.index)


@pytest.mark.parametrize("strategy_name", sorted(STRATEGIES))
def test_registered_strategy_does_not_mutate_input(strategy_name):
    import pandas as pd

    candles = from_closes(MARKETS["choppy"])
    before = candles.copy(deep=True)
    STRATEGIES[strategy_name](candles)
    pd.testing.assert_frame_equal(candles, before)


def test_the_harness_actually_catches_a_peeking_strategy():
    """Proof the gate has teeth.

    A strategy reading candle N+1 must be caught by assert_causal. Without
    this, a harness that silently passed everything would look identical to a
    harness that works.
    """

    def peeking(candles):
        return (candles[schema.CLOSE].shift(-1) > candles[schema.CLOSE]).astype(float)

    candles = from_closes(MARKETS["choppy"])
    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_causal(peeking, candles, "peeking")


def test_the_harness_catches_a_subtle_centred_window():
    """The non-obvious case: a centred rolling mean leaks a little future."""

    def centred(candles):
        ma = candles[schema.CLOSE].rolling(11, center=True, min_periods=1).mean()
        return (candles[schema.CLOSE] > ma).astype(float)

    candles = from_closes(MARKETS["random_walk"])
    with pytest.raises(AssertionError, match="LOOK-AHEAD"):
        assert_causal(centred, candles, "centred")
