"""The execution-timing guarantee: signal from candle N fills at OPEN of N+1.

These are the tests that must fail if anyone ever removes the shift in
``engine._target_positions``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.backtest import BacktestConfig, backtest
from trading_bot.data import schema

from .synthetic import make_candles


def cfg(**kw) -> BacktestConfig:
    """A config with the risk-budget mapping pinned to identity.

    Execution TIMING is what is under test. Pinning the mapping keeps the
    look-ahead profit/loss magnitudes below meaningful — a 0.25 scaling would
    shrink them without changing the property being proved, and the assertion
    thresholds would then be measuring the wrong thing.
    """
    kw.setdefault("strategy_max_allocation", 1.0)
    return BacktestConfig(**kw)


def _alternating_candles(n: int = 60) -> pd.DataFrame:
    """Bars that alternate strictly up, strictly down, up, down, ...

    Even bar i: 100 -> 110. Odd bar i: 110 -> 100. A strategy that knows in
    advance which bars rise makes a fortune; one that acts a bar later buys
    at the top of every up-bar and rides it straight back down.
    """
    rows = []
    for i in range(n):
        if i % 2 == 0:
            o, c = 100.0, 110.0
        else:
            o, c = 110.0, 100.0
        rows.append((o, max(o, c), min(o, c), c, 10.0))
    return make_candles(rows)


def _clairvoyant_signal(candles: pd.DataFrame) -> pd.Series:
    """1.0 on exactly the bars that close above their open — pure look-ahead."""
    return (candles[schema.CLOSE] > candles[schema.OPEN]).astype(float)


def test_lookahead_strategy_loses_money():
    """A signal that is only profitable with look-ahead must LOSE under this engine.

    If execution ever moved to the close of candle N (or the open of N), this
    strategy would print money and this assertion would fail.
    """
    candles = _alternating_candles()
    config = cfg(starting_capital=10_000.0, min_order_units=0.0)
    result = backtest(candles, _clairvoyant_signal, config)

    assert result.final_equity < config.starting_capital, (
        "a look-ahead-only strategy turned a profit — execution timing is broken"
    )
    # it is not a small loss: it buys the top of every up-bar
    assert result.final_equity < config.starting_capital * 0.5
    assert result.trades, "expected the strategy to actually trade"


def test_lookahead_strategy_would_win_with_same_bar_execution():
    """Control: the same signal IS hugely profitable if executed same-bar.

    This proves the previous test's assertion has teeth — the signal really
    does contain exploitable look-ahead, and only the engine's N+1 execution
    prevents it from being harvested.
    """
    candles = _alternating_candles()
    config = cfg(starting_capital=10_000.0, min_order_units=0.0)

    # deliberately cheat: hand the engine tomorrow's signal today
    def shifted_forward(c: pd.DataFrame) -> pd.Series:
        return _clairvoyant_signal(c).shift(-1).fillna(0.0)

    cheating = backtest(candles, shifted_forward, config)
    assert cheating.final_equity > config.starting_capital * 5


def test_every_fill_is_strictly_after_its_signal_candle():
    candles = _alternating_candles(20)
    result = backtest(
        candles, _clairvoyant_signal, cfg(min_order_units=0.0)
    )
    signal_driven = [f for f in result.fills if not f.forced]
    assert signal_driven, "expected signal-driven fills to inspect"
    for fill in signal_driven:
        assert fill.signal_time < fill.time, (
            f"fill at {fill.time} used a signal from {fill.signal_time}"
        )
    # the only fill without a prior signal is the forced end-of-run liquidation
    assert all(pd.isna(f.signal_time) for f in result.fills if f.forced)


def test_fill_price_is_the_open_not_the_close():
    """Entry price must derive from the candle OPEN, never its close."""
    candles = make_candles([(100.0, 105.0, 99.0, 104.0, 5.0)] * 10)
    # go long from the first opportunity onward
    result = backtest(
        candles,
        lambda c: pd.Series([1.0] * len(c)),
        cfg(min_order_units=0.0, slippage_bps=0.0),
    )
    first_buy = next(f for f in result.fills if f.side == "buy")
    assert first_buy.reference_price == 100.0  # the open, not the 104.0 close


def test_signal_on_final_candle_never_executes():
    """There is no candle N+1 for the last signal, so it cannot be traded."""
    candles = make_candles([(100.0, 101.0, 99.0, 100.0, 1.0)] * 5)
    signals = pd.Series([0.0, 0.0, 0.0, 0.0, 1.0])  # only the last candle says buy
    result = backtest(
        candles, lambda c: signals, cfg(min_order_units=0.0)
    )
    assert result.trades == []
    assert result.fills == []
    assert result.final_equity == pytest.approx(result.config.starting_capital)
