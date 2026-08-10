"""The four mandated engine-validation tests, plus accounting invariants."""

from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.backtest import (
    BacktestConfig,
    SignalError,
    backtest,
    buy_and_hold_equity,
    compute_metrics,
)
from trading_bot.data import schema

from .synthetic import flat_candles, from_closes, make_candles

CONFIG = BacktestConfig(starting_capital=10_000.0, min_order_units=0.0)


def _rising_market(n: int = 200) -> pd.DataFrame:
    closes = [100.0 * (1.004**i) for i in range(n)]
    return from_closes(closes)


# --- mandated test 1 ---------------------------------------------------------


def test_always_hold_equals_buy_and_hold_to_within_a_cent():
    candles = _rising_market()
    result = backtest(candles, lambda c: pd.Series([1.0] * len(c)), CONFIG)
    bh = buy_and_hold_equity(candles, CONFIG)

    assert result.final_equity == pytest.approx(float(bh.iloc[-1]), abs=0.01)

    # and it really did pay exactly one round trip
    assert len(result.trades) == 1
    metrics = compute_metrics(result, CONFIG)
    assert metrics.excess_return == pytest.approx(0.0, abs=1e-6)


# --- mandated test 2 ---------------------------------------------------------


def test_never_trade_returns_exactly_zero():
    candles = _rising_market()
    result = backtest(candles, lambda c: pd.Series([0.0] * len(c)), CONFIG)

    assert result.final_equity == CONFIG.starting_capital  # exact, not approx
    assert result.trades == []
    assert result.fills == []
    assert result.total_fees == 0.0

    metrics = compute_metrics(result, CONFIG)
    assert metrics.total_return == 0.0
    assert metrics.max_drawdown == 0.0


# --- mandated test 3 ---------------------------------------------------------


def test_daily_round_trips_bleed_exactly_the_expected_cost():
    """On a flat market every cent lost is a fee or slippage — so it is countable.

    One round trip costs, on a notional N: fee on the buy, slippage on the buy,
    fee on the sell, slippage on the sell. This asserts the realised drag
    matches that analytically, which proves fees are actually charged rather
    than merely reported.
    """
    n = 365
    candles = flat_candles(n)
    config = BacktestConfig(
        starting_capital=10_000.0,
        taker_fee_bps=26.0,
        slippage_bps=5.0,
        min_order_units=0.0,
    )
    # alternate in/out every candle -> a full round trip every 2 candles
    signals = pd.Series([1.0 if i % 2 == 0 else 0.0 for i in range(n)])
    result = backtest(candles, lambda c: signals, config)

    assert result.trades, "expected round trips"
    assert result.total_fees > 0, "fees were never charged"
    assert result.final_equity < config.starting_capital

    # per round trip the equity multiplier is
    #   buy:  1/((1+slip)(1+fee))     sell: (1-slip)(1-fee)
    fee, slip = config.fee_rate, config.slippage_rate
    per_round_trip = (1 - slip) * (1 - fee) / ((1 + slip) * (1 + fee))
    expected = config.starting_capital * per_round_trip ** len(result.trades)
    assert result.final_equity == pytest.approx(expected, rel=1e-9)

    # sanity: the annual bleed is on the order of (2*fee + 2*slip) per round trip
    drag_per_round_trip = 1 - per_round_trip
    assert drag_per_round_trip == pytest.approx(2 * (fee + slip), rel=0.01)

    # a year of daily flip-flopping is a serious amount of money
    total_drag = 1 - result.final_equity / config.starting_capital
    assert total_drag > 0.09, f"expected heavy fee drag, got {total_drag:.2%}"


def test_fees_scale_with_the_fee_setting():
    """Doubling the fee roughly doubles the fee bill for an identical trade path.

    Only roughly: a higher fee shrinks the equity that later trades compound
    on, so the total bill grows slightly sub-linearly. Anything near-linear
    and strictly increasing is correct; a flat or shrinking bill is not.
    """
    candles = flat_candles(50)
    signals = pd.Series([1.0 if i % 2 == 0 else 0.0 for i in range(50)])

    cheap = backtest(
        candles, lambda c: signals, BacktestConfig(taker_fee_bps=10.0, min_order_units=0.0)
    )
    dear = backtest(
        candles, lambda c: signals, BacktestConfig(taker_fee_bps=20.0, min_order_units=0.0)
    )
    ratio = dear.total_fees / cheap.total_fees
    assert 1.9 < ratio <= 2.0, f"fee bill scaled by {ratio:.3f}, expected just under 2x"
    assert dear.final_equity < cheap.final_equity


def test_maker_mode_is_cheaper_than_taker():
    candles = flat_candles(50)
    signals = pd.Series([1.0 if i % 2 == 0 else 0.0 for i in range(50)])
    taker = backtest(candles, lambda c: signals, BacktestConfig(min_order_units=0.0))
    maker = backtest(
        candles, lambda c: signals, BacktestConfig(fee_mode="maker", min_order_units=0.0)
    )
    assert maker.total_fees < taker.total_fees


# --- mandated test 4 ---------------------------------------------------------


def test_cash_plus_position_value_is_conserved_across_every_transaction():
    """Equity may change only by market moves and by fees actually recorded."""
    candles = _rising_market(120)
    signals = pd.Series([1.0 if (i // 3) % 2 == 0 else 0.0 for i in range(120)])
    config = BacktestConfig(starting_capital=10_000.0, min_order_units=0.0)
    result = backtest(candles, lambda c: signals, config)

    assert result.fills, "expected transactions to check"

    cash = config.starting_capital
    units = 0.0
    for fill in result.fills:
        equity_before = cash + units * fill.reference_price
        if fill.side == "buy":
            notional = fill.units * fill.price
            cash -= notional + fill.fee
            units += fill.units
        else:
            notional = fill.units * fill.price
            cash += notional - fill.fee
            units -= fill.units
        equity_after = cash + units * fill.reference_price

        # slippage moves the fill price away from the reference; the total
        # value destroyed by a transaction is exactly fee + slippage cost
        slip_cost = fill.units * abs(fill.price - fill.reference_price)
        assert equity_after == pytest.approx(equity_before - fill.fee - slip_cost, abs=1e-6)
        assert cash >= -1e-9, "cash went negative — that is leverage"
        assert units >= -1e-9, "units went negative — that is a short"

        # the engine's own books must agree with this independent replay
        assert cash == pytest.approx(fill.cash_after, abs=1e-6)
        assert units == pytest.approx(fill.units_after, abs=1e-6)


def test_cash_never_goes_negative_under_a_volatile_signal():
    candles = _rising_market(150)
    signals = pd.Series([float(i % 2) for i in range(150)])
    result = backtest(candles, lambda c: signals, CONFIG)
    for fill in result.fills:
        assert fill.cash_after >= -1e-9
        assert fill.units_after >= -1e-9


# --- signal validation -------------------------------------------------------


def test_short_signal_is_rejected():
    candles = flat_candles(10)
    with pytest.raises(SignalError, match="short"):
        backtest(candles, lambda c: pd.Series([-1.0] * len(c)), CONFIG)


def test_leveraged_signal_is_rejected():
    candles = flat_candles(10)
    with pytest.raises(SignalError, match="leverage"):
        backtest(candles, lambda c: pd.Series([1.5] * len(c)), CONFIG)


def test_nan_signal_is_rejected():
    candles = flat_candles(10)
    signals = pd.Series([float("nan")] + [0.0] * 9)
    with pytest.raises(SignalError, match="NaN"):
        backtest(candles, lambda c: signals, CONFIG)


def test_wrong_length_signal_is_rejected():
    candles = flat_candles(10)
    with pytest.raises(SignalError, match="returned 3 values"):
        backtest(candles, lambda c: pd.Series([0.0, 1.0, 0.0]), CONFIG)


# --- minimum order size ------------------------------------------------------


def test_orders_below_exchange_minimum_are_skipped_not_submitted():
    candles = flat_candles(20, price=100_000.0)  # tiny unit sizes at this price
    config = BacktestConfig(starting_capital=100.0, min_order_units=0.01)
    signals = pd.Series([1.0 if i % 2 == 0 else 0.0 for i in range(20)])
    result = backtest(candles, lambda c: signals, config)

    assert result.fills == []
    assert result.skipped_orders > 0
    assert result.final_equity == config.starting_capital  # untouched


def test_orders_above_minimum_still_execute():
    candles = flat_candles(20, price=100.0)
    config = BacktestConfig(starting_capital=10_000.0, min_order_units=0.01)
    signals = pd.Series([1.0 if i % 2 == 0 else 0.0 for i in range(20)])
    result = backtest(candles, lambda c: signals, config)
    assert result.fills
    assert result.skipped_orders == 0


# --- candle validation -------------------------------------------------------


def test_non_monotonic_candles_are_rejected():
    candles = flat_candles(5)
    candles = candles.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="monotonically increasing"):
        backtest(candles, lambda c: pd.Series([0.0] * len(c)), CONFIG)


def test_duplicate_candle_timestamps_are_rejected():
    candles = flat_candles(5)
    candles.loc[3, schema.TIMESTAMP] = candles.loc[2, schema.TIMESTAMP]
    with pytest.raises(ValueError, match="unique"):
        backtest(candles, lambda c: pd.Series([0.0] * len(c)), CONFIG)


def test_empty_candles_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        backtest(make_candles([]), lambda c: pd.Series(dtype=float), CONFIG)
