"""min_rebalance_delta dead-band and the Kraken costmin, both engine-enforced."""

from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.backtest import BacktestConfig, backtest
from trading_bot.data.schema import UnknownPairError, cost_minimum

from .synthetic import flat_candles

# --- min_rebalance_delta -----------------------------------------------------


def test_constant_half_target_on_flat_market_is_exactly_one_entry():
    """The mandated test: no fee-drift chatter, one entry, nothing else."""
    candles = flat_candles(200, price=100.0)
    config = BacktestConfig(min_order_units=0.0)  # delta at its 0.05 default
    result = backtest(candles, lambda c: pd.Series([0.5] * len(c)), config)

    assert len(result.fills) == 1
    assert result.fills[0].side == "buy"
    assert result.trades == []  # position still open, marked to market
    assert result.suppressed_rebalances > 0  # the chatter went here instead


def test_target_change_below_delta_is_a_noop():
    candles = flat_candles(50, price=100.0)
    config = BacktestConfig(min_order_units=0.0)
    signals = [0.5] * 25 + [0.53] * 25  # 0.03 < 0.05
    result = backtest(candles, lambda c: pd.Series(signals), config)
    assert len(result.fills) == 1  # entry only; the 0.03 shift never trades


def test_target_change_above_delta_trades():
    candles = flat_candles(50, price=100.0)
    config = BacktestConfig(min_order_units=0.0)
    signals = [0.5] * 25 + [0.60] * 25  # 0.10 >= 0.05
    result = backtest(candles, lambda c: pd.Series(signals), config)
    assert len(result.fills) == 2


def test_full_exit_is_exempt_from_the_dead_band():
    """Going flat is a decision, not chatter — it must execute at any size.

    Enter at 0.5, scale down to 0.04 (inside the dead-band relative to 0),
    then exit to 0. The final 0.04 -> 0 move is below the 0.05 band and must
    trade anyway.
    """
    candles = flat_candles(45, price=100.0)
    config = BacktestConfig(min_order_units=0.0)
    signals = [0.5] * 15 + [0.04] * 15 + [0.0] * 15
    result = backtest(candles, lambda c: pd.Series(signals), config)
    assert result.ends_flat
    assert [f.side for f in result.fills] == ["buy", "sell", "sell"]


def test_entry_smaller_than_dead_band_is_suppressed():
    """Symmetry check for the exemption: a tiny ENTRY is chatter and stays out."""
    candles = flat_candles(30, price=100.0)
    config = BacktestConfig(min_order_units=0.0)
    signals = [0.0] * 5 + [0.04] * 25  # 0.04 < 0.05 from flat
    result = backtest(candles, lambda c: pd.Series(signals), config)
    assert result.fills == []
    assert result.suppressed_rebalances > 0


def test_delta_zero_restores_every_rebalance():
    candles = flat_candles(50, price=100.0)
    config = BacktestConfig(min_order_units=0.0, min_rebalance_delta=0.0)
    result = backtest(candles, lambda c: pd.Series([0.5] * len(c)), config)
    assert result.suppressed_rebalances == 0
    assert len(result.fills) > 1  # the chatter is back, by request


def test_delta_must_be_a_sane_fraction():
    with pytest.raises(ValueError, match="min_rebalance_delta"):
        BacktestConfig(min_rebalance_delta=1.5)


# --- costmin -----------------------------------------------------------------


def test_costmin_registry():
    assert cost_minimum("XBTEUR") == 1.0
    assert cost_minimum("ETHEUR") == 1.0
    with pytest.raises(UnknownPairError, match="costmin"):
        cost_minimum("DOGEEUR")


def test_order_passing_ordermin_but_failing_costmin_is_skipped_and_counted():
    """The mandated test. XBTEUR: ordermin 0.0001 units, costmin 1 EUR.

    At 100 EUR/unit, a 0.008-unit order (0.80 EUR) clears ordermin by 80x and
    fails costmin — it must be skipped, and counted as a costmin skip.
    """
    candles = flat_candles(20, price=100.0)
    config = BacktestConfig(
        starting_capital=10_000.0,
        min_rebalance_delta=0.0,  # isolate costmin from the dead-band
        slippage_bps=0.0,
    )
    signals = [0.0] * 5 + [0.00008] * 15  # 0.8 EUR of a 10k account
    result = backtest(candles, lambda c: pd.Series(signals), config, pair="XBTEUR")

    assert result.fills == []
    assert result.skipped_costmin > 0
    assert result.skipped_orders == result.skipped_costmin  # ordermin was passed


def test_order_clearing_both_minimums_fills():
    candles = flat_candles(20, price=100.0)
    config = BacktestConfig(min_rebalance_delta=0.0, slippage_bps=0.0)
    signals = [0.0] * 5 + [0.0002] * 15  # 2 EUR: clears 1 EUR costmin, 0.0001 ordermin
    result = backtest(candles, lambda c: pd.Series(signals), config, pair="XBTEUR")

    assert len(result.fills) == 1  # the 2 EUR entry cleared both minimums
    fill = result.fills[0]
    assert fill.units * fill.price == pytest.approx(2.0, rel=0.01)
    # with the dead-band deliberately off, later fee-drift chatter produces
    # micro-orders; they fail ordermin (checked first) and never fill
    assert result.skipped_orders > 0
    assert len(result.fills) == 1


def test_costmin_override_for_what_if_runs():
    candles = flat_candles(20, price=100.0)
    config = BacktestConfig(min_rebalance_delta=0.0, slippage_bps=0.0, costmin=50.0)
    signals = [0.0] * 5 + [0.004] * 15  # 40 EUR: above 1, below the 50 override
    result = backtest(candles, lambda c: pd.Series(signals), config, pair="XBTEUR")
    assert result.fills == []
    assert result.skipped_costmin > 0


def test_override_mode_without_pair_defaults_costmin_to_zero():
    """Explicit test mode (min_order_units set, no pair): costmin inert."""
    candles = flat_candles(10, price=100.0)
    config = BacktestConfig(min_order_units=0.0, min_rebalance_delta=0.0)
    signals = [0.0] * 2 + [0.00001] * 8
    result = backtest(candles, lambda c: pd.Series(signals), config)
    assert result.skipped_costmin == 0
