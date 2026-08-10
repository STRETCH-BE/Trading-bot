"""Per-pair exchange minimum order size, with no silent fallback."""

from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.backtest import BacktestConfig, backtest
from trading_bot.data import schema
from trading_bot.data.schema import UnknownPairError, min_order_units

from .synthetic import flat_candles


def test_pairs_have_distinct_ordermins():
    """The whole point: one global value cannot be right for both pairs."""
    assert min_order_units("XBTEUR") == 0.0001
    assert min_order_units("ETHEUR") == 0.01
    assert min_order_units("XBTEUR") != min_order_units("ETHEUR")


def test_lookup_accepts_pair_object_or_name():
    assert min_order_units(schema.PAIRS["XBTEUR"]) == min_order_units("XBTEUR")


def test_unknown_pair_raises_loudly():
    with pytest.raises(UnknownPairError, match="no registered ordermin"):
        min_order_units("DOGEEUR")


def test_unknown_pair_error_names_the_known_pairs():
    with pytest.raises(UnknownPairError, match="ETHEUR"):
        min_order_units("SOLEUR")


def test_unregistered_pair_object_raises():
    rogue = schema.Pair(kraken_name="XRPEUR", ccxt_symbol="XRP/EUR", ordermin=1.0, costmin=1.0)
    with pytest.raises(UnknownPairError, match="not in the registry"):
        min_order_units(rogue)


def test_backtest_uses_the_pairs_own_minimum():
    """An order legal for XBT but illegal for ETH must be skipped for ETH."""
    # 0.005 units per order: above XBT's 0.0001, below ETH's 0.01
    candles = flat_candles(20, price=100_000.0)
    signals = pd.Series([float(i % 2) for i in range(20)])
    config = BacktestConfig(starting_capital=500.0)

    xbt = backtest(candles, lambda c: signals, config, pair="XBTEUR")
    eth = backtest(candles, lambda c: signals, config, pair="ETHEUR")

    assert xbt.fills, "0.005 units is above XBT ordermin and should fill"
    assert xbt.skipped_orders == 0
    assert eth.fills == [], "0.005 units is below ETH ordermin and must be skipped"
    assert eth.skipped_orders > 0


def test_backtest_without_pair_or_override_refuses_to_guess():
    candles = flat_candles(10)
    with pytest.raises(ValueError, match="needs a `pair`"):
        backtest(candles, lambda c: pd.Series([0.0] * len(c)), BacktestConfig())


def test_backtest_with_unknown_pair_raises():
    candles = flat_candles(10)
    with pytest.raises(UnknownPairError):
        backtest(
            candles, lambda c: pd.Series([0.0] * len(c)), BacktestConfig(), pair="ADAEUR"
        )


def test_explicit_override_still_works_for_what_if_runs():
    candles = flat_candles(10)
    result = backtest(
        candles,
        lambda c: pd.Series([0.0] * len(c)),
        BacktestConfig(min_order_units=0.0),
    )
    assert result.skipped_orders == 0


def test_config_yaml_does_not_set_a_global_minimum():
    """A global value in config.yaml would silently override every pair."""
    assert BacktestConfig.load("config.yaml").min_order_units is None
