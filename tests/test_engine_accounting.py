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

    # It bought once and never sold, so there is no CLOSED trade — the position
    # is held open and marked to market, exactly like the benchmark.
    assert result.trades == []
    assert not result.ends_flat
    assert len(result.fills) == 1 and result.fills[0].side == "buy"
    metrics = compute_metrics(result, CONFIG)
    assert metrics.excess_return == pytest.approx(0.0, abs=1e-6)
    assert metrics.ends_with_open_position
    assert metrics.unpaid_exit_cost > 0


# --- finding 7: no synthetic end-of-run liquidation --------------------------


def test_flat_market_always_long_reports_no_risk_when_costless():
    """The headline check: no synthetic exit tick may appear in the metrics.

    Zero fees and zero slippage on a flat market means nothing whatsoever
    happens after the entry, so Sharpe and drawdown must be exactly zero. The
    old forced liquidation injected a closing fee tick and reported
    Sharpe -1.57 / max DD 0.62% here.
    """
    candles = flat_candles(300, price=100.0)
    config = BacktestConfig(
        starting_capital=10_000.0,
        maker_fee_bps=0.0,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        min_order_units=0.0,
    )
    result = backtest(candles, lambda c: pd.Series([1.0] * len(c)), config)
    metrics = compute_metrics(result, config)

    assert metrics.sharpe == 0.0
    assert metrics.max_drawdown == 0.0
    assert metrics.total_return == pytest.approx(0.0, abs=1e-12)
    assert not any(f.forced for f in result.fills)


def test_flat_market_drawdown_is_entry_cost_only_not_double():
    """With real costs, only the entry is charged — the unpaid exit is reported.

    The strategy chose to buy, so that cost is real. It never chose to sell,
    so no exit cost may appear in the equity curve.
    """
    candles = flat_candles(300, price=100.0)
    config = BacktestConfig(starting_capital=10_000.0, min_order_units=0.0)
    result = backtest(candles, lambda c: pd.Series([1.0] * len(c)), config)
    metrics = compute_metrics(result, config)

    entry_cost = 1 - 1 / ((1 + config.fee_rate) * (1 + config.slippage_rate))
    assert metrics.max_drawdown == pytest.approx(entry_cost, rel=1e-6)

    # exactly one non-zero return in the whole series: the entry
    rets = result.equity.pct_change().dropna()
    assert int((rets != 0).sum()) == 1

    # the exit cost is disclosed, not buried
    assert metrics.ends_with_open_position
    assert metrics.unpaid_exit_cost > 0


def test_open_position_is_marked_to_market_not_closed():
    candles = _rising_market(100)
    result = backtest(candles, lambda c: pd.Series([1.0] * len(c)), CONFIG)
    last_close = float(candles[schema.CLOSE].iloc[-1])
    assert result.open_units > 0
    assert result.final_equity == pytest.approx(result.open_units * last_close, rel=1e-9)


# --- finding 1: partial-sell accounting + the reconciliation invariant --------


@pytest.mark.parametrize(
    "label, signal_values",
    [
        ("binary", [1.0 if (i // 10) % 2 == 0 else 0.0 for i in range(120)]),
        ("fractional", [1.0 if (i // 10) % 2 == 0 else 0.5 for i in range(120)]),
        ("fractional_ladder", [min(1.0, (i % 25) / 20) for i in range(120)]),
        ("all_in_all_out", [float(i % 2) for i in range(120)]),
    ],
)
def test_pnl_reconciles_to_equity_change(label, signal_values):
    """sum(closed pnl) + unrealised == final equity - starting capital.

    This is the invariant that the old partial-sell bug violated by 34,702 on
    a 10,000 account, because a partial sell retired no cost basis.
    """
    candles = _rising_market(120)
    result = backtest(candles, lambda c: pd.Series(signal_values), CONFIG)

    accounted = result.realized_pnl + result.unrealized_pnl
    actual = result.final_equity - CONFIG.starting_capital
    assert accounted == pytest.approx(actual, abs=0.01), (
        f"[{label}] books do not reconcile: "
        f"realised {result.realized_pnl:.2f} + unrealised {result.unrealized_pnl:.2f} "
        f"!= equity change {actual:.2f}"
    )


def test_flat_ending_strategy_reconciles_with_no_unrealised():
    candles = _rising_market(120)
    signals = [1.0 if (i // 10) % 2 == 0 else 0.0 for i in range(120)]
    signals[-1] = 0.0
    result = backtest(candles, lambda c: pd.Series(signals), CONFIG)
    assert result.ends_flat
    assert result.unrealized_pnl == 0.0
    assert result.realized_pnl == pytest.approx(
        result.final_equity - CONFIG.starting_capital, abs=0.01
    )


def test_partial_sell_emits_trades_within_the_real_traded_range():
    """The direct regression: the old bug reported entry_price 850 on an asset
    that never traded above 135, because a partial sell retired no cost basis.

    Trade COUNT is deliberately not asserted — a fractional target rebalances
    on every candle as the price moves, so many small closes are correct.
    """
    candles = _rising_market(60)
    signals = [1.0] * 20 + [0.5] * 20 + [0.0] * 20
    result = backtest(candles, lambda c: pd.Series(signals), CONFIG)

    assert result.trades, "a partial sell must close a proportional slice"

    lo = float(candles[schema.LOW].min())
    hi = float(candles[schema.HIGH].max())
    for t in result.trades:
        assert lo <= t.entry_price <= hi, (
            f"entry price {t.entry_price:.2f} outside the traded range "
            f"[{lo:.2f}, {hi:.2f}] — cost basis was not retired proportionally"
        )

    # every unit bought is eventually accounted for by a closed slice
    bought = sum(f.units for f in result.fills if f.side == "buy")
    closed = sum(t.units for t in result.trades)
    assert closed == pytest.approx(bought, rel=1e-9)


def test_partial_sell_leaves_correct_basis_for_the_remainder():
    """On a flat market every slice must report the SAME entry price.

    If a partial sell failed to retire its share of the basis, later slices
    would inherit it and report inflated entry prices.
    """
    candles = flat_candles(40, price=100.0)
    signals = [1.0] * 10 + [0.5] * 10 + [0.0] * 20
    result = backtest(candles, lambda c: pd.Series(signals), CONFIG)

    assert len(result.trades) >= 2
    first = result.trades[0].entry_price
    for t in result.trades:
        assert t.entry_price == pytest.approx(first, rel=1e-9)


def test_regression_probe1_fractional_books_reconcile():
    """Locks the exact scenario from the audit probe that was off by 34,702."""
    candles = _rising_market(120)
    signals = [1.0 if (i // 10) % 2 == 0 else 0.5 for i in range(120)]
    result = backtest(candles, lambda c: pd.Series(signals), CONFIG)

    drift = (result.realized_pnl + result.unrealized_pnl) - (
        result.final_equity - CONFIG.starting_capital
    )
    assert abs(drift) < 0.01, f"books drifted by {drift:,.2f}"


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
