"""Strategy-agnostic backtest engine.

Execution model, which is the whole point of the engine:

    a signal computed from candle N is executed at the OPEN of candle N+1

Never the close of N. A strategy therefore cannot profit from information it
did not have when the decision was made. The shift happens exactly once, in
``_target_positions``, and ``tests/test_engine_lookahead.py`` fails if it is
ever removed.

Long-only, no leverage, no shorting: target positions live in [0.0, 1.0] and
denote the fraction of equity held in the asset. Anything outside that range
is rejected rather than clamped.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from trading_bot.backtest.config import BacktestConfig
from trading_bot.backtest.holdout import HoldoutUnlock, HoldoutViolation
from trading_bot.data import schema
from trading_bot.execution import FillModel, PortfolioState, orders_for_target

SignalFn = Callable[[pd.DataFrame], pd.Series]

_EPS = 1e-12


class SignalError(ValueError):
    """The signal function returned something the engine refuses to trade."""


@dataclass
class Trade:
    """One completed round trip (flat -> long -> flat)."""

    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    units: float
    entry_fee: float
    exit_fee: float
    pnl: float  # net of both fees and of slippage already baked into prices

    @property
    def fees(self) -> float:
        return self.entry_fee + self.exit_fee

    @property
    def holding_period(self) -> pd.Timedelta:
        return self.exit_time - self.entry_time

    @property
    def return_pct(self) -> float:
        cost = self.units * self.entry_price
        return self.pnl / cost if cost else 0.0


@dataclass
class Fill:
    """A single execution. ``signal_time`` is the candle the decision came from.

    ``signal_time`` is NaT only for the forced end-of-backtest liquidation
    (``forced=True``), which closes the book rather than acting on a signal.
    For every signal-driven fill, ``signal_time < time`` always holds.
    """

    time: pd.Timestamp
    signal_time: pd.Timestamp
    side: str  # "buy" | "sell"
    price: float  # slippage-adjusted execution price
    reference_price: float  # the raw candle open the fill was derived from
    units: float
    fee: float
    cash_after: float
    units_after: float
    forced: bool = False


@dataclass
class BacktestResult:
    equity: pd.Series  # mark-to-market equity at each candle close
    trades: list[Trade]
    fills: list[Fill]
    config: BacktestConfig
    candles: pd.DataFrame = field(repr=False)
    skipped_orders: int = 0  # orders below an exchange minimum (ordermin OR costmin)
    skipped_costmin: int = 0  # subset of skipped_orders: passed ordermin, failed costmin
    suppressed_rebalances: int = 0  # target moves inside the min_rebalance_delta dead-band
    open_units: float = 0.0  # still held at the end, marked to market
    unrealized_pnl: float = 0.0  # P&L of that open position, not yet realised

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1]) if len(self.equity) else self.config.starting_capital

    @property
    def total_fees(self) -> float:
        return sum(f.fee for f in self.fills)

    @property
    def ends_flat(self) -> bool:
        return self.open_units <= _EPS

    @property
    def realized_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)


def backtest(
    candles: pd.DataFrame,
    signal_fn: SignalFn,
    config: BacktestConfig | None = None,
    pair: schema.Pair | str | None = None,
    *,
    holdout_unlock: HoldoutUnlock | None = None,
    strategy_name: str = "backtest",
) -> BacktestResult:
    """Run ``signal_fn`` over ``candles`` under ``config``.

    ``signal_fn`` must be pure: same candles in, same signals out, no I/O.

    ``pair`` supplies the exchange minimums (ordermin and costmin), which
    differ per pair. It may be omitted only when ``config.min_order_units``
    is set explicitly; otherwise the run fails rather than guessing.

    Candles on/after ``config.holdout_start`` raise ``HoldoutViolation``
    unless a one-shot ``holdout_unlock`` token is supplied.
    """
    config = config or BacktestConfig()
    fill_model = FillModel.from_config(config)
    # Resolve limits eagerly so a missing pair fails before any work is done,
    # exactly as the pre-refactor engine did.
    fill_model.limits_for(pair)
    candles = _validate_candles(candles)

    last_ts = candles[schema.TIMESTAMP].iloc[-1]
    if last_ts >= config.holdout_ts:
        if holdout_unlock is None:
            raise HoldoutViolation(
                f"candles reach {last_ts:%Y-%m-%d %H:%M} — on/after the holdout "
                f"boundary {config.holdout_ts:%Y-%m-%d}. The holdout exists so "
                f"one untouched period survives development. Trim the candles, "
                f"or pass an explicitly minted --unlock-holdout token."
            )
        holdout_unlock.consume(
            context=f"backtest over candles ending {last_ts:%Y-%m-%d %H:%M}"
        )
    signals = _validate_signals(signal_fn(candles), candles)
    targets = _target_positions(signals)

    ts = candles[schema.TIMESTAMP]
    opens = candles[schema.OPEN].to_numpy(dtype=float)
    closes = candles[schema.CLOSE].to_numpy(dtype=float)

    cash = config.starting_capital
    units = 0.0

    fills: list[Fill] = []
    trades: list[Trade] = []
    equity_curve: list[float] = []
    skipped = 0
    skipped_costmin = 0
    suppressed = 0

    # Open-position bookkeeping for the trade log, on a weighted-average basis.
    # `open_units` must be tracked explicitly: without it a partial sell cannot
    # know what fraction of the cost basis it just closed.
    open_entry_time: pd.Timestamp | None = None
    open_units = 0.0
    open_cost_basis = 0.0  # cash paid for the open units, excluding fees
    open_entry_fee = 0.0

    for i in range(len(candles)):
        target = float(targets.iloc[i])
        open_px = opens[i]

        # --- rebalance at this candle's OPEN, on the previous candle's signal.
        # Translation decides WHAT to trade; this loop only fills it. In
        # production the risk gate sits between these two steps.
        translation = orders_for_target(
            PortfolioState(cash=cash, units=units),
            target,
            open_px,
            pair if pair is not None else None,
            fill_model,
            min_rebalance_delta=config.min_rebalance_delta,
            timestamp=ts.iloc[i],
            signal_timestamp=ts.iloc[i - 1] if i > 0 else ts.iloc[i],
            strategy=strategy_name,
        )

        if translation.suppressed_by_deadband:
            suppressed += 1
        if translation.skipped_ordermin:
            skipped += 1  # below the exchange's ordermin — never submitted
        if translation.skipped_costmin:
            skipped += 1  # passes ordermin but fails the quote-value costmin
            skipped_costmin += 1

        if translation.orders:
            order = translation.orders[0]
            side = order.side
            fill_px = translation.fill_price
            notional = translation.notional
            order_units = order.units

            fee = fill_model.fee(notional)
            if side == "buy":
                cash -= notional + fee
                units += order_units
                if open_entry_time is None:
                    open_entry_time = ts.iloc[i]
                open_units += order_units
                open_cost_basis += notional
                open_entry_fee += fee
            else:
                cash += notional - fee
                units -= order_units
                if open_entry_time is not None and open_units > _EPS:
                    # Close a PROPORTIONAL slice of the open position. A
                    # partial sell must retire its share of the cost basis,
                    # or the remainder's basis is overstated for good.
                    closed = min(order_units, open_units)
                    share = closed / open_units
                    closed_basis = open_cost_basis * share
                    closed_entry_fee = open_entry_fee * share
                    trades.append(
                        Trade(
                            entry_time=open_entry_time,
                            entry_price=closed_basis / closed,
                            exit_time=ts.iloc[i],
                            exit_price=fill_px,
                            units=closed,
                            entry_fee=closed_entry_fee,
                            exit_fee=fee,
                            pnl=(notional - fee) - (closed_basis + closed_entry_fee),
                        )
                    )
                    open_units -= closed
                    open_cost_basis -= closed_basis
                    open_entry_fee -= closed_entry_fee
                    if open_units <= _EPS:
                        units = 0.0
                        open_units = 0.0
                        open_cost_basis = 0.0
                        open_entry_fee = 0.0
                        open_entry_time = None

            fills.append(
                Fill(
                    time=ts.iloc[i],
                    signal_time=ts.iloc[i - 1] if i > 0 else pd.NaT,
                    side=side,
                    price=fill_px,
                    reference_price=open_px,
                    units=order_units,
                    fee=fee,
                    cash_after=cash,
                    units_after=units,
                )
            )

        equity_curve.append(cash + units * closes[i])

    # A position still open at the end is MARKED TO MARKET, never synthetically
    # closed. Injecting a fake exit here would add a fee/slippage tick that the
    # strategy never chose, and that tick lands in the returns series used for
    # Sharpe and drawdown — on a short window it dominates them.
    unrealized = 0.0
    if units > _EPS:
        unrealized = (units * closes[-1]) - (open_cost_basis + open_entry_fee)

    equity = pd.Series(equity_curve, index=ts.to_numpy(), name="equity")
    return BacktestResult(
        equity=equity,
        trades=trades,
        fills=fills,
        config=config,
        candles=candles,
        skipped_orders=skipped,
        skipped_costmin=skipped_costmin,
        suppressed_rebalances=suppressed,
        open_units=units,
        unrealized_pnl=unrealized,
    )


def buy_and_hold_equity(candles: pd.DataFrame, config: BacktestConfig) -> pd.Series:
    """Equity curve of buying at the first executable open and holding to the end.

    Treated exactly like a strategy that stays long throughout: charged entry
    fee and slippage, then MARKED TO MARKET at the final close. No synthetic
    exit — the benchmark and the strategy must be costed identically or the
    excess-return comparison is rigged.

    The entry is at candle 1's open (not candle 0's) because the engine cannot
    act before its first signal either — same starting line for both.
    """
    if len(candles) < 2:
        return pd.Series(
            [config.starting_capital] * len(candles),
            index=candles[schema.TIMESTAMP].to_numpy(),
            name="buy_and_hold",
        )

    fee_rate = config.fee_rate
    slip = config.slippage_rate
    entry_px = float(candles[schema.OPEN].iloc[1]) * (1 + slip)
    notional = config.starting_capital / (1 + fee_rate)
    units = notional / entry_px

    closes = candles[schema.CLOSE].to_numpy(dtype=float)
    curve = [config.starting_capital]  # flat during candle 0
    curve.extend(units * closes[1:])

    return pd.Series(curve, index=candles[schema.TIMESTAMP].to_numpy(), name="buy_and_hold")


def round_trip_cost(config: BacktestConfig) -> float:
    """Fractional cost of one full round trip: fee+slippage in, fee+slippage out.

    Reported alongside marked-to-market results so the exit cost an open
    position has NOT yet paid stays visible instead of being silently ignored.
    """
    fee, slip = config.fee_rate, config.slippage_rate
    return 1.0 - (1 - slip) * (1 - fee) / ((1 + slip) * (1 + fee))


def _validate_candles(candles: pd.DataFrame) -> pd.DataFrame:
    required = [schema.TIMESTAMP, schema.OPEN, schema.HIGH, schema.LOW, schema.CLOSE]
    missing = [c for c in required if c not in candles.columns]
    if missing:
        raise ValueError(f"candles missing required columns: {missing}")
    if candles.empty:
        raise ValueError("cannot backtest an empty candle set")
    ts = candles[schema.TIMESTAMP]
    if not ts.is_monotonic_increasing:
        raise ValueError("candle timestamps must be monotonically increasing")
    if ts.duplicated().any():
        raise ValueError("candle timestamps must be unique")
    return candles.reset_index(drop=True)


def _validate_signals(signals: pd.Series, candles: pd.DataFrame) -> pd.Series:
    if not isinstance(signals, pd.Series):
        raise SignalError(f"signal_fn must return a pandas Series, got {type(signals).__name__}")
    if len(signals) != len(candles):
        raise SignalError(
            f"signal_fn returned {len(signals)} values for {len(candles)} candles"
        )
    values = pd.Series(signals.to_numpy(), dtype="float64")
    if values.isna().any():
        raise SignalError(
            f"signal_fn returned {int(values.isna().sum())} NaN value(s); a strategy must "
            f"emit 0.0 during warmup, not NaN"
        )
    if (values < 0).any():
        raise SignalError(
            "negative target position implies a short; this engine is long-only"
        )
    if (values > 1).any():
        raise SignalError(
            "target position above 1.0 implies leverage; this engine is unlevered"
        )
    return values


def _target_positions(signals: pd.Series) -> pd.Series:
    """The position to hold during each candle, given signals indexed by candle.

    THE execution-timing guarantee: shifting by one means the position held
    during candle N was decided by the signal from candle N-1, and is entered
    at candle N's open. Candle 0 is unavoidably flat — there is no prior signal.
    """
    return signals.shift(1).fillna(0.0)
