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
from trading_bot.data import schema

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
    skipped_orders: int = 0  # orders below the exchange minimum

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1]) if len(self.equity) else self.config.starting_capital

    @property
    def total_fees(self) -> float:
        return sum(f.fee for f in self.fills)


def backtest(
    candles: pd.DataFrame,
    signal_fn: SignalFn,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run ``signal_fn`` over ``candles`` under ``config``.

    ``signal_fn`` must be pure: same candles in, same signals out, no I/O.
    """
    config = config or BacktestConfig()
    candles = _validate_candles(candles)
    signals = _validate_signals(signal_fn(candles), candles)
    targets = _target_positions(signals)

    ts = candles[schema.TIMESTAMP]
    opens = candles[schema.OPEN].to_numpy(dtype=float)
    closes = candles[schema.CLOSE].to_numpy(dtype=float)

    cash = config.starting_capital
    units = 0.0
    fee_rate = config.fee_rate
    slip = config.slippage_rate

    fills: list[Fill] = []
    trades: list[Trade] = []
    equity_curve: list[float] = []
    skipped = 0

    # open-position bookkeeping for the trade log
    open_entry_time: pd.Timestamp | None = None
    open_cost_basis = 0.0  # cash paid for the units, excluding fees
    open_entry_fee = 0.0

    for i in range(len(candles)):
        target = float(targets.iloc[i])
        open_px = opens[i]

        # --- rebalance at this candle's OPEN, on the previous candle's signal
        equity_at_open = cash + units * open_px
        desired_notional = equity_at_open * target
        current_notional = units * open_px
        delta_notional = desired_notional - current_notional

        if abs(delta_notional) > _EPS:
            side = "buy" if delta_notional > 0 else "sell"
            fill_px = open_px * (1 + slip) if side == "buy" else open_px * (1 - slip)

            if side == "buy":
                # spend `budget` of cash in total, fee included
                budget = min(delta_notional, cash)
                notional = budget / (1 + fee_rate)
                order_units = notional / fill_px
            else:
                order_units = min(-delta_notional / fill_px, units)
                notional = order_units * fill_px

            if order_units < config.min_order_units:
                skipped += 1  # below Kraken's ordermin — never submitted
            else:
                fee = notional * fee_rate
                if side == "buy":
                    cash -= notional + fee
                    units += order_units
                    if open_entry_time is None:
                        open_entry_time = ts.iloc[i]
                        open_cost_basis = 0.0
                        open_entry_fee = 0.0
                    open_cost_basis += notional
                    open_entry_fee += fee
                else:
                    cash += notional - fee
                    units -= order_units
                    if units <= _EPS and open_entry_time is not None:
                        trades.append(
                            Trade(
                                entry_time=open_entry_time,
                                entry_price=open_cost_basis / order_units,
                                exit_time=ts.iloc[i],
                                exit_price=fill_px,
                                units=order_units,
                                entry_fee=open_entry_fee,
                                exit_fee=fee,
                                pnl=(notional - fee) - (open_cost_basis + open_entry_fee),
                            )
                        )
                        units = 0.0
                        open_entry_time = None
                        open_cost_basis = 0.0
                        open_entry_fee = 0.0

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

    # --- liquidate anything still open at the final close, so that the
    # headline return is money actually in hand and "always long" is exactly
    # comparable to buy-and-hold (both pay one round trip).
    if units > _EPS:
        final_px = closes[-1] * (1 - slip)
        notional = units * final_px
        fee = notional * fee_rate
        cash += notional - fee
        if open_entry_time is not None:
            trades.append(
                Trade(
                    entry_time=open_entry_time,
                    entry_price=open_cost_basis / units,
                    exit_time=ts.iloc[-1],
                    exit_price=final_px,
                    units=units,
                    entry_fee=open_entry_fee,
                    exit_fee=fee,
                    pnl=(notional - fee) - (open_cost_basis + open_entry_fee),
                )
            )
        fills.append(
            Fill(
                time=ts.iloc[-1],
                signal_time=pd.NaT,  # not signal-driven: the book is being closed
                side="sell",
                price=final_px,
                reference_price=closes[-1],
                units=units,
                fee=fee,
                cash_after=cash,
                units_after=0.0,
                forced=True,
            )
        )
        units = 0.0
        equity_curve[-1] = cash

    equity = pd.Series(equity_curve, index=ts.to_numpy(), name="equity")
    return BacktestResult(
        equity=equity,
        trades=trades,
        fills=fills,
        config=config,
        candles=candles,
        skipped_orders=skipped,
    )


def buy_and_hold_equity(candles: pd.DataFrame, config: BacktestConfig) -> pd.Series:
    """Equity curve of buying at the first executable open and holding to the end.

    Charged exactly one round trip of fees and slippage, so it is directly
    comparable to a strategy that stays long the whole time. The entry is at
    candle 1's open (not candle 0's) because the engine cannot act before its
    first signal either — same starting line for both.
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
    exit_px = closes[-1] * (1 - slip)
    exit_notional = units * exit_px
    curve[-1] = exit_notional - exit_notional * fee_rate

    return pd.Series(curve, index=candles[schema.TIMESTAMP].to_numpy(), name="buy_and_hold")


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
