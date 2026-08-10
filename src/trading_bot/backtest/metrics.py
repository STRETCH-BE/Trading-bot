"""Performance metrics for a completed backtest.

Every metric is computed from the mark-to-market equity curve and the trade
log, both of which are already net of fees and slippage. Nothing here
re-derives returns from prices, so no metric can quietly ignore costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_bot.backtest.config import BacktestConfig
from trading_bot.backtest.engine import BacktestResult, buy_and_hold_equity
from trading_bot.data import schema

SECONDS_PER_YEAR = 365.25 * 24 * 3600


@dataclass(frozen=True)
class Metrics:
    total_return: float  # fraction, e.g. 0.12 = +12%
    cagr: float
    max_drawdown: float  # positive fraction, e.g. 0.30 = -30% peak to trough
    sharpe: float
    sortino: float
    win_rate: float
    n_trades: int
    avg_holding_period: pd.Timedelta
    total_fees: float
    fees_pct_of_capital: float

    # mandated by CLAUDE.md: never report a strategy without its benchmark
    buy_and_hold_return: float  # after one round trip of fees
    excess_return: float  # strategy minus buy-and-hold

    starting_capital: float
    final_equity: float
    start: pd.Timestamp
    end: pd.Timestamp
    skipped_orders: int

    # Equity is marked to market, so a position still open at the end has not
    # paid its exit cost yet. Surfaced rather than silently ignored.
    ends_with_open_position: bool = False
    unpaid_exit_cost: float = 0.0

    @property
    def beats_buy_and_hold(self) -> bool:
        return self.excess_return > 0


def infer_periods_per_year(index: pd.Index) -> float:
    """Annualisation factor from the median spacing of the equity curve."""
    if len(index) < 2:
        return 365.0
    deltas = pd.Series(pd.to_datetime(index)).diff().dropna()
    median = deltas.median()
    if pd.isna(median) or median.total_seconds() <= 0:
        return 365.0
    return SECONDS_PER_YEAR / median.total_seconds()


def compute_metrics(result: BacktestResult, config: BacktestConfig | None = None) -> Metrics:
    config = config or result.config
    equity = result.equity
    start_capital = config.starting_capital
    final = float(equity.iloc[-1])

    returns = equity.pct_change().dropna()
    ppy = config.periods_per_year or infer_periods_per_year(equity.index)

    total_return = final / start_capital - 1.0

    start_ts = pd.Timestamp(equity.index[0])
    end_ts = pd.Timestamp(equity.index[-1])
    years = max((end_ts - start_ts).total_seconds() / SECONDS_PER_YEAR, 1e-9)
    cagr = (final / start_capital) ** (1 / years) - 1.0 if final > 0 else -1.0

    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak
    max_drawdown = float(-drawdown.min()) if len(drawdown) else 0.0

    sharpe = _sharpe(returns, ppy, config.risk_free_rate)
    sortino = _sortino(returns, ppy, config.risk_free_rate)

    trades = result.trades
    wins = [t for t in trades if t.pnl > 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    avg_hold = (
        sum((t.holding_period for t in trades), pd.Timedelta(0)) / len(trades)
        if trades
        else pd.Timedelta(0)
    )

    total_fees = result.total_fees

    bh_curve = buy_and_hold_equity(result.candles, config)
    bh_return = float(bh_curve.iloc[-1]) / start_capital - 1.0

    return Metrics(
        total_return=total_return,
        cagr=cagr,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        sortino=sortino,
        win_rate=win_rate,
        n_trades=len(trades),
        avg_holding_period=avg_hold,
        total_fees=total_fees,
        fees_pct_of_capital=total_fees / start_capital,
        buy_and_hold_return=bh_return,
        excess_return=total_return - bh_return,
        starting_capital=start_capital,
        final_equity=final,
        start=start_ts,
        end=end_ts,
        skipped_orders=result.skipped_orders,
        ends_with_open_position=not result.ends_flat,
        unpaid_exit_cost=(
            0.0
            if result.ends_flat
            else result.open_units * float(result.candles[schema.CLOSE].iloc[-1])
            * (config.fee_rate + config.slippage_rate)
        ),
    )


def _sharpe(returns: pd.Series, ppy: float, risk_free: float) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / ppy
    sd = float(np.std(excess, ddof=1))
    if sd == 0 or math.isnan(sd):
        return 0.0
    return float(np.mean(excess)) / sd * math.sqrt(ppy)


def _sortino(returns: pd.Series, ppy: float, risk_free: float) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / ppy
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if float(np.mean(excess)) > 0 else 0.0
    dd = float(np.sqrt(np.mean(np.square(downside))))
    if dd == 0:
        return 0.0
    return float(np.mean(excess)) / dd * math.sqrt(ppy)
