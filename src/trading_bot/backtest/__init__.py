"""Stage 2: strategy-agnostic backtest engine."""

from trading_bot.backtest.config import BacktestConfig
from trading_bot.backtest.engine import (
    BacktestResult,
    Fill,
    SignalError,
    Trade,
    backtest,
    buy_and_hold_equity,
)
from trading_bot.backtest.metrics import Metrics, compute_metrics
from trading_bot.backtest.report import format_report, format_trade_log

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Fill",
    "Metrics",
    "SignalError",
    "Trade",
    "backtest",
    "buy_and_hold_equity",
    "compute_metrics",
    "format_report",
    "format_trade_log",
]
