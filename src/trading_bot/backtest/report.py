"""Formatted text report for a backtest.

Per CLAUDE.md the buy-and-hold benchmark is not optional, and a strategy that
loses to it says so at the very top, before any flattering number.
"""

from __future__ import annotations

import pandas as pd

from trading_bot.backtest.metrics import Metrics

_WIDTH = 66


def format_report(metrics: Metrics, title: str = "BACKTEST REPORT") -> str:
    lines: list[str] = []

    if not metrics.beats_buy_and_hold:
        lines += [
            "!" * _WIDTH,
            "!!  WARNING: THIS STRATEGY LOSES TO BUY-AND-HOLD".ljust(_WIDTH - 2) + "!!",
            "!!".ljust(_WIDTH - 2) + "!!",
            f"!!  strategy      {_pct(metrics.total_return):>12}".ljust(_WIDTH - 2) + "!!",
            f"!!  buy & hold    {_pct(metrics.buy_and_hold_return):>12}".ljust(_WIDTH - 2) + "!!",
            f"!!  excess        {_pct(metrics.excess_return):>12}".ljust(_WIDTH - 2) + "!!",
            "!!".ljust(_WIDTH - 2) + "!!",
            "!!  You would have done better doing nothing but buying.".ljust(_WIDTH - 2) + "!!",
            "!" * _WIDTH,
            "",
        ]

    lines += [
        "=" * _WIDTH,
        title,
        "=" * _WIDTH,
        _row("period", f"{metrics.start:%Y-%m-%d} .. {metrics.end:%Y-%m-%d}"),
        _row("starting capital", f"{metrics.starting_capital:,.2f}"),
        _row("final equity", f"{metrics.final_equity:,.2f}"),
        "",
        _row("total return (net)", _pct(metrics.total_return)),
        _row("CAGR", _pct(metrics.cagr)),
        _row("max drawdown", _magnitude(metrics.max_drawdown)),
        _row("Sharpe", f"{metrics.sharpe:.2f}"),
        _row("Sortino", _fmt_float(metrics.sortino)),
        "",
        _row("trades", str(metrics.n_trades)),
        _row("win rate", _magnitude(metrics.win_rate)),
        _row("avg holding period", _fmt_timedelta(metrics.avg_holding_period)),
        _row("orders skipped (< min size)", str(metrics.skipped_orders)),
        "",
        _row("total fees paid", f"{metrics.total_fees:,.2f}"),
        _row("fees as % of capital", _magnitude(metrics.fees_pct_of_capital)),
    ]

    if metrics.ends_with_open_position:
        lines += [
            "",
            _row("position still open at end", "YES — marked to market"),
            _row("exit cost not yet paid", f"{metrics.unpaid_exit_cost:,.2f}"),
        ]

    lines += [
        "",
        "-" * _WIDTH,
        "BENCHMARK (mandatory)",
        "-" * _WIDTH,
        _row("buy & hold (after 1 round trip)", _pct(metrics.buy_and_hold_return)),
        _row("strategy excess return", _pct(metrics.excess_return)),
        _row(
            "verdict",
            "BEATS buy & hold" if metrics.beats_buy_and_hold else "LOSES to buy & hold",
        ),
        "=" * _WIDTH,
    ]
    return "\n".join(lines)


def format_trade_log(trades: list, limit: int | None = 20) -> str:
    if not trades:
        return "no trades"
    header = (
        f"{'entry':<12} {'exit':<12} {'units':>12} {'entry_px':>11} "
        f"{'exit_px':>11} {'fees':>9} {'pnl':>11}"
    )
    lines = [header, "-" * len(header)]
    shown = trades if limit is None else trades[:limit]
    for t in shown:
        lines.append(
            f"{t.entry_time:%Y-%m-%d}   {t.exit_time:%Y-%m-%d}   "
            f"{t.units:>12.6f} {t.entry_price:>11.2f} {t.exit_price:>11.2f} "
            f"{t.fees:>9.2f} {t.pnl:>11.2f}"
        )
    if limit is not None and len(trades) > limit:
        lines.append(f"... and {len(trades) - limit} more")
    return "\n".join(lines)


def _row(label: str, value: str) -> str:
    return f"  {label:<34} {value:>28}"


def _pct(value: float) -> str:
    """Signed, for quantities where direction is the point (returns, excess)."""
    formatted = f"{value * 100:+.2f}%"
    return "0.00%" if formatted in ("+0.00%", "-0.00%") else formatted


def _magnitude(value: float) -> str:
    """Unsigned, for quantities that are magnitudes (drawdown, win rate, fees)."""
    return f"{abs(value) * 100:.2f}%"


def _fmt_float(value: float) -> str:
    if value == float("inf"):
        return "inf (no losing periods)"
    return f"{value:.2f}"


def _fmt_timedelta(td: pd.Timedelta) -> str:
    if td == pd.Timedelta(0):
        return "-"
    days = td.total_seconds() / 86400
    return f"{days:.1f} days"
