"""One trading cycle, and the loop that schedules it.

Cycle sequence, in order. Step 1 is deliberate and non-obvious:

    1. broker.settle(now)      <- fills scheduled in a PREVIOUS cycle land here
    2. reconcile               (first cycle, or after any halt)
    3. fetch data / top up
    4. validate freshness      (hard fail if stale)
    5. compute signal          (pure strategy function)
    6. translate to orders     (orders_for_target)
    7. risk approval           (ApprovedOrder or rejection)
    8. submit
    9. record everything, including no-action decisions
   10. notify

Settling FIRST matters: a real venue fills asynchronously, so an order
submitted in cycle N may only fill during N+1. Reading positions before
settling would size the next decision against a stale book and double up.

Every record produced by a cycle carries its ``cycle_id``, so the decisions
table answers "why did it do that" for a specific run rather than in general.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from trading_bot.backtest.config import BacktestConfig
from trading_bot.broker.base import Broker
from trading_bot.broker.state import Decision, StateStore
from trading_bot.data import schema
from trading_bot.execution import FillModel, PortfolioState, orders_for_target
from trading_bot.risk import AccountState, RiskGate
from trading_bot.scheduler.reconcile import reconcile

log = logging.getLogger(__name__)

SignalFn = Callable[[pd.DataFrame], pd.Series]


class CycleHalted(RuntimeError):
    """The cycle stopped in a state that requires a human. Never auto-resumed."""


@dataclass
class CycleResult:
    cycle_id: str
    started_at: pd.Timestamp
    settled: int = 0
    reconciled: bool = False
    signal: float | None = None
    orders_submitted: int = 0
    rejections: list[str] = field(default_factory=list)
    decisions: int = 0
    halted: bool = False
    reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.halted


@dataclass
class CycleContext:
    """Everything one cycle needs. Assembled once in run.py."""

    local: StateStore
    broker: Broker
    gate: RiskGate
    fill_model: FillModel
    config: BacktestConfig
    strategy_name: str
    signal_fn: SignalFn
    candles_for: Callable[[str], pd.DataFrame]
    pairs: tuple[str, ...] = ("XBTEUR",)
    dry_run: bool = False
    notifier: Callable[[str, str], None] | None = None

    def notify(self, kind: str, message: str) -> None:
        log.info("NOTIFY[%s] %s", kind, message)
        if self.notifier is not None:
            self.notifier(kind, message)


def run_cycle(
    ctx: CycleContext,
    *,
    now: datetime | pd.Timestamp | None = None,
    reconcile_first: bool = False,
) -> CycleResult:
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if now.tz is None:
        now = now.tz_localize("UTC")
    result = CycleResult(cycle_id=uuid.uuid4().hex[:12], started_at=now)
    log.info("cycle %s starting at %s (dry_run=%s)", result.cycle_id, now, ctx.dry_run)

    # --- 1. settle first: fills from earlier cycles land before we read state
    settled = _settle(ctx, now)
    result.settled = len(settled)
    for state in settled:
        _mirror_fills_locally(ctx, state.order.client_order_id)
        ctx.notify(
            "fill",
            f"{state.order.pair} {state.order.side} {state.filled_units:.8f} "
            f"@ {state.avg_fill_price:.2f} ({state.status})",
        )

    # --- 2. reconcile
    if reconcile_first:
        report = reconcile(ctx.local, ctx.broker)
        result.reconciled = True
        if not report.agrees:
            ctx.gate.engage_halt(f"reconciliation mismatch: {report.render()}")
            ctx.notify("reconciliation_mismatch", report.render())
            result.halted = True
            result.reason = "reconciliation mismatch"
            raise CycleHalted(report.render())

    for pair in ctx.pairs:
        _run_pair(ctx, pair, now, result)

    ctx.broker.mark_to_market(now) if hasattr(ctx.broker, "mark_to_market") else None
    log.info(
        "cycle %s done: settled=%d submitted=%d decisions=%d rejections=%d",
        result.cycle_id, result.settled, result.orders_submitted,
        result.decisions, len(result.rejections),
    )
    return result


def _settle(ctx: CycleContext, now: pd.Timestamp) -> list:
    settle = getattr(ctx.broker, "settle", None)
    return list(settle(now)) if callable(settle) else []


def _mirror_fills_locally(ctx: CycleContext, client_order_id: str) -> None:
    """Copy venue fills the bot did not already have into local state.

    The local store is the bot's BELIEF; the broker is the venue. Keeping them
    as separate records is what makes reconciliation able to detect drift.
    """
    venue = ctx.broker.get_order_status(client_order_id)
    local = ctx.local.get_order(client_order_id)
    if local is None:
        ctx.local.insert_order(venue.order, venue.status, venue.submitted_at)
        local = ctx.local.get_order(client_order_id)

    already = sum(f["units"] for f in ctx.local.fills_for(client_order_id))
    outstanding = venue.filled_units - already
    if outstanding <= 1e-12:
        return

    price = venue.avg_fill_price
    notional = outstanding * price
    fee = venue.fees - sum(f["fee"] for f in ctx.local.fills_for(client_order_id))
    cash_delta = -(notional + fee) if venue.order.side == "buy" else (notional - fee)
    ctx.local.record_fill(
        client_order_id, venue.submitted_at, outstanding, price, fee,
        "taker" if venue.order.order_type == "market" else "maker",
        cash_delta=cash_delta, pair=venue.order.pair, side=venue.order.side,
    )


def _run_pair(
    ctx: CycleContext, pair: str, now: pd.Timestamp, result: CycleResult
) -> None:
    # --- 3. data
    candles = ctx.candles_for(pair)
    if candles is None or candles.empty:
        _record(ctx, result, pair, 0.0, 0.0, "none", "no candles available", now)
        return
    latest = candles[schema.TIMESTAMP].iloc[-1]

    # --- 5. signal (pure)
    signal = float(ctx.signal_fn(candles).iloc[-1])
    result.signal = signal

    # --- current position, read AFTER settling
    positions = {p.pair: p.units for p in ctx.broker.get_positions()}
    units = positions.get(pair, 0.0)
    cash = ctx.broker.get_balance().cash
    price = float(candles[schema.CLOSE].iloc[-1])
    portfolio = PortfolioState(cash=cash, units=units)
    current_fraction = portfolio.position_fraction(price)

    # --- 6. translate
    translation = orders_for_target(
        portfolio, signal, price, pair, ctx.fill_model,
        min_rebalance_delta=ctx.config.min_rebalance_delta,
        timestamp=now, signal_timestamp=latest, strategy=ctx.strategy_name,
    )
    if not translation.orders:
        _record(ctx, result, pair, current_fraction, signal, "none",
                translation.reason, now)
        return

    order = translation.orders[0]

    # --- 7. risk gate. Nothing reaches the broker without an ApprovedOrder.
    account = AccountState(
        equity=portfolio.equity(price),
        cash=cash,
        positions=positions,
        prices={pair: price},
        peak_equity=_peak_equity(ctx, portfolio.equity(price)),
        day_start_equity=_day_start_equity(ctx, now, portfolio.equity(price)),
        latest_data_time=latest,
        recent_order_times=_recent_order_times(ctx),
    )
    decision = ctx.gate.approve(order, account, now=now)
    if decision.rejected:
        result.rejections.append(decision.limit)
        ctx.notify("risk_rejection", f"{decision.limit}: {decision.message}")
        _record(ctx, result, pair, current_fraction, signal, "rejected",
                f"risk[{decision.limit}] {decision.message}", now)
        if decision.halted:
            result.halted = True
            result.reason = decision.limit
            raise CycleHalted(decision.message)
        return

    # --- 8. submit
    if ctx.dry_run:
        _record(ctx, result, pair, current_fraction, signal, "dry_run",
                f"WOULD SUBMIT {order.side} {order.units:.8f} {pair} "
                f"({order.client_order_id})", now)
        log.info("DRY RUN — would submit %s", order)
        return

    state = ctx.broker.submit_order(decision.approved)
    result.orders_submitted += 1
    ctx.local.insert_order(order, state.status, now) if ctx.local.get_order(
        order.client_order_id
    ) is None else None
    _record(ctx, result, pair, current_fraction, signal, "submitted",
            f"{order.side} {order.units:.8f} -> {order.client_order_id} "
            f"({state.status})", now)
    ctx.notify("submit", f"{pair} {order.side} {order.units:.8f} ({state.status})")


def _record(
    ctx: CycleContext, result: CycleResult, pair: str,
    current: float, target: float, action: str, reasoning: str, now: pd.Timestamp,
) -> None:
    """EVERY evaluation is recorded, including 'do nothing'."""
    ctx.local.record_decision(Decision(
        timestamp=now, strategy=ctx.strategy_name, pair=pair,
        current_position=current, target_position=target,
        action_taken=action, reasoning=reasoning, cycle_id=result.cycle_id,
    ))
    result.decisions += 1
    log.info("decision[%s] %s %s: %s", result.cycle_id, pair, action, reasoning)


def _peak_equity(ctx: CycleContext, current: float) -> float:
    history = ctx.local.equity_history()
    return max([row["total_equity"] for row in history] + [current]) if history else current


def _day_start_equity(ctx: CycleContext, now: pd.Timestamp, current: float) -> float:
    day = now.normalize()
    for row in ctx.local.equity_history():
        if pd.Timestamp(row["timestamp"]) >= day:
            return float(row["total_equity"])
    return current


def _recent_order_times(ctx: CycleContext) -> list[pd.Timestamp]:
    return [o.submitted_at for o in ctx.local.all_orders()]


def now_utc() -> datetime:
    return datetime.now(UTC)
