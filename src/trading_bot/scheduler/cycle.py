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
from dataclasses import dataclass, field, replace
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
    equity: float | None = None
    account_rejection: str = ""

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
    # Reconcile every N cycles. 1 = every cycle, which is the default because
    # the alternative shipped as "once, at boot": a bot running for months
    # compared its books to the venue exactly one time. Drift that appeared at
    # cycle 2 was never detected. Raise this only if reconciliation becomes
    # expensive against a real venue, and never past a handful of cycles.
    reconcile_every_cycles: int = 1
    _cycles_since_reconcile: int = field(default=0, repr=False)

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

    # --- 0. the kill switch, checked BEFORE anything else.
    # The gate also checks it, but only when there is an order to approve — a
    # strategy sitting flat would never consult it, so `touch HALT` could leave
    # the process running indefinitely. Checked here, it stops the next cycle
    # whatever the strategy wants to do.
    if ctx.gate.is_halted():
        result.halted = True
        result.reason = "halt file present"
        message = (
            f"HALT file present at {ctx.gate.halt_file.resolve()} — stopping. "
            f"Remove it by hand to resume."
        )
        log.critical(message)
        ctx.notify("halt", message)
        raise CycleHalted(message)

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

    # --- 2. reconcile. Every cycle by default, NOT once at boot.
    ctx._cycles_since_reconcile += 1
    if reconcile_first or ctx._cycles_since_reconcile >= ctx.reconcile_every_cycles:
        ctx._cycles_since_reconcile = 0
        report = reconcile(ctx.local, ctx.broker)
        result.reconciled = True
        if not report.agrees:
            ctx.gate.engage_halt(f"reconciliation mismatch: {report.render()}")
            ctx.notify("reconciliation_mismatch", report.render())
            result.halted = True
            result.reason = "reconciliation mismatch"
            raise CycleHalted(report.render())

    # --- 3. data for every pair, fetched BEFORE any risk evaluation because
    # the account state needs prices and a freshness timestamp.
    candles_by_pair = {pair: ctx.candles_for(pair) for pair in ctx.pairs}
    usable = {
        pair: df for pair, df in candles_by_pair.items()
        if df is not None and not df.empty
    }
    for pair in ctx.pairs:
        if pair not in usable:
            _record(ctx, result, pair, 0.0, 0.0, "none", "no candles available", now)
    if not usable:
        log.warning("cycle %s: no usable data for any pair", result.cycle_id)
        return result

    prices = {pair: float(df[schema.CLOSE].iloc[-1]) for pair, df in usable.items()}

    # --- 4. record equity to the LOCAL store, from the bot's OWN books.
    # This is what makes peak_equity and day_start_equity meaningful. It was
    # previously written only to the venue store by an optional broker method,
    # so the local history was always empty and drawdown was always 0%.
    equity = _record_equity(ctx, now, prices)
    result.equity = equity

    # --- 5. account-scoped risk, evaluated whether or not anything wants to
    # trade. A flat strategy used to skip every limit in the system.
    account = _account_state(ctx, now, prices, usable, equity)
    breach = ctx.gate.check_account(account, now=now)
    if breach is not None:
        result.rejections.append(breach.limit)
        result.account_rejection = breach.limit
        ctx.notify("account_risk", f"{breach.limit}: {breach.message}")
        for pair in usable:
            _record(ctx, result, pair, _fraction(account, pair), 0.0, "account_blocked",
                    f"account risk[{breach.limit}] {breach.message}", now)
        if breach.halted:
            result.halted = True
            result.reason = breach.limit
            raise CycleHalted(breach.message)
        log.warning("cycle %s: account risk %s — no order logic will run",
                    result.cycle_id, breach.limit)
        return result

    # --- 6. per-pair order logic
    for pair in usable:
        _run_pair(ctx, pair, now, result, usable[pair], account)

    log.info(
        "cycle %s done: equity=%.2f settled=%d submitted=%d decisions=%d rejections=%d",
        result.cycle_id, equity, result.settled, result.orders_submitted,
        result.decisions, len(result.rejections),
    )
    return result


def _record_equity(ctx: CycleContext, now: pd.Timestamp, prices: dict) -> float:
    """Snapshot the bot's OWN equity into its OWN store, every cycle.

    Deliberately computed from ``ctx.local`` — the bot's belief — not from the
    broker and not from the venue. The broker's view is what reconciliation
    checks this against; using it here would make the drawdown limit trust the
    very thing it is meant to be independent of.

    No hasattr guard: a broker that cannot support this does not get to make
    the drawdown and daily-loss limits silently inert.
    """
    cash = ctx.local.get_cash()
    value = sum(p.units * prices.get(p.pair, 0.0) for p in ctx.local.positions())
    ctx.local.record_equity(now, cash, value)
    return cash + value


def _account_state(
    ctx: CycleContext, now: pd.Timestamp, prices: dict, candles: dict, equity: float
) -> AccountState:
    """The account as the BOT sees it. Never hand-constructed in tests."""
    positions = {p.pair: p.units for p in ctx.local.positions()}
    # The most stale pair governs freshness: trading on one fresh feed while
    # another is days behind is exactly the situation the limit exists for.
    latest = min(df[schema.TIMESTAMP].iloc[-1] for df in candles.values())
    return AccountState(
        equity=equity,
        cash=ctx.local.get_cash(),
        positions=positions,
        prices=prices,
        peak_equity=_peak_equity(ctx, equity),
        day_start_equity=_day_start_equity(ctx, now, equity),
        latest_data_time=latest,
        recent_order_times=_recent_order_times(ctx),
    )


def _fraction(account: AccountState, pair: str) -> float:
    value = account.positions.get(pair, 0.0) * account.prices.get(pair, 0.0)
    return value / account.equity if account.equity > 0 else 0.0


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
    ctx: CycleContext, pair: str, now: pd.Timestamp, result: CycleResult,
    candles: pd.DataFrame, account: AccountState,
) -> None:
    latest = candles[schema.TIMESTAMP].iloc[-1]

    # --- signal (pure)
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
        max_allocation=ctx.config.strategy_max_allocation,
        timestamp=now, signal_timestamp=latest, strategy=ctx.strategy_name,
    )
    # Log the MAPPED target, not the raw signal: current_fraction is a fraction
    # of equity, so a raw 1.0 next to it would read as "quadruple the position"
    # when the bot is in fact fully allocated. The raw signal is preserved in
    # the reasoning text so the mapping stays auditable from the row alone.
    target_frac = translation.equity_target
    budget = (
        f"signal {signal:.4f} x budget "
        f"{ctx.config.strategy_max_allocation:.4f} = {target_frac:.4f} of equity"
    )

    if not translation.orders:
        _record(ctx, result, pair, current_fraction, target_frac, "none",
                f"{translation.reason} [{budget}]", now)
        return

    order = translation.orders[0]

    # --- risk gate. Nothing reaches the broker without an ApprovedOrder.
    # The account-scoped fields come from the cycle-level state built in
    # run_cycle; only the order-scoped view of THIS pair is refreshed here,
    # from the broker's post-settlement position.
    order_state = replace(
        account,
        equity=portfolio.equity(price),
        cash=cash,
        positions=positions,
        prices={**account.prices, pair: price},
        latest_data_time=latest,
    )
    decision = ctx.gate.approve(order, order_state, now=now)
    if decision.rejected:
        result.rejections.append(decision.limit)
        ctx.notify("risk_rejection", f"{decision.limit}: {decision.message}")
        _record(ctx, result, pair, current_fraction, target_frac, "rejected",
                f"risk[{decision.limit}] {decision.message} [{budget}]", now)
        if decision.halted:
            result.halted = True
            result.reason = decision.limit
            raise CycleHalted(decision.message)
        return

    # --- 8. submit
    if ctx.dry_run:
        _record(ctx, result, pair, current_fraction, target_frac, "dry_run",
                f"WOULD SUBMIT {order.side} {order.units:.8f} {pair} "
                f"({order.client_order_id}) [{budget}]", now)
        log.info("DRY RUN — would submit %s", order)
        return

    state = ctx.broker.submit_order(decision.approved)
    result.orders_submitted += 1
    ctx.local.insert_order(order, state.status, now) if ctx.local.get_order(
        order.client_order_id
    ) is None else None
    _record(ctx, result, pair, current_fraction, target_frac, "submitted",
            f"{order.side} {order.units:.8f} -> {order.client_order_id} "
            f"({state.status}) [{budget}]", now)
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
    """Equity as of the START of today.

    Preference order, and the ordering is the whole point:

    1. the LAST snapshot taken before today began — i.e. yesterday's close;
    2. failing that, the first snapshot taken today;
    3. failing that, the current equity (a genuinely fresh account).

    Taking (2) first, which is the obvious implementation, makes the limit
    unfireable at the cadence this bot actually runs. The default interval is
    one day, so the first snapshot of the day IS the current cycle: day_start
    would equal current equity and ``daily_loss_pct()`` would return 0.0 on
    every cycle forever. Yesterday's close is both the correct meaning of
    "since the day started" and the only reading that works at daily cadence.
    """
    day = now.normalize()
    history = ctx.local.equity_history()

    before = [r for r in history if pd.Timestamp(r["timestamp"]) < day]
    if before:
        return float(before[-1]["total_equity"])

    for row in history:
        if pd.Timestamp(row["timestamp"]) >= day:
            return float(row["total_equity"])
    return current


def _recent_order_times(ctx: CycleContext) -> list[pd.Timestamp]:
    return [o.submitted_at for o in ctx.local.all_orders()]


def now_utc() -> datetime:
    return datetime.now(UTC)
