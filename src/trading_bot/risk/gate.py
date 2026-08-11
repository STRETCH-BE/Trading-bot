"""The mandatory risk gate. Every order passes through ``approve()`` or dies.

Ordering of checks is deliberate: the kill switch and the hard rejects run
before any limit arithmetic, so a halted system rejects without needing a
coherent account state, and an illegal order is refused even if the numbers
would have allowed it.

A breach of the daily-loss or max-drawdown limit does not merely reject the
order — it WRITES THE HALT FILE. That makes the halt survive a restart, which
is what "require manual restart, never auto-resume" means in practice: the
operator must delete the file by hand.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from trading_bot.data import schema
from trading_bot.execution.fill_model import FillModel
from trading_bot.execution.order import Order
from trading_bot.risk.approval import ApprovedOrder, _mint_approved
from trading_bot.risk.limits import AccountState, RiskLimits

log = logging.getLogger(__name__)

DEFAULT_HALT_FILE = Path("HALT")


@dataclass(frozen=True)
class RiskDecision:
    """The outcome. ``approved`` is None whenever ``rejected`` is True."""

    approved: ApprovedOrder | None
    rejected: bool
    limit: str = ""  # which limit fired
    actual: float | None = None
    limit_value: float | None = None
    message: str = ""
    halted: bool = False  # this decision also tripped a system halt

    @property
    def ok(self) -> bool:
        return not self.rejected


class RiskGate:
    """Constructs ``ApprovedOrder``. Nothing else can."""

    def __init__(
        self,
        limits: RiskLimits,
        fill_model: FillModel,
        *,
        halt_file: Path = DEFAULT_HALT_FILE,
    ) -> None:
        self.limits = limits
        self.fill_model = fill_model
        self.halt_file = Path(halt_file)

    # --- kill switch ---------------------------------------------------------

    def is_halted(self) -> bool:
        """Checked on EVERY approval, never cached — `touch HALT` must work
        from an SSH session within one cycle."""
        return self.halt_file.exists()

    def engage_halt(self, reason: str) -> None:
        """Write the HALT file. Survives restart; only a human removes it."""
        stamp = datetime.now(UTC).isoformat()
        self.halt_file.write_text(
            f"HALTED {stamp}\nreason: {reason}\n\n"
            f"Trading is blocked while this file exists. Investigate before "
            f"deleting it. Deleting it is a deliberate human decision.\n"
        )
        log.critical("HALT ENGAGED: %s", reason)

    # --- the gate ------------------------------------------------------------

    def approve(
        self,
        order: Order,
        state: AccountState,
        *,
        now: datetime | pd.Timestamp | None = None,
    ) -> RiskDecision:
        now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
        if now.tz is None:
            now = now.tz_localize("UTC")
        passed: list[str] = []

        for check in (
            self._check_halt_file,
            self._check_known_pair,
            self._check_no_leverage_or_short,
            self._check_stale_data,
            self._check_exchange_minimums,
            self._check_order_size,
            self._check_position_limit,
            self._check_total_exposure,
            self._check_concurrent_positions,
            self._check_daily_loss,
            self._check_max_drawdown,
            self._check_order_rate,
        ):
            decision = check(order, state, now)
            if decision is not None:
                self._log_rejection(decision, order, state)
                return decision
            passed.append(check.__name__.removeprefix("_check_"))

        return RiskDecision(
            approved=_mint_approved(order, now, state.equity, tuple(passed)),
            rejected=False,
            message=f"approved after {len(passed)} checks",
        )

    # --- individual checks: return a RiskDecision to reject, None to pass ----

    def _check_halt_file(self, order, state, now) -> RiskDecision | None:
        if self.is_halted():
            return RiskDecision(
                None, True, "kill_switch", message=(
                    f"HALT file present at {self.halt_file.resolve()} — all order "
                    f"approval is blocked until it is removed by hand"
                ),
                halted=True,
            )
        return None

    def _check_known_pair(self, order, state, now) -> RiskDecision | None:
        if order.pair not in schema.PAIRS:
            self.engage_halt(f"order for unregistered pair {order.pair!r}")
            return RiskDecision(
                None, True, "unknown_pair", message=(
                    f"pair {order.pair!r} is not in schema.PAIRS "
                    f"({sorted(schema.PAIRS)}). HARD REJECT."
                ),
                halted=True,
            )
        return None

    def _check_no_leverage_or_short(self, order, state, now) -> RiskDecision | None:
        """HARD REJECT: no config, no override, loud, and it halts."""
        if order.units <= 0:
            self.engage_halt(f"non-positive order size {order.units!r}")
            return RiskDecision(
                None, True, "short_or_leverage", actual=order.units,
                message=f"order units {order.units!r} implies a short. HARD REJECT.",
                halted=True,
            )
        if order.side == "sell":
            held = state.positions.get(order.pair, 0.0)
            if order.units > held + 1e-9:
                self.engage_halt(
                    f"sell of {order.units} exceeds holding {held} — naked short"
                )
                return RiskDecision(
                    None, True, "short_or_leverage", actual=order.units,
                    limit_value=held,
                    message=(
                        f"selling {order.units} of {order.pair} while holding only "
                        f"{held} would open a SHORT. HARD REJECT."
                    ),
                    halted=True,
                )
        if order.side == "buy":
            price = state.prices.get(order.pair, 0.0)
            cost = order.units * price
            if cost > state.cash + 1e-9:
                self.engage_halt(
                    f"buy costing {cost:.2f} exceeds cash {state.cash:.2f} — margin"
                )
                return RiskDecision(
                    None, True, "short_or_leverage", actual=cost,
                    limit_value=state.cash,
                    message=(
                        f"buying {order.units} of {order.pair} costs {cost:.2f} but "
                        f"cash is {state.cash:.2f} — this implies MARGIN. HARD REJECT."
                    ),
                    halted=True,
                )
        return None

    def _check_stale_data(self, order, state, now) -> RiskDecision | None:
        """Backstop for audit finding 5: clock drift makes a silent no-op look
        like success. Always compute and log the delta, even when it passes."""
        if state.latest_data_time is None:
            return RiskDecision(
                None, True, "stale_data",
                message="no latest_data_time supplied — freshness cannot be proven",
            )
        age = (now - pd.Timestamp(state.latest_data_time)).total_seconds()
        log.info(
            "data freshness: latest=%s now=%s delta=%.1fs limit=%.1fs",
            state.latest_data_time, now, age, self.limits.stale_data_max_seconds,
        )
        if age < 0:
            return RiskDecision(
                None, True, "stale_data", actual=age, limit_value=0.0,
                message=(
                    f"latest candle is {abs(age):.1f}s in the FUTURE relative to "
                    f"wall clock — the machine clock is wrong or data is corrupt"
                ),
            )
        if age > self.limits.stale_data_max_seconds:
            return RiskDecision(
                None, True, "stale_data", actual=age,
                limit_value=self.limits.stale_data_max_seconds,
                message=(
                    f"latest candle is {age:.1f}s old, limit "
                    f"{self.limits.stale_data_max_seconds:.1f}s"
                ),
            )
        return None

    def _check_exchange_minimums(self, order, state, now) -> RiskDecision | None:
        """Delegates to FillModel — never reimplemented here."""
        price = state.prices.get(order.pair, 0.0)
        check = self.fill_model.clears_minimums(order.pair, order.units, order.units * price)
        if not check.ok:
            return RiskDecision(
                None, True, check.failed_limit, actual=order.units,
                message=f"below exchange minimum: {check.reason}",
            )
        return None

    def _check_order_size(self, order, state, now) -> RiskDecision | None:
        """Fat-finger / runaway-loop guard, applied to EVERY order as specified.

        KNOWN INTERACTION, deliberately not silently "fixed" here: applied to
        sells this blocks fully exiting any position worth more than
        max_order_size_pct of equity — i.e. it can prevent risk REDUCTION. And
        on buys it never binds, because max_position_pct (25%) is tighter than
        max_order_size_pct (30%). Net effect on the default limits: this guard
        fires only on exits. See tests/test_risk_limits.py::test_FINDING_*.
        Exempting sells is a one-line change and is the recommended fix, but
        it changes a stated safety limit, so it is the operator's call.
        """
        price = state.prices.get(order.pair, 0.0)
        notional = order.units * price
        pct = notional / state.equity * 100.0 if state.equity > 0 else float("inf")
        if pct > self.limits.max_order_size_pct:
            return RiskDecision(
                None, True, "max_order_size_pct", actual=pct,
                limit_value=self.limits.max_order_size_pct,
                message=(
                    f"order notional {notional:.2f} is {pct:.2f}% of equity "
                    f"{state.equity:.2f}, above the {self.limits.max_order_size_pct}% "
                    f"fat-finger guard"
                ),
            )
        return None

    def _check_position_limit(self, order, state, now) -> RiskDecision | None:
        if order.side != "buy":
            return None
        price = state.prices.get(order.pair, 0.0)
        resulting = state.position_value(order.pair) + order.units * price
        pct = resulting / state.equity * 100.0 if state.equity > 0 else float("inf")
        if pct > self.limits.max_position_pct:
            return RiskDecision(
                None, True, "max_position_pct", actual=pct,
                limit_value=self.limits.max_position_pct,
                message=(
                    f"resulting {order.pair} position {resulting:.2f} would be "
                    f"{pct:.2f}% of equity, above {self.limits.max_position_pct}%"
                ),
            )
        return None

    def _check_total_exposure(self, order, state, now) -> RiskDecision | None:
        if order.side != "buy":
            return None
        price = state.prices.get(order.pair, 0.0)
        resulting = state.total_exposure() + order.units * price
        pct = resulting / state.equity * 100.0 if state.equity > 0 else float("inf")
        if pct > self.limits.max_total_exposure_pct:
            return RiskDecision(
                None, True, "max_total_exposure_pct", actual=pct,
                limit_value=self.limits.max_total_exposure_pct,
                message=(
                    f"resulting total exposure {resulting:.2f} would be {pct:.2f}% "
                    f"of equity, above {self.limits.max_total_exposure_pct}%"
                ),
            )
        return None

    def _check_concurrent_positions(self, order, state, now) -> RiskDecision | None:
        if order.side != "buy":
            return None
        already_open = abs(state.positions.get(order.pair, 0.0)) > 1e-12
        resulting = state.open_position_count() + (0 if already_open else 1)
        if resulting > self.limits.max_concurrent_positions:
            return RiskDecision(
                None, True, "max_concurrent_positions", actual=resulting,
                limit_value=self.limits.max_concurrent_positions,
                message=(
                    f"opening {order.pair} would make {resulting} concurrent "
                    f"positions, above {self.limits.max_concurrent_positions}"
                ),
            )
        return None

    def _check_daily_loss(self, order, state, now) -> RiskDecision | None:
        loss = state.daily_loss_pct()
        if loss >= self.limits.daily_loss_limit_pct:
            self.engage_halt(
                f"daily loss {loss:.2f}% >= limit {self.limits.daily_loss_limit_pct}% "
                f"(day start {state.day_start_equity:.2f}, now {state.equity:.2f})"
            )
            return RiskDecision(
                None, True, "daily_loss_limit_pct", actual=loss,
                limit_value=self.limits.daily_loss_limit_pct,
                message=(
                    f"daily loss {loss:.2f}% breached the "
                    f"{self.limits.daily_loss_limit_pct}% limit. FLATTEN AND HALT — "
                    f"manual restart required."
                ),
                halted=True,
            )
        return None

    def _check_max_drawdown(self, order, state, now) -> RiskDecision | None:
        dd = state.drawdown_pct()
        if dd >= self.limits.max_drawdown_pct:
            self.engage_halt(
                f"drawdown {dd:.2f}% >= limit {self.limits.max_drawdown_pct}% "
                f"(peak {state.peak_equity:.2f}, now {state.equity:.2f})"
            )
            return RiskDecision(
                None, True, "max_drawdown_pct", actual=dd,
                limit_value=self.limits.max_drawdown_pct,
                message=(
                    f"drawdown from peak is {dd:.2f}%, breaching the "
                    f"{self.limits.max_drawdown_pct}% limit. FLATTEN AND HALT — "
                    f"manual restart required."
                ),
                halted=True,
            )
        return None

    def _check_order_rate(self, order, state, now) -> RiskDecision | None:
        cutoff = now - pd.Timedelta(hours=1)
        recent = [pd.Timestamp(t) for t in state.recent_order_times]
        recent = [t for t in recent if (t.tz_localize("UTC") if t.tz is None else t) > cutoff]
        if len(recent) >= self.limits.max_orders_per_hour:
            return RiskDecision(
                None, True, "max_orders_per_hour", actual=len(recent),
                limit_value=self.limits.max_orders_per_hour,
                message=(
                    f"{len(recent)} orders in the last hour, at or above the "
                    f"{self.limits.max_orders_per_hour} circuit breaker — this "
                    f"usually means a logic bug is looping"
                ),
            )
        return None

    # --- logging -------------------------------------------------------------

    def _log_rejection(self, d: RiskDecision, order: Order, state: AccountState) -> None:
        log.warning(
            "RISK REJECT limit=%s actual=%s limit_value=%s | order=%s | "
            "equity=%.2f cash=%.2f positions=%s prices=%s exposure=%.2f "
            "peak=%.2f day_start=%.2f | %s",
            d.limit, d.actual, d.limit_value, order, state.equity, state.cash,
            state.positions, state.prices, state.total_exposure(),
            state.peak_equity, state.day_start_equity, d.message,
        )
