"""Startup reconciliation: local belief vs what the venue actually reports.

BROKER-AGNOSTIC BY CONSTRUCTION. This module imports the ``Broker`` ABC and
nothing else — no PaperBroker import, no KrakenBroker import, no isinstance
check anywhere. It compares the local SQLite store against whatever
``get_balance``/``get_positions``/``get_open_orders`` return.

For that comparison to mean anything in paper mode, the PaperBroker must be
given its OWN store, separate from the bot's local one — the simulated venue
is a different database from the bot's belief about it. That is wiring, done
in ``run.py``, not a special case here.

On any disagreement: log the full diff, alert, HALT. Never guess, never
auto-correct, never assume local state is right. A human resolves it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from trading_bot.broker.base import Broker
from trading_bot.broker.state import StateStore

log = logging.getLogger(__name__)

CASH_TOLERANCE = 0.01  # one cent
UNITS_TOLERANCE = 1e-9


@dataclass
class Discrepancy:
    kind: str  # "cash" | "position" | "open_order"
    key: str
    local: object
    venue: object

    def describe(self) -> str:
        return f"{self.kind}[{self.key}]: local={self.local!r} venue={self.venue!r}"


@dataclass
class ReconciliationReport:
    discrepancies: list[Discrepancy] = field(default_factory=list)
    local_cash: float = 0.0
    venue_cash: float = 0.0
    local_positions: dict[str, float] = field(default_factory=dict)
    venue_positions: dict[str, float] = field(default_factory=dict)
    local_open_orders: list[str] = field(default_factory=list)
    venue_open_orders: list[str] = field(default_factory=list)

    @property
    def agrees(self) -> bool:
        return not self.discrepancies

    def render(self) -> str:
        lines = [
            "RECONCILIATION",
            f"  cash    local={self.local_cash:,.2f}  venue={self.venue_cash:,.2f}",
            f"  positions local={self.local_positions}  venue={self.venue_positions}",
            f"  open orders local={sorted(self.local_open_orders)} "
            f"venue={sorted(self.venue_open_orders)}",
        ]
        if self.agrees:
            lines.append("  RESULT: AGREE")
        else:
            lines.append(f"  RESULT: {len(self.discrepancies)} DISCREPANCY(IES) — HALTING")
            lines += [f"    - {d.describe()}" for d in self.discrepancies]
        return "\n".join(lines)


def reconcile(local: StateStore, broker: Broker) -> ReconciliationReport:
    """Compare local state against the broker's reported state. Read-only."""
    report = ReconciliationReport()

    report.local_cash = local.get_cash()
    report.venue_cash = broker.get_balance().cash
    if abs(report.local_cash - report.venue_cash) > CASH_TOLERANCE:
        report.discrepancies.append(
            Discrepancy("cash", "EUR", report.local_cash, report.venue_cash)
        )

    report.local_positions = {
        p.pair: p.units for p in local.positions() if abs(p.units) > UNITS_TOLERANCE
    }
    report.venue_positions = {
        p.pair: p.units for p in broker.get_positions() if abs(p.units) > UNITS_TOLERANCE
    }
    for pair in sorted(set(report.local_positions) | set(report.venue_positions)):
        lo = report.local_positions.get(pair, 0.0)
        ve = report.venue_positions.get(pair, 0.0)
        if abs(lo - ve) > UNITS_TOLERANCE:
            report.discrepancies.append(Discrepancy("position", pair, lo, ve))

    report.local_open_orders = [o.order.client_order_id for o in local.open_orders()]
    report.venue_open_orders = [
        o.order.client_order_id for o in broker.get_open_orders()
    ]
    for coid in sorted(set(report.local_open_orders) | set(report.venue_open_orders)):
        in_local = coid in report.local_open_orders
        in_venue = coid in report.venue_open_orders
        if in_local != in_venue:
            report.discrepancies.append(
                Discrepancy("open_order", coid, in_local, in_venue)
            )

    if report.agrees:
        log.info("reconciliation: local and venue agree\n%s", report.render())
    else:
        log.critical("RECONCILIATION MISMATCH\n%s", report.render())
    return report
