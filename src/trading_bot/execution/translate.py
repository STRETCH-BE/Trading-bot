"""Target position -> concrete orders. Decides; never executes.

This is the second half of the seam. ``orders_for_target`` applies the
rebalance dead-band, sizes the order, and checks exchange minimums, then
hands back orders. Something else fills them, and in production the risk gate
sits between the two.

Arithmetic note: every expression here is written in exactly the order the
pre-refactor engine loop used. Floating-point addition and multiplication are
not associative, so reordering — even into something algebraically identical
— changes the last bits and breaks the equivalence proof. Do not "simplify"
these expressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trading_bot.data import schema
from trading_bot.execution.fill_model import FillModel
from trading_bot.execution.order import Order, PortfolioState, make_client_order_id

EPS = 1e-12


@dataclass
class TargetTranslation:
    """What the translator decided, including the reasons it decided nothing."""

    orders: list[Order] = field(default_factory=list)
    # The strategy target AFTER the risk-budget mapping, i.e. a fraction of
    # EQUITY. Callers that log a target must log this one, never the raw
    # strategy value — otherwise the audit trail mixes two different units.
    equity_target: float = 0.0
    suppressed_by_deadband: bool = False
    skipped_ordermin: bool = False
    skipped_costmin: bool = False
    # Sizing intermediates the filler needs, so it does not recompute them.
    fill_price: float | None = None
    notional: float | None = None
    reason: str = ""

    @property
    def skipped_minimum(self) -> bool:
        return self.skipped_ordermin or self.skipped_costmin


def orders_for_target(
    state: PortfolioState,
    target: float,
    price: float,
    pair: schema.Pair | str | None,
    fill_model: FillModel,
    *,
    min_rebalance_delta: float = 0.0,
    max_allocation: float = 1.0,
    timestamp: pd.Timestamp | None = None,
    strategy: str = "backtest",
    signal_timestamp: pd.Timestamp | None = None,
    order_type: str = "market",
) -> TargetTranslation:
    """Translate a strategy target into zero or one order.

    ``target`` is the strategy's target as a fraction of its RISK BUDGET, in
    [0, 1]. ``max_allocation`` converts that budget into a fraction of equity:

        equity_fraction = target * max_allocation

    So target 1.0 with max_allocation 0.25 means "fully allocated within my
    budget" = 25% of equity, not 100%.

    THE MAPPING LIVES HERE AND ONLY HERE. Both the backtest engine and the
    live cycle call this function, so neither can drift from the other — the
    same rule that keeps FillModel single-sourced.

    Note on the dead-band: ``min_rebalance_delta`` is compared in EQUITY
    space, after the mapping. With max_allocation 0.25 a 0.05 dead-band is
    therefore 20% of the strategy's expressible range rather than 5% — it
    bites proportionally harder. That is deliberate (the dead-band exists to
    avoid churning small ORDERS, and order size is an equity-space quantity)
    but it does mean a rescaled run is not a pure 0.25x of an unscaled one.
    """
    target = target * max_allocation
    equity_at_open = state.equity(price)
    desired_notional = equity_at_open * target
    current_notional = state.units * price
    delta_notional = desired_notional - current_notional

    out = TargetTranslation(equity_target=target)
    if abs(delta_notional) <= EPS:
        out.reason = "already at target"
        return out

    # Dead-band: a target move smaller than min_rebalance_delta is noise, not a
    # decision. A full exit is exempt — going flat is always a real decision,
    # however small the residual.
    current_frac = current_notional / equity_at_open if equity_at_open > EPS else 0.0
    full_exit = target <= 0.0 and state.units > EPS
    if not full_exit and abs(target - current_frac) < min_rebalance_delta:
        out.suppressed_by_deadband = True
        out.reason = (
            f"target move {abs(target - current_frac):.4f} inside dead-band "
            f"{min_rebalance_delta:.4f}"
        )
        return out

    side = "buy" if delta_notional > 0 else "sell"
    fill_price = fill_model.fill_price(side, price)
    fee_rate = fill_model.fee_rate()

    if side == "buy":
        # Spend `budget` of cash in total, fee included.
        budget = min(delta_notional, state.cash)
        notional = budget / (1 + fee_rate)
        order_units = notional / fill_price
    else:
        order_units = min(-delta_notional / fill_price, state.units)
        notional = order_units * fill_price

    order_units = fill_model.round_units(pair, order_units)

    check = fill_model.clears_minimums(pair, order_units, notional)
    if not check.ok:
        out.skipped_ordermin = check.failed_limit == "ordermin"
        out.skipped_costmin = check.failed_limit == "costmin"
        out.reason = check.reason
        return out

    ts = timestamp if timestamp is not None else pd.Timestamp.utcnow()
    sig_ts = signal_timestamp if signal_timestamp is not None else ts
    pair_name = pair.kraken_name if isinstance(pair, schema.Pair) else (pair or "UNKNOWN")

    out.fill_price = fill_price
    out.notional = notional
    out.orders = [
        Order(
            client_order_id=make_client_order_id(
                strategy, pair_name, sig_ts, f"{side}:{target:.6f}"
            ),
            pair=pair_name,
            side=side,
            units=order_units,
            order_type=order_type,  # type: ignore[arg-type]
            limit_price=(
                fill_model.round_price(pair, fill_price) if order_type == "limit" else None
            ),
            reason=f"rebalance {current_frac:.4f} -> {target:.4f}",
            timestamp=ts,
        )
    ]
    out.reason = out.orders[0].reason
    return out
