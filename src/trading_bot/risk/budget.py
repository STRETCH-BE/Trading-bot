"""Startup coherence check between the strategy's risk budget and the limits.

The failure this prevents: a strategy whose full signal asks for more than the
risk gate will ever approve. That configuration does not fail loudly at
startup — it fails once per cycle, forever, as a rejection, while the bot
looks alive and does nothing. It is exactly what happened before
``strategy_max_allocation`` existed (voltrend asked for 100%, the gate capped
at 25%, every cycle was rejected).

Checked at startup, refusing to start on violation and naming both values.
"""

from __future__ import annotations

from dataclasses import dataclass

from trading_bot.backtest.config import BacktestConfig
from trading_bot.risk.limits import RiskLimits


class AllocationBudgetError(ValueError):
    """The strategy's risk budget is incompatible with the risk limits."""


@dataclass(frozen=True)
class BudgetCheck:
    ok: bool
    messages: tuple[str, ...] = ()

    def raise_if_violated(self) -> None:
        if not self.ok:
            raise AllocationBudgetError(
                "strategy risk budget is incompatible with the risk limits — "
                "refusing to start:\n  " + "\n  ".join(self.messages)
            )


def check_allocation_budget(
    config: BacktestConfig, limits: RiskLimits
) -> BudgetCheck:
    """Assert the strategy can actually express a full signal.

    Two conditions, both stated in the same units so the numbers are directly
    comparable in the failure message:

    1. a full allocation must not exceed the per-position cap;
    2. the largest single order — going flat to fully allocated in one step —
       must not exceed the fat-finger cap.
    """
    allocation_pct = config.strategy_max_allocation * 100.0
    problems: list[str] = []

    if allocation_pct > limits.max_position_pct:
        problems.append(
            f"strategy_max_allocation={config.strategy_max_allocation} "
            f"({allocation_pct:.2f}% of equity) exceeds "
            f"max_position_pct={limits.max_position_pct}%. A full strategy "
            f"signal would be rejected by the position cap on every cycle."
        )

    # The largest order is flat -> fully allocated in one step.
    if allocation_pct > limits.max_order_size_pct:
        problems.append(
            f"largest single order is {allocation_pct:.2f}% of equity "
            f"(flat -> fully allocated) which exceeds "
            f"max_order_size_pct={limits.max_order_size_pct}%. The entry would "
            f"be rejected by the fat-finger guard on every cycle."
        )

    return BudgetCheck(ok=not problems, messages=tuple(problems))


def drift_ceiling(config: BacktestConfig) -> float:
    """The highest exposure the budget can actually reach, as a percentage.

    ``strategy_max_allocation`` is a cap at TRADE time, not a continuous one.
    Between rebalances the position drifts with the price, and the dead-band
    deliberately suppresses the trim until it is ``min_rebalance_delta`` out of
    line. So a 25% budget with a 0.05 dead-band really sits in [20%, 30%].
    """
    return (config.strategy_max_allocation + config.min_rebalance_delta) * 100.0


def describe_budget(config: BacktestConfig, limits: RiskLimits) -> str:
    """Human-readable summary for the startup banner."""
    allocation_pct = config.strategy_max_allocation * 100.0
    ceiling = drift_ceiling(config)
    lines = [
        f"  strategy risk budget : target 1.0 -> {allocation_pct:.2f}% of equity",
        f"  max_position_pct     : {limits.max_position_pct}%"
        f"  ({'OK' if allocation_pct <= limits.max_position_pct else 'VIOLATED'})",
        f"  max_order_size_pct   : {limits.max_order_size_pct}%"
        f"  ({'OK' if allocation_pct <= limits.max_order_size_pct else 'VIOLATED'})",
    ]
    # Not a violation and not a gate bypass — no ORDER creates this exposure,
    # price movement does. But an operator reading 'budget 25%' should not be
    # surprised by a 30% position, so it is stated rather than left implicit.
    if ceiling > limits.max_position_pct:
        lines.append(
            f"  NOTE: between rebalances, price drift can carry the position to "
            f"{ceiling:.2f}% (budget {allocation_pct:.2f}% + dead-band "
            f"{config.min_rebalance_delta * 100:.2f}%), above "
            f"max_position_pct={limits.max_position_pct}%. No order causes this, "
            f"so the gate does not reject it; the gate still blocks any BUY "
            f"while the position is over the cap."
        )
    return "\n".join(lines)
