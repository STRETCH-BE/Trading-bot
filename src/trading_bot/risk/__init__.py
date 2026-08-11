"""Stage 5b: the mandatory risk gate.

``RiskGate.approve()`` is the only producer of ``ApprovedOrder``, and
``ApprovedOrder`` is the only thing a broker will accept. See
``approval.py`` for how that is enforced and where the enforcement stops.
"""

from trading_bot.risk.approval import (
    ApprovedOrder,
    ForgedApprovalError,
    verify_approval,
)
from trading_bot.risk.budget import (
    AllocationBudgetError,
    BudgetCheck,
    check_allocation_budget,
    describe_budget,
    drift_ceiling,
)
from trading_bot.risk.gate import DEFAULT_HALT_FILE, RiskDecision, RiskGate
from trading_bot.risk.limits import AccountState, RiskLimits

__all__ = [
    "DEFAULT_HALT_FILE",
    "AllocationBudgetError",
    "BudgetCheck",
    "AccountState",
    "ApprovedOrder",
    "ForgedApprovalError",
    "RiskDecision",
    "RiskGate",
    "RiskLimits",
    "check_allocation_budget",
    "describe_budget",
    "drift_ceiling",
    "verify_approval",
]
