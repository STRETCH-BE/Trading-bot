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
from trading_bot.risk.gate import DEFAULT_HALT_FILE, RiskDecision, RiskGate
from trading_bot.risk.limits import AccountState, RiskLimits

__all__ = [
    "DEFAULT_HALT_FILE",
    "AccountState",
    "ApprovedOrder",
    "ForgedApprovalError",
    "RiskDecision",
    "RiskGate",
    "RiskLimits",
    "verify_approval",
]
