"""``ApprovedOrder`` — the only thing a broker will accept.

Python has no true private constructor, so a single trick is not enough. The
guarantee here is layered, and each layer catches a different class of bypass:

1. **Type.** Broker signatures accept ``ApprovedOrder``, never ``Order``. This
   catches the ordinary mistake: someone wiring a raw order into execution.
2. **Mint key.** The constructor demands a sentinel that exists only inside a
   closure in this module and is never bound to a module attribute. Direct
   construction therefore fails.
3. **Signature.** Every approval carries an HMAC over the order's identity,
   keyed by a per-process secret held only here. Brokers *verify* rather than
   trust, so an instance forged by any route that skips the mint — object
   ``__new__``, unpickling, subclassing, prying open the closure cell — still
   fails at submission because it cannot produce a valid signature without
   the secret.

Layer 3 is what turns "structurally discouraged" into "verifiable". Forging
an approval requires extracting a secret from this module's namespace, which
is a deliberate, greppable act rather than an accident or a careless refactor.

Honest limitation, stated rather than hidden: a process that can execute
arbitrary Python in-process can eventually read anything, including the
secret. No pure-Python construct prevents that. What this design does
guarantee is that no *accidental* path, and no path that looks like normal
code, reaches a broker unapproved.
"""

from __future__ import annotations

import hmac
import secrets
from dataclasses import dataclass
from hashlib import sha256

import pandas as pd

from trading_bot.execution.order import Order

# Per-process signing secret. Never exported, never logged, never persisted.
_SIGNING_SECRET = secrets.token_bytes(32)


def _signature_payload(order: Order, equity_at_approval: float) -> bytes:
    """The exact identity being vouched for. Any change invalidates the approval."""
    return "|".join(
        [
            order.client_order_id,
            order.pair,
            order.side,
            repr(order.units),
            order.order_type,
            repr(order.limit_price),
            repr(equity_at_approval),
        ]
    ).encode("utf-8")


def _sign(order: Order, equity_at_approval: float) -> str:
    return hmac.new(
        _SIGNING_SECRET, _signature_payload(order, equity_at_approval), sha256
    ).hexdigest()


def _build_approval_machinery():
    """Create ApprovedOrder with a mint key that lives only in this closure."""
    mint_key = object()

    @dataclass(frozen=True)
    class ApprovedOrder:
        """An order the risk gate has cleared. Only ``_mint`` can make a valid one."""

        order: Order
        approved_at: pd.Timestamp
        equity_at_approval: float
        approval_signature: str
        checks_passed: tuple[str, ...]

        def __init__(self, *args, _key=None, **kwargs):
            if _key is not mint_key:
                raise PermissionError(
                    "ApprovedOrder cannot be constructed directly. Only the risk "
                    "gate may approve an order — call RiskGate.approve(). If you "
                    "are trying to skip the gate, that is the thing this class "
                    "exists to prevent."
                )
            # frozen dataclass: assign through object.__setattr__
            field_names = (
                "order", "approved_at", "equity_at_approval",
                "approval_signature", "checks_passed",
            )
            values = dict(zip(field_names, args, strict=False))
            values.update(kwargs)
            for name in field_names:
                object.__setattr__(self, name, values[name])

        def __init_subclass__(cls, **kwargs):
            raise TypeError(
                "ApprovedOrder may not be subclassed — a subclass would be a "
                "second, ungated way to satisfy the broker's type check."
            )

        def __reduce__(self):
            raise TypeError(
                "ApprovedOrder is not picklable: unpickling constructs instances "
                "without the mint key, which would be a bypass."
            )

        def __copy__(self):
            raise TypeError("ApprovedOrder may not be copied")

        def __deepcopy__(self, memo):
            raise TypeError("ApprovedOrder may not be deep-copied")

    def _mint(
        order: Order,
        approved_at: pd.Timestamp,
        equity_at_approval: float,
        checks_passed: tuple[str, ...],
    ) -> ApprovedOrder:
        return ApprovedOrder(
            order,
            approved_at,
            equity_at_approval,
            _sign(order, equity_at_approval),
            checks_passed,
            _key=mint_key,
        )

    return ApprovedOrder, _mint


ApprovedOrder, _mint_approved = _build_approval_machinery()


class ForgedApprovalError(PermissionError):
    """An ApprovedOrder failed signature verification — it was not minted here."""


def verify_approval(approved: object) -> None:
    """Raise unless ``approved`` is a genuinely minted, unmodified approval.

    Brokers MUST call this before submitting. The type check alone is not
    enough: it proves shape, not provenance.
    """
    if not isinstance(approved, ApprovedOrder):
        raise ForgedApprovalError(
            f"expected an ApprovedOrder from the risk gate, got "
            f"{type(approved).__name__}. Raw orders may not reach a broker."
        )
    expected = _sign(approved.order, approved.equity_at_approval)
    if not hmac.compare_digest(expected, approved.approval_signature):
        raise ForgedApprovalError(
            "approval signature does not match the order it accompanies. Either "
            "this approval was fabricated without the risk gate, or the order "
            "was altered after approval. Refusing to submit."
        )
