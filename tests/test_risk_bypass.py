"""Deliberate attempts to get an unapproved order past the gate.

Each test is a real attack, named for the technique, with the outcome
documented. Where an attack partially succeeds, the test says so explicitly
rather than asserting a comfortable falsehood — the point of this file is to
establish where the guarantee actually stops.

Threat model: a careless or mistaken developer, and a refactor that quietly
routes around the gate. NOT an adversary executing arbitrary code in-process
— against that, no pure-Python construct holds, and this file demonstrates
exactly that at the end.
"""

from __future__ import annotations

import copy
import pickle

import pandas as pd
import pytest

from trading_bot.execution import FillModel, Order
from trading_bot.risk import (
    AccountState,
    ApprovedOrder,
    ForgedApprovalError,
    RiskGate,
    RiskLimits,
    verify_approval,
)

TS = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")


def make_order(**kw) -> Order:
    base = dict(
        client_order_id="tb-attack", pair="XBTEUR", side="buy", units=0.001,
        order_type="market", limit_price=None, reason="attack", timestamp=TS,
    )
    base.update(kw)
    return Order(**base)


def make_state(**kw) -> AccountState:
    base = dict(
        equity=10_000.0, cash=10_000.0, positions={}, prices={"XBTEUR": 50_000.0},
        peak_equity=10_000.0, day_start_equity=10_000.0, latest_data_time=TS,
    )
    base.update(kw)
    return AccountState(**base)


@pytest.fixture
def gate(tmp_path) -> RiskGate:
    return RiskGate(
        RiskLimits(),
        FillModel(maker_fee_bps=16.0, taker_fee_bps=26.0, slippage_bps=5.0),
        halt_file=tmp_path / "HALT",
    )


def a_real_approval(gate: RiskGate) -> ApprovedOrder:
    decision = gate.approve(make_order(), make_state(), now=TS)
    assert decision.ok, decision.message
    return decision.approved


# --- a broker stand-in that behaves as Stage 6a requires ---------------------


class Broker:
    """Minimal stand-in: accepts ApprovedOrder only, and VERIFIES provenance."""

    def submit_order(self, approved: ApprovedOrder) -> str:
        verify_approval(approved)  # type check is not enough; prove it was minted
        return approved.order.client_order_id


# --- BYPASS 1: just construct one ------------------------------------------


def test_bypass_1_direct_construction_is_blocked():
    """The obvious attempt: build an ApprovedOrder yourself. BLOCKED."""
    with pytest.raises(PermissionError, match="cannot be constructed directly"):
        ApprovedOrder(
            order=make_order(),
            approved_at=TS,
            equity_at_approval=10_000.0,
            approval_signature="x" * 64,
            checks_passed=(),
        )


# --- BYPASS 2: skip __init__ entirely via object.__new__ --------------------


def test_bypass_2_object_new_produces_an_object_the_broker_rejects():
    """object.__new__ skips __init__, so the mint key never gets checked.

    OUTCOME: an ApprovedOrder-shaped instance IS created — the type check
    alone would pass it. The signature layer is what stops it at the broker.
    """
    forged = object.__new__(ApprovedOrder)
    object.__setattr__(forged, "order", make_order())
    object.__setattr__(forged, "approved_at", TS)
    object.__setattr__(forged, "equity_at_approval", 10_000.0)
    object.__setattr__(forged, "approval_signature", "0" * 64)
    object.__setattr__(forged, "checks_passed", ())

    assert isinstance(forged, ApprovedOrder)  # the attack DOES defeat the type check
    with pytest.raises(ForgedApprovalError, match="signature does not match"):
        Broker().submit_order(forged)


# --- BYPASS 3: swap the order inside a genuine approval ---------------------


def test_bypass_3_mutating_a_real_approval_is_blocked(gate):
    """Get one legitimately, then point it at a different order."""
    approved = a_real_approval(gate)
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        approved.order = make_order(units=999.0)  # type: ignore[misc]


def test_bypass_3b_forcing_the_swap_past_frozen_is_caught_by_signature(gate):
    """object.__setattr__ defeats frozen. The signature does not.

    OUTCOME: the field IS swapped — frozen is not a security boundary. The
    broker still refuses, because the HMAC covers the order's identity.
    """
    approved = a_real_approval(gate)
    object.__setattr__(approved, "order", make_order(units=999.0))

    assert approved.order.units == 999.0  # mutation succeeded
    with pytest.raises(ForgedApprovalError, match="altered after approval"):
        Broker().submit_order(approved)


# --- BYPASS 4: subclass it --------------------------------------------------


def test_bypass_4_subclassing_is_blocked():
    """A subclass would satisfy isinstance() with its own constructor."""
    with pytest.raises(TypeError, match="may not be subclassed"):

        class Sneaky(ApprovedOrder):  # noqa: D401
            def __init__(self, order):
                object.__setattr__(self, "order", order)


# --- BYPASS 5: serialise / copy round-trip ----------------------------------


def test_bypass_5_pickle_round_trip_is_blocked(gate):
    """Unpickling reconstructs without the mint key. Blocked at __reduce__."""
    approved = a_real_approval(gate)
    with pytest.raises(TypeError, match="not picklable"):
        pickle.dumps(approved)


def test_bypass_5b_copy_and_deepcopy_are_blocked(gate):
    approved = a_real_approval(gate)
    with pytest.raises(TypeError, match="may not be copied"):
        copy.copy(approved)
    with pytest.raises(TypeError, match="may not be deep-copied"):
        copy.deepcopy(approved)


# --- BYPASS 6: hand the broker a raw Order ----------------------------------


def test_bypass_6_raw_order_is_rejected_by_the_broker():
    """The ordinary mistake this whole stage exists to prevent."""
    with pytest.raises(ForgedApprovalError, match="Raw orders may not reach"):
        Broker().submit_order(make_order())  # type: ignore[arg-type]


def test_bypass_6b_duck_typed_lookalike_is_rejected():
    """A different class with identical attributes must not pass."""

    class LooksApproved:
        order = make_order()
        approved_at = TS
        equity_at_approval = 10_000.0
        approval_signature = "0" * 64
        checks_passed = ()

    with pytest.raises(ForgedApprovalError, match="got LooksApproved"):
        Broker().submit_order(LooksApproved())  # type: ignore[arg-type]


# --- BYPASS 7: steal the mint key out of the closure ------------------------


def test_bypass_7_stealing_the_closure_key_SUCCEEDS_and_is_documented():
    """The honest one: introspection defeats the mint key.

    ``_mint_approved.__closure__`` exposes the sentinel, so an attacker with
    in-process code execution can construct an ApprovedOrder. The signature
    layer still holds here ONLY because minting also requires the signing
    secret — which the same attacker could read from the module.

    This test asserts what genuinely happens. It is the boundary of the
    guarantee, and it is recorded rather than hidden: the gate protects
    against mistakes and refactors, not against arbitrary in-process code.
    """
    from trading_bot.risk import approval as approval_mod

    stolen_key = None
    for cell in approval_mod._mint_approved.__closure__ or ():
        contents = cell.cell_contents
        if type(contents) is object:
            stolen_key = contents
    assert stolen_key is not None, "closure layout changed — revisit this test"

    # Construction now succeeds: the mint-key layer IS defeated.
    forged = ApprovedOrder(
        make_order(), TS, 10_000.0, "0" * 64, (), _key=stolen_key
    )
    assert isinstance(forged, ApprovedOrder)

    # But the broker still refuses, because the signature is not valid.
    with pytest.raises(ForgedApprovalError):
        Broker().submit_order(forged)


def test_bypass_7b_stealing_the_signing_secret_defeats_everything():
    """Full disclosure: with the secret, a valid approval can be forged.

    Nothing in pure Python prevents this. Recorded so the guarantee is never
    overstated in review: the defence is against accident, not against an
    attacker already running code inside the process.
    """
    from trading_bot.risk import approval as approval_mod

    order = make_order()
    forged_signature = approval_mod._sign(order, 10_000.0)
    forged = approval_mod._mint_approved(order, TS, 10_000.0, ("forged",))
    object.__setattr__(forged, "approval_signature", forged_signature)

    assert Broker().submit_order(forged) == order.client_order_id  # it works


# --- the guarantee that DOES hold -------------------------------------------


def test_only_the_gate_produces_a_submittable_approval(gate):
    """Summary: through every non-introspective route, only approve() works."""
    assert Broker().submit_order(a_real_approval(gate)) == "tb-attack"


def test_a_rejected_decision_carries_no_approval(gate, tmp_path):
    gate.engage_halt("test")
    decision = gate.approve(make_order(), make_state(), now=TS)
    assert decision.rejected
    assert decision.approved is None
    with pytest.raises(ForgedApprovalError):
        Broker().submit_order(decision.approved)  # type: ignore[arg-type]


def test_approval_records_which_checks_it_passed(gate):
    approved = a_real_approval(gate)
    assert "halt_file" in approved.checks_passed
    assert "no_leverage_or_short" in approved.checks_passed
    assert "max_drawdown" in approved.checks_passed
    assert len(approved.checks_passed) == 12
