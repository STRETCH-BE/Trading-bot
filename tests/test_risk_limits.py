"""One test per risk limit: it blocks what it should, allows what it should."""

from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.execution import FillModel, Order
from trading_bot.risk import AccountState, RiskGate, RiskLimits

TS = pd.Timestamp("2026-06-01 12:00:00", tz="UTC")
PRICE = 50_000.0
FILL_MODEL = FillModel(maker_fee_bps=16.0, taker_fee_bps=26.0, slippage_bps=5.0)


def order(**kw) -> Order:
    base = dict(
        client_order_id="tb-t", pair="XBTEUR", side="buy", units=0.001,
        order_type="market", limit_price=None, reason="test", timestamp=TS,
    )
    base.update(kw)
    return Order(**base)


def state(**kw) -> AccountState:
    base = dict(
        equity=10_000.0, cash=10_000.0, positions={}, prices={"XBTEUR": PRICE},
        peak_equity=10_000.0, day_start_equity=10_000.0, latest_data_time=TS,
    )
    base.update(kw)
    return AccountState(**base)


def gate_with(tmp_path, **limit_kw) -> RiskGate:
    return RiskGate(
        RiskLimits(**limit_kw), FILL_MODEL, halt_file=tmp_path / "HALT"
    )


@pytest.fixture
def gate(tmp_path) -> RiskGate:
    return gate_with(tmp_path)


# --- kill switch -------------------------------------------------------------


def test_halt_file_blocks_within_a_single_call(gate, tmp_path):
    """`touch HALT` must take effect on the very next approval — never cached."""
    assert gate.approve(order(), state(), now=TS).ok

    (tmp_path / "HALT").write_text("stopped by hand")

    decision = gate.approve(order(), state(), now=TS)
    assert decision.rejected
    assert decision.limit == "kill_switch"
    assert decision.approved is None


def test_removing_the_halt_file_restores_approval(gate, tmp_path):
    (tmp_path / "HALT").write_text("x")
    assert gate.approve(order(), state(), now=TS).rejected
    (tmp_path / "HALT").unlink()
    assert gate.approve(order(), state(), now=TS).ok


# --- hard rejects ------------------------------------------------------------


def test_unknown_pair_is_hard_rejected_and_halts(gate):
    d = gate.approve(order(pair="DOGEEUR"), state(prices={"DOGEEUR": 1.0}), now=TS)
    assert d.rejected and d.limit == "unknown_pair" and d.halted
    assert gate.is_halted()


def test_buy_exceeding_cash_is_rejected_as_margin(gate):
    d = gate.approve(order(units=1.0), state(cash=100.0), now=TS)
    assert d.rejected and d.limit == "short_or_leverage" and d.halted
    assert "MARGIN" in d.message


def test_sell_exceeding_holdings_is_rejected_as_a_short(gate):
    d = gate.approve(
        order(side="sell", units=0.5), state(positions={"XBTEUR": 0.1}), now=TS
    )
    assert d.rejected and d.limit == "short_or_leverage" and d.halted
    assert "SHORT" in d.message


def test_selling_exactly_what_is_held_is_allowed(gate):
    """A full exit that is small relative to equity passes."""
    d = gate.approve(
        order(side="sell", units=0.02), state(positions={"XBTEUR": 0.02}), now=TS
    )
    assert d.ok


def test_FINDING_large_full_exit_is_blocked_by_the_fat_finger_guard(gate):
    """DOCUMENTED INTERACTION, not an endorsement.

    max_order_size_pct is specified as an absolute cap on any single order.
    Applied to sells, it means a position worth more than 30% of equity
    CANNOT BE FULLY EXITED through the gate — the guard blocks risk
    reduction. This test pins the behaviour as currently specified so the
    consequence is visible; see the Stage 5b report for the recommended fix
    (exempt sells, whose risk is bounded by the holding).
    """
    d = gate.approve(
        order(side="sell", units=0.1), state(positions={"XBTEUR": 0.1}), now=TS
    )
    assert d.rejected
    assert d.limit == "max_order_size_pct"
    assert d.actual == pytest.approx(50.0)


def test_FINDING_order_size_guard_is_unreachable_for_buys(gate):
    """The 30% guard never binds on a buy: max_position_pct (25%) is tighter.

    A buy large enough to trip the 30% guard has already exceeded the 25%
    position cap, and a buy under 30% still trips the position cap above 25%.
    So on the default limits the fat-finger guard fires ONLY on sells.
    """
    big = gate.approve(order(units=0.07), state(), now=TS)  # 35%
    assert big.limit == "max_order_size_pct"
    mid = gate.approve(order(units=0.058), state(), now=TS)  # 29%
    assert mid.limit == "max_position_pct"  # the tighter cap catches it first


# --- stale data --------------------------------------------------------------


def test_stale_data_is_rejected(gate):
    d = gate.approve(order(), state(latest_data_time=TS - pd.Timedelta(seconds=400)), now=TS)
    assert d.rejected and d.limit == "stale_data"
    assert d.actual == pytest.approx(400.0)


def test_fresh_data_is_allowed(gate):
    d = gate.approve(order(), state(latest_data_time=TS - pd.Timedelta(seconds=60)), now=TS)
    assert d.ok


def test_data_from_the_future_is_rejected_as_clock_skew(gate):
    """Audit finding 5's backstop: a wrong clock must be loud, not silent."""
    d = gate.approve(order(), state(latest_data_time=TS + pd.Timedelta(seconds=120)), now=TS)
    assert d.rejected and d.limit == "stale_data"
    assert "FUTURE" in d.message


def test_missing_data_timestamp_is_rejected(gate):
    d = gate.approve(order(), state(latest_data_time=None), now=TS)
    assert d.rejected and d.limit == "stale_data"


# --- exchange minimums (delegated to FillModel) -----------------------------


def test_below_ordermin_is_rejected(gate):
    d = gate.approve(order(units=0.00001), state(), now=TS)
    assert d.rejected and d.limit == "ordermin"


def test_below_costmin_is_rejected(gate):
    """Clears ordermin (0.0001) but the notional is under 1 EUR."""
    d = gate.approve(order(units=0.00015), state(prices={"XBTEUR": 5.0}), now=TS)
    assert d.rejected and d.limit == "costmin"


# --- max_order_size_pct ------------------------------------------------------


def test_oversized_order_is_rejected(gate):
    # 0.07 BTC @ 50k = 3,500 = 35% of 10,000 equity, above the 30% guard
    d = gate.approve(order(units=0.07), state(), now=TS)
    assert d.rejected and d.limit == "max_order_size_pct"
    assert d.actual == pytest.approx(35.0)
    assert d.limit_value == 30.0


def test_order_just_under_both_size_caps_is_allowed(gate):
    d = gate.approve(order(units=0.048), state(), now=TS)  # 24% order and position
    assert d.ok


# --- max_position_pct --------------------------------------------------------


def test_position_limit_blocks_a_top_up_past_the_cap(gate):
    """Order itself is small (6%), but it would take the position to 26%."""
    d = gate.approve(
        order(units=0.012), state(positions={"XBTEUR": 0.04}), now=TS
    )
    assert d.rejected and d.limit == "max_position_pct"
    assert d.actual == pytest.approx(26.0)


def test_position_within_the_cap_is_allowed(gate):
    d = gate.approve(order(units=0.008), state(positions={"XBTEUR": 0.04}), now=TS)
    assert d.ok  # 24%


# --- max_total_exposure_pct --------------------------------------------------


def test_total_exposure_limit_blocks(tmp_path):
    """Isolated with a relaxed position cap — see the note in the report about
    this limit being dormant under the default 25%/2-position combination."""
    gate = gate_with(tmp_path, max_position_pct=100.0, max_concurrent_positions=5)
    d = gate.approve(
        order(units=0.05),  # +25% on top of 60% held
        state(positions={"XBTEUR": 0.12}),
        now=TS,
    )
    assert d.rejected and d.limit == "max_total_exposure_pct"
    assert d.actual == pytest.approx(85.0)


def test_total_exposure_within_the_cap_is_allowed(tmp_path):
    gate = gate_with(tmp_path, max_position_pct=100.0, max_concurrent_positions=5)
    d = gate.approve(order(units=0.02), state(positions={"XBTEUR": 0.12}), now=TS)
    assert d.ok  # 70%


# --- max_concurrent_positions ------------------------------------------------


def test_third_concurrent_position_is_blocked(tmp_path):
    gate = gate_with(tmp_path, max_position_pct=100.0, max_total_exposure_pct=100.0)
    d = gate.approve(
        order(pair="XBTEUR", units=0.01),
        state(
            positions={"ETHEUR": 1.0, "XRPEUR": 100.0},
            prices={"XBTEUR": PRICE, "ETHEUR": 2_000.0, "XRPEUR": 1.0},
        ),
        now=TS,
    )
    assert d.rejected and d.limit == "max_concurrent_positions"
    assert d.actual == 3


def test_adding_to_an_existing_position_does_not_count_as_a_new_one(tmp_path):
    gate = gate_with(tmp_path, max_position_pct=100.0, max_total_exposure_pct=100.0)
    d = gate.approve(
        order(pair="ETHEUR", units=0.5),
        state(
            positions={"ETHEUR": 1.0, "XBTEUR": 0.01},
            prices={"XBTEUR": PRICE, "ETHEUR": 2_000.0},
        ),
        now=TS,
    )
    assert d.ok


# --- daily_loss_limit_pct ----------------------------------------------------


def test_daily_loss_breach_rejects_and_halts(gate):
    d = gate.approve(order(), state(equity=9_400.0, day_start_equity=10_000.0), now=TS)
    assert d.rejected and d.limit == "daily_loss_limit_pct" and d.halted
    assert d.actual == pytest.approx(6.0)
    assert gate.is_halted()


def test_daily_loss_does_not_auto_resume_on_the_next_cycle(gate):
    """The mandated behaviour: a breach requires a MANUAL restart.

    Even with a fully recovered account on the following cycle, approval stays
    blocked, because the breach wrote the HALT file and only a human removes it.
    """
    breach = gate.approve(order(), state(equity=9_000.0, day_start_equity=10_000.0), now=TS)
    assert breach.rejected and breach.halted

    healthy = state(equity=10_500.0, day_start_equity=10_000.0, peak_equity=10_500.0)
    next_cycle = gate.approve(order(), healthy, now=TS + pd.Timedelta(days=1))
    assert next_cycle.rejected, "auto-resumed after a daily-loss halt"
    assert next_cycle.limit == "kill_switch"


def test_halt_survives_a_new_gate_instance(gate, tmp_path):
    """Restarting the process must not clear a halt — the file persists."""
    gate.approve(order(), state(equity=9_000.0, day_start_equity=10_000.0), now=TS)
    fresh = gate_with(tmp_path)  # as if the process had restarted
    assert fresh.approve(order(), state(), now=TS).rejected


def test_loss_within_the_daily_limit_is_allowed(gate):
    d = gate.approve(order(), state(equity=9_600.0, day_start_equity=10_000.0), now=TS)
    assert d.ok  # 4%


# --- max_drawdown_pct --------------------------------------------------------


def test_drawdown_breach_rejects_and_halts(gate):
    d = gate.approve(
        order(),
        state(equity=7_500.0, peak_equity=10_000.0, day_start_equity=7_500.0),
        now=TS,
    )
    assert d.rejected and d.limit == "max_drawdown_pct" and d.halted
    assert d.actual == pytest.approx(25.0)


def test_drawdown_within_the_limit_is_allowed(gate):
    d = gate.approve(
        order(),
        state(equity=8_500.0, peak_equity=10_000.0, day_start_equity=8_500.0),
        now=TS,
    )
    assert d.ok  # 15%


# --- max_orders_per_hour -----------------------------------------------------


def test_order_rate_circuit_breaker_trips(gate):
    recent = [TS - pd.Timedelta(minutes=m) for m in range(10)]
    d = gate.approve(order(), state(recent_order_times=recent), now=TS)
    assert d.rejected and d.limit == "max_orders_per_hour"
    assert d.actual == 10


def test_orders_older_than_an_hour_do_not_count(gate):
    old = [TS - pd.Timedelta(hours=2, minutes=m) for m in range(20)]
    assert gate.approve(order(), state(recent_order_times=old), now=TS).ok


def test_rate_under_the_breaker_is_allowed(gate):
    recent = [TS - pd.Timedelta(minutes=m) for m in range(9)]
    assert gate.approve(order(), state(recent_order_times=recent), now=TS).ok


# --- rejection logging -------------------------------------------------------


def test_every_rejection_logs_limit_actual_and_state(gate, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        gate.approve(order(units=0.07), state(), now=TS)
    record = caplog.text
    assert "RISK REJECT" in record
    assert "max_order_size_pct" in record
    assert "equity=10000.00" in record
    assert "limit_value=30.0" in record
    assert "tb-t" in record  # the full order


# --- limits config -----------------------------------------------------------


def test_limits_are_frozen():
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        RiskLimits().max_position_pct = 99.0  # type: ignore[misc]


def test_repo_config_yaml_matches_declared_defaults():
    assert RiskLimits.load("config.yaml") == RiskLimits()


def test_unknown_limit_key_is_rejected():
    with pytest.raises(ValueError, match="unknown risk limits"):
        RiskLimits.from_dict({"max_position_pct": 10.0, "typo_limit": 1.0})


def test_nonsense_limits_rejected():
    with pytest.raises(ValueError, match="max_position_pct"):
        RiskLimits(max_position_pct=0.0)
