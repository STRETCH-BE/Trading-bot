"""The Stage 5a seam: Order, deterministic ids, FillModel, translation."""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import pandas as pd
import pytest

from trading_bot.backtest import BacktestConfig
from trading_bot.data.schema import UnknownPairError
from trading_bot.execution import (
    FillModel,
    Order,
    PortfolioState,
    make_client_order_id,
    orders_for_target,
)

TS = pd.Timestamp("2024-03-01 00:00:00", tz="UTC")


def model(**kw) -> FillModel:
    base = dict(maker_fee_bps=16.0, taker_fee_bps=26.0, slippage_bps=5.0)
    base.update(kw)
    return FillModel(**base)


# --- deterministic client order ids ------------------------------------------


def test_same_decision_yields_same_id():
    a = make_client_order_id("voltrend", "XBTEUR", TS, "buy:1.0")
    b = make_client_order_id("voltrend", "XBTEUR", TS, "buy:1.0")
    assert a == b
    assert a.startswith("tb-")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": "donchian"},
        {"pair": "ETHEUR"},
        {"signal_timestamp": TS + pd.Timedelta(days=1)},
        {"intent": "sell:0.0"},
    ],
)
def test_any_input_change_yields_a_different_id(kwargs):
    base = dict(strategy="voltrend", pair="XBTEUR", signal_timestamp=TS, intent="buy:1.0")
    assert make_client_order_id(**base) != make_client_order_id(**{**base, **kwargs})


def test_id_is_stable_across_process_restarts():
    """THE idempotency prerequisite.

    Python's builtin hash() is salted per process, so an id derived from it
    would change on restart — precisely when a crashed process needs to ask
    the exchange "did my order land?". This spawns fresh interpreters with
    hostile PYTHONHASHSEED values and requires identical output.
    """
    code = (
        "import pandas as pd;"
        "from trading_bot.execution import make_client_order_id;"
        "print(make_client_order_id('voltrend','XBTEUR',"
        "pd.Timestamp('2024-03-01',tz='UTC'),'buy:1.0'))"
    )
    seen = set()
    for seed in ("0", "1", "12345", "random"):
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, check=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": seed},
        )
        seen.add(out.stdout.strip())
    assert len(seen) == 1, f"id varies across processes: {seen}"
    assert seen.pop() == make_client_order_id("voltrend", "XBTEUR", TS, "buy:1.0")


def test_timestamp_representation_is_normalised():
    """An equivalent instant in another tz must map to the same id."""
    utc = pd.Timestamp("2024-03-01 00:00:00", tz="UTC")
    other = utc.tz_convert("Europe/Brussels")
    naive = pd.Timestamp("2024-03-01 00:00:00")
    a = make_client_order_id("s", "XBTEUR", utc, "i")
    assert make_client_order_id("s", "XBTEUR", other, "i") == a
    assert make_client_order_id("s", "XBTEUR", naive, "i") == a


# --- Order contract ----------------------------------------------------------


def _order(**kw) -> Order:
    base = dict(
        client_order_id="tb-x", pair="XBTEUR", side="buy", units=1.0,
        order_type="market", limit_price=None, reason="test", timestamp=TS,
    )
    base.update(kw)
    return Order(**base)


def test_order_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _order().units = 5.0  # type: ignore[misc]


def test_negative_units_rejected_as_implicit_short():
    with pytest.raises(ValueError, match="implicit short"):
        _order(units=-1.0)


def test_zero_units_rejected():
    with pytest.raises(ValueError, match="positive"):
        _order(units=0.0)


def test_limit_order_requires_a_price():
    with pytest.raises(ValueError, match="limit_price"):
        _order(order_type="limit", limit_price=None)


def test_bad_side_rejected():
    with pytest.raises(ValueError, match="side"):
        _order(side="short")


# --- FillModel ---------------------------------------------------------------


def test_slippage_always_moves_against_the_trader():
    fm = model()
    assert fm.fill_price("buy", 100.0) > 100.0
    assert fm.fill_price("sell", 100.0) < 100.0


def test_fee_uses_the_configured_liquidity():
    fm = model()
    assert fm.fee(1000.0, "taker") == pytest.approx(1000.0 * 0.0026)
    assert fm.fee(1000.0, "maker") == pytest.approx(1000.0 * 0.0016)
    assert fm.fee(1000.0) == fm.fee(1000.0, "taker")  # default_liquidity


def test_minimums_check_ordermin_before_costmin():
    """Order matters: a costmin failure must mean 'big enough, too cheap'."""
    fm = model()
    tiny = fm.clears_minimums("XBTEUR", 0.00001, 0.5)
    assert not tiny.ok and tiny.failed_limit == "ordermin"
    cheap = fm.clears_minimums("XBTEUR", 0.001, 0.5)  # clears ordermin, fails costmin
    assert not cheap.ok and cheap.failed_limit == "costmin"
    good = fm.clears_minimums("XBTEUR", 0.001, 50.0)
    assert good.ok


def test_limits_require_a_pair_or_an_explicit_override():
    with pytest.raises(ValueError, match="needs a `pair`"):
        model().limits_for(None)
    assert model(min_order_units=0.0).limits_for(None) == (0.0, 0.0)


def test_unknown_pair_raises_rather_than_defaulting():
    with pytest.raises(UnknownPairError):
        model().limits_for("DOGEEUR")


def test_from_config_matches_the_backtest_config():
    cfg = BacktestConfig(maker_fee_bps=25.0, taker_fee_bps=40.0, slippage_bps=7.0)
    fm = FillModel.from_config(cfg)
    assert fm.fee_rate("taker") == cfg.fee_rate
    assert fm.slippage_rate == cfg.slippage_rate


# --- rounding (off by default, so equivalence holds) -------------------------


def test_rounding_is_disabled_by_default():
    fm = model()
    assert fm.round_units("XBTEUR", 0.123456789123) == 0.123456789123
    assert fm.round_price("XBTEUR", 74500.123456) == 74500.123456


def test_rounding_when_enabled_truncates_down_never_up():
    fm = model(apply_rounding=True)
    rounded = fm.round_units("XBTEUR", 0.1234567891)  # lot_decimals=8
    assert rounded <= 0.1234567891
    assert rounded == pytest.approx(0.12345678, abs=1e-12)


def test_price_rounding_uses_pair_tick():
    fm = model(apply_rounding=True)
    assert fm.round_price("XBTEUR", 74500.16) == pytest.approx(74500.2)  # 1 dp
    assert fm.round_price("ETHEUR", 2526.456) == pytest.approx(2526.46)  # 2 dp


# --- translation -------------------------------------------------------------


def test_translation_returns_orders_and_does_not_execute():
    out = orders_for_target(
        PortfolioState(cash=10_000.0, units=0.0), 1.0, 100.0, "XBTEUR", model(),
        timestamp=TS,
    )
    assert len(out.orders) == 1
    order = out.orders[0]
    assert order.side == "buy"
    assert order.units > 0
    assert order.pair == "XBTEUR"
    assert "rebalance" in order.reason
    # translation reports sizing but performs no state change
    assert out.fill_price == pytest.approx(100.0 * 1.0005)


def test_deadband_suppresses_a_small_move():
    state = PortfolioState(cash=5_000.0, units=50.0)  # 50% at price 100
    out = orders_for_target(
        state, 0.52, 100.0, "XBTEUR", model(), min_rebalance_delta=0.05, timestamp=TS
    )
    assert out.orders == []
    assert out.suppressed_by_deadband
    assert "dead-band" in out.reason


def test_full_exit_is_exempt_from_the_deadband():
    state = PortfolioState(cash=9_600.0, units=4.0)  # 4% at price 100
    out = orders_for_target(
        state, 0.0, 100.0, "XBTEUR", model(), min_rebalance_delta=0.05, timestamp=TS
    )
    assert len(out.orders) == 1
    assert out.orders[0].side == "sell"
    assert not out.suppressed_by_deadband


def test_translation_reports_which_minimum_failed():
    state = PortfolioState(cash=100.0, units=0.0)
    out = orders_for_target(state, 0.005, 100.0, "XBTEUR", model(), timestamp=TS)
    assert out.orders == []
    assert out.skipped_costmin and not out.skipped_ordermin
    assert out.skipped_minimum


def test_no_order_when_already_at_target():
    state = PortfolioState(cash=0.0, units=100.0)
    out = orders_for_target(state, 1.0, 100.0, "XBTEUR", model(), timestamp=TS)
    assert out.orders == []
    assert "already at target" in out.reason


def test_limit_orders_carry_a_rounded_price():
    out = orders_for_target(
        PortfolioState(cash=10_000.0, units=0.0), 1.0, 74500.16, "XBTEUR",
        model(apply_rounding=True), timestamp=TS, order_type="limit",
    )
    order = out.orders[0]
    assert order.order_type == "limit"
    assert order.limit_price == pytest.approx(round(74500.16 * 1.0005, 1))


def test_translation_ids_are_deterministic_for_the_same_decision():
    args = (PortfolioState(cash=10_000.0, units=0.0), 1.0, 100.0, "XBTEUR", model())
    kw = dict(timestamp=TS, signal_timestamp=TS, strategy="voltrend")
    a = orders_for_target(*args, **kw).orders[0]
    b = orders_for_target(*args, **kw).orders[0]
    assert a.client_order_id == b.client_order_id


# --- the anti-duplication guard ---------------------------------------------


def test_fee_and_slippage_arithmetic_exists_in_exactly_one_place():
    """Audit finding 3: nothing outside FillModel may compute costs.

    Greps the source for the raw expressions. The end-of-run liquidation in
    the engine is the one known exception and is asserted explicitly so that
    it cannot grow silently.
    """
    import pathlib

    import trading_bot

    root = pathlib.Path(trading_bot.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("execution/"):
            continue
        text = path.read_text()
        for token in ("/ 10_000.0", "/ 10000.0"):
            if token in text:
                offenders.append(f"{rel}: bps conversion ({token})")
    assert not offenders, (
        "cost arithmetic found outside FillModel — this is exactly the "
        f"divergence audit finding 3 warned about: {offenders}"
    )
