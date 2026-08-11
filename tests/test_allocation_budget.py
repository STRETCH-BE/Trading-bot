"""The startup coherence check between the strategy's risk budget and the limits.

The failure mode this exists to prevent is silent: a bot that starts, looks
alive, and has every single order rejected forever because the allocation it
asks for is larger than the gate will ever approve. That must be a refusal to
start, not a per-cycle log line nobody reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trading_bot.backtest.config import BacktestConfig
from trading_bot.risk import RiskLimits
from trading_bot.risk.budget import (
    AllocationBudgetError,
    check_allocation_budget,
    describe_budget,
    drift_ceiling,
)

CONFIG_YAML = Path(__file__).resolve().parents[1] / "config.yaml"


def test_the_shipped_configuration_is_coherent():
    """The real config.yaml against the real RiskLimits — not a fixture.

    If this fails the bot cannot start, so it is the single most important
    assertion in this file.
    """
    config = BacktestConfig.load(CONFIG_YAML)
    limits = RiskLimits.load(CONFIG_YAML)
    check_allocation_budget(config, limits).raise_if_violated()  # must not raise


def test_allocation_above_the_position_cap_is_refused():
    config = BacktestConfig(strategy_max_allocation=0.50)
    limits = RiskLimits(max_position_pct=25.0, max_order_size_pct=100.0)

    check = check_allocation_budget(config, limits)
    assert not check.ok
    with pytest.raises(AllocationBudgetError) as exc:
        check.raise_if_violated()

    # both values must be named, so the message alone is enough to fix it
    message = str(exc.value)
    assert "strategy_max_allocation=0.5" in message
    assert "50.00% of equity" in message
    assert "max_position_pct=25.0%" in message


def test_allocation_above_the_order_size_cap_is_refused():
    """The order cap can bite even when the position cap does not."""
    config = BacktestConfig(strategy_max_allocation=0.40)
    limits = RiskLimits(max_position_pct=100.0, max_order_size_pct=30.0)

    check = check_allocation_budget(config, limits)
    assert not check.ok
    with pytest.raises(AllocationBudgetError, match="max_order_size_pct=30.0%"):
        check.raise_if_violated()


def test_both_violations_are_reported_together():
    """One restart per problem is one restart too many."""
    check = check_allocation_budget(
        BacktestConfig(strategy_max_allocation=1.0), RiskLimits()
    )
    assert len(check.messages) == 2
    joined = " ".join(check.messages)
    assert "max_position_pct" in joined and "max_order_size_pct" in joined


def test_allocation_exactly_at_the_cap_is_allowed():
    """25% allocation against a 25% cap is coherent — the gate approves it.

    Asserted because an off-by-one here (`>=` instead of `>`) would refuse the
    shipped configuration, which is exactly this case.
    """
    check = check_allocation_budget(
        BacktestConfig(strategy_max_allocation=0.25),
        RiskLimits(max_position_pct=25.0, max_order_size_pct=25.0),
    )
    assert check.ok
    assert check.messages == ()


def test_banner_marks_each_limit_ok_or_violated():
    ok = describe_budget(BacktestConfig(strategy_max_allocation=0.25), RiskLimits())
    assert "VIOLATED" not in ok
    assert "25.00% of equity" in ok

    bad = describe_budget(BacktestConfig(strategy_max_allocation=1.0), RiskLimits())
    assert bad.count("VIOLATED") == 2


# --- the drift gap: a budget is a cap at TRADE time, not a continuous one ----


def test_drift_ceiling_is_budget_plus_dead_band():
    config = BacktestConfig(strategy_max_allocation=0.25, min_rebalance_delta=0.05)
    assert drift_ceiling(config) == pytest.approx(30.0)


def test_shipped_config_discloses_that_drift_exceeds_the_position_cap():
    """The shipped 25% budget against a 25% cap leaves NO drift headroom.

    A rising market carries the position to 30% before the dead-band lets it
    trim. No order creates that exposure, so the gate never rejects it — which
    is precisely why the banner has to say so out loud.
    """
    config = BacktestConfig.load(CONFIG_YAML)
    limits = RiskLimits.load(CONFIG_YAML)

    assert drift_ceiling(config) > limits.max_position_pct
    banner = describe_budget(config, limits)
    assert "NOTE: between rebalances" in banner
    assert "30.00%" in banner

    # it is a disclosure, not a violation: the bot must still start
    check_allocation_budget(config, limits).raise_if_violated()


def test_no_note_when_the_budget_leaves_drift_headroom():
    """Give the budget room under the cap and the disclosure disappears."""
    config = BacktestConfig(strategy_max_allocation=0.15, min_rebalance_delta=0.05)
    banner = describe_budget(config, RiskLimits(max_position_pct=25.0))
    assert "NOTE" not in banner


# --- the guarantee that matters: the process refuses to start ----------------


def _write_config(tmp_path: Path, allocation: float) -> Path:
    """A copy of the shipped config with one value changed."""
    data = yaml.safe_load(CONFIG_YAML.read_text())
    data["backtest"]["strategy_max_allocation"] = allocation
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_run_main_refuses_to_start_on_an_incoherent_budget(tmp_path):
    from trading_bot.run import main

    path = _write_config(tmp_path, 1.0)  # 100% against the shipped 25%/30% caps
    with pytest.raises(AllocationBudgetError, match="refusing to start"):
        main([
            "--once", "--dry-run",
            "--config", str(path),
            "--state-dir", str(tmp_path / "state"),
            "--halt-file", str(tmp_path / "HALT"),
        ])

    # it failed BEFORE opening any state, not halfway through a cycle
    assert not (tmp_path / "state").exists(), "state was created before the refusal"


def test_run_main_prints_the_budget_before_deciding(tmp_path, capsys):
    """The banner must show the numbers even on the failing path — a refusal
    that does not say what it refused is a support ticket."""
    from trading_bot.run import main

    path = _write_config(tmp_path, 1.0)
    with pytest.raises(AllocationBudgetError):
        main([
            "--once", "--dry-run",
            "--config", str(path),
            "--state-dir", str(tmp_path / "state"),
            "--halt-file", str(tmp_path / "HALT"),
        ])

    out = capsys.readouterr().out
    assert "strategy risk budget" in out
    assert "VIOLATED" in out
