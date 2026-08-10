"""The holdout boundary: blocked by default, unlockable exactly once."""

from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.backtest import BacktestConfig, backtest
from trading_bot.backtest.holdout import HoldoutUnlock, HoldoutViolation

from .synthetic import make_candles

CONFIG = BacktestConfig(min_order_units=0.0)  # holdout_start default: 2026-01-01


def _candles(start: str, n: int) -> pd.DataFrame:
    return make_candles(
        [(100.0, 101.0, 99.0, 100.0, 5.0)] * n, start=pd.Timestamp(start, tz="UTC")
    )


def _flat(c: pd.DataFrame) -> pd.Series:
    return pd.Series([0.0] * len(c))


def test_candles_before_holdout_run_normally():
    result = backtest(_candles("2025-12-01", 20), _flat, CONFIG)
    assert result.final_equity == CONFIG.starting_capital


def test_candles_crossing_holdout_raise():
    with pytest.raises(HoldoutViolation, match="holdout"):
        backtest(_candles("2025-12-25", 20), _flat, CONFIG)


def test_candle_exactly_on_the_boundary_raises():
    # last candle opens exactly at holdout_start — that IS holdout data
    with pytest.raises(HoldoutViolation):
        backtest(_candles("2025-12-28", 5), _flat, CONFIG)


def test_last_candle_day_before_boundary_is_allowed():
    result = backtest(_candles("2025-12-27", 5), _flat, CONFIG)  # ends 2025-12-31
    assert result.final_equity == CONFIG.starting_capital


def test_unlock_allows_exactly_one_run(capsys):
    unlock = HoldoutUnlock()
    holdout_candles = _candles("2026-01-05", 10)

    result = backtest(holdout_candles, _flat, CONFIG, holdout_unlock=unlock)
    assert result.final_equity == CONFIG.starting_capital
    assert unlock.used
    assert "HOLDOUT UNLOCKED" in capsys.readouterr().err  # loud, on stderr

    # the same token must not authorise a second run
    with pytest.raises(HoldoutViolation, match="already spent"):
        backtest(holdout_candles, _flat, CONFIG, holdout_unlock=unlock)


def test_unlock_is_not_consumed_by_a_non_holdout_run():
    unlock = HoldoutUnlock()
    backtest(_candles("2025-01-01", 10), _flat, CONFIG, holdout_unlock=unlock)
    assert not unlock.used  # nothing to unlock, nothing spent


def test_no_default_unlock_exists():
    """The unlock is a parameter someone must mint explicitly — never config."""
    import inspect

    sig = inspect.signature(backtest)
    assert sig.parameters["holdout_unlock"].default is None
    assert not hasattr(BacktestConfig(), "holdout_unlock")


def test_holdout_boundary_is_configurable():
    config = BacktestConfig(min_order_units=0.0, holdout_start="2025-06-01")
    with pytest.raises(HoldoutViolation):
        backtest(_candles("2025-05-25", 20), _flat, config)


def test_unparseable_holdout_start_fails_at_config_time():
    with pytest.raises((ValueError, TypeError)):
        BacktestConfig(holdout_start="not-a-date")


def test_backtest_config_yaml_matches_declared_defaults():
    """config.yaml must not silently drift from the dataclass defaults."""
    assert BacktestConfig.load("config.yaml") == BacktestConfig()
