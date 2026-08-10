"""Volatility-targeted trend: contract, warmup, sizing behaviour, schedule.

Causality is covered for every registered strategy by tests/test_causality.py;
this file covers the behaviour specific to vol-targeted sizing.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest import BacktestConfig, backtest
from trading_bot.data import schema
from trading_bot.strategies.voltrend import (
    VolTrendParams,
    VolTrendSpan,
    voltrend,
    voltrend_schedule,
)

from .synthetic import flat_candles, from_closes

DEFAULTS = VolTrendParams()


def _trending(n: int = 400, drift: float = 0.004) -> pd.DataFrame:
    return from_closes([100.0 * (1 + drift) ** i for i in range(n)])


def _noisy(n: int = 400, vol: float = 0.02, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return from_closes(list(100 * np.cumprod(1 + rng.normal(0.001, vol, n))))


# --- output contract ---------------------------------------------------------


def test_signal_is_within_zero_and_one():
    sig = voltrend(_noisy(500), DEFAULTS)
    assert (sig >= 0.0).all() and (sig <= 1.0).all()
    assert not sig.isna().any()


def test_warmup_is_flat_and_never_nan():
    sig = voltrend(_trending(300), DEFAULTS)
    assert (sig.iloc[: DEFAULTS.warmup] == 0.0).all()
    assert not sig.isna().any()


def test_warmup_covers_the_volatility_window_plus_the_differencing_candle():
    params = VolTrendParams(trend_lookback=10, vol_lookback=60)
    assert params.warmup == 61
    sig = voltrend(_trending(200), params)
    assert (sig.iloc[:61] == 0.0).all()


def test_series_shorter_than_warmup_is_all_flat():
    assert (voltrend(_trending(30), DEFAULTS) == 0.0).all()


def test_signal_aligned_to_candle_index():
    candles = _trending(200)
    candles.index = range(500, 500 + len(candles))
    assert list(voltrend(candles, DEFAULTS).index) == list(candles.index)


def test_does_not_mutate_input():
    candles = _noisy(200)
    before = candles.copy(deep=True)
    voltrend(candles, DEFAULTS)
    pd.testing.assert_frame_equal(candles, before)


def test_non_monotonic_candles_rejected():
    candles = _trending(100).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="monotonically increasing"):
        voltrend(candles, DEFAULTS)


# --- the trend leg -----------------------------------------------------------


def test_downtrend_is_flat():
    """Below the trend filter the position is zero regardless of volatility."""
    sig = voltrend(from_closes([300.0 * 0.99**i for i in range(300)]), DEFAULTS)
    assert (sig == 0.0).all()


def test_sustained_uptrend_takes_a_position():
    sig = voltrend(_trending(400), DEFAULTS)
    assert sig.iloc[-1] > 0.0
    assert (sig.iloc[DEFAULTS.warmup :] > 0).mean() > 0.9


def test_constant_price_never_takes_a_position():
    """close > SMA is false when they are equal, so a flat tape stays flat."""
    assert (voltrend(flat_candles(300, price=100.0), DEFAULTS) == 0.0).all()


# --- the volatility leg ------------------------------------------------------


def test_higher_volatility_gives_a_smaller_position():
    """The core mechanism: same trend, more vol, less exposure."""
    calm = voltrend(_noisy(500, vol=0.01, seed=3), DEFAULTS)
    wild = voltrend(_noisy(500, vol=0.06, seed=3), DEFAULTS)
    calm_size = calm[calm > 0].mean()
    wild_size = wild[wild > 0].mean()
    assert wild_size < calm_size


def test_position_is_capped_at_one_in_very_low_volatility():
    """A near-zero-vol uptrend would imply huge leverage; it must clip to 1.0."""
    sig = voltrend(_trending(300, drift=0.0005), DEFAULTS)
    assert sig.max() == pytest.approx(1.0)
    assert (sig <= 1.0).all()


def test_zero_variance_window_does_not_produce_inf_or_nan():
    """A perfectly flat stretch makes realized vol zero — must not blow up."""
    closes = [100.0] * 150 + [100.0 * 1.01**i for i in range(1, 151)]
    sig = voltrend(from_closes(closes), DEFAULTS)
    assert np.isfinite(sig.to_numpy()).all()
    assert (sig <= 1.0).all()


def test_higher_target_vol_gives_a_larger_or_equal_position():
    candles = _noisy(500, vol=0.05, seed=9)
    low = voltrend(candles, VolTrendParams(target_vol=0.40))
    high = voltrend(candles, VolTrendParams(target_vol=0.60))
    assert (high.to_numpy() >= low.to_numpy() - 1e-12).all()
    assert high.sum() > low.sum()


def test_sizing_matches_the_formula_on_a_known_bar():
    candles = _noisy(300, vol=0.03, seed=11)
    params = VolTrendParams(trend_lookback=50, vol_lookback=20, target_vol=0.40)
    sig = voltrend(candles, params)

    close = candles[schema.CLOSE].astype(float)
    log_ret = np.log(close / close.shift(1))
    realized = log_ret.rolling(20).std(ddof=1) * np.sqrt(365.0)
    sma = close.rolling(50).mean()

    i = 250
    expected = min(0.40 / realized.iloc[i], 1.0) if close.iloc[i] > sma.iloc[i] else 0.0
    assert sig.iloc[i] == pytest.approx(expected)


# --- params ------------------------------------------------------------------


def test_params_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULTS.target_vol = 9.9  # type: ignore[misc]


def test_leverage_is_rejected_at_construction():
    with pytest.raises(ValueError, match="leverage"):
        VolTrendParams(max_position=1.5)


def test_invalid_params_rejected():
    with pytest.raises(ValueError, match="vol_lookback"):
        VolTrendParams(vol_lookback=1)
    with pytest.raises(ValueError, match="target_vol"):
        VolTrendParams(target_vol=0.0)


def test_params_load_from_config_yaml(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("strategies:\n  voltrend:\n    trend_lookback: 200\n")
    params = VolTrendParams.load(cfg)
    assert params.trend_lookback == 200
    assert params.vol_lookback == 20  # default preserved


def test_repo_config_yaml_matches_declared_defaults():
    assert VolTrendParams.load("config.yaml") == VolTrendParams()


# --- schedule ----------------------------------------------------------------


def test_single_span_schedule_equals_plain_voltrend():
    candles = _noisy(400, seed=17)
    for params in (
        VolTrendParams(),
        VolTrendParams(trend_lookback=200, vol_lookback=60, target_vol=0.6),
    ):
        np.testing.assert_array_equal(
            voltrend(candles, params).to_numpy(),
            voltrend_schedule(candles, [VolTrendSpan(0, params)]).to_numpy(),
        )


def test_schedule_swaps_parameters_at_the_boundary():
    candles = _noisy(400, vol=0.03, seed=23)
    low = VolTrendParams(trend_lookback=100, vol_lookback=20, target_vol=0.40)
    high = VolTrendParams(trend_lookback=100, vol_lookback=20, target_vol=0.60)

    swap_at = 250
    swapped = voltrend_schedule(
        candles, [VolTrendSpan(0, low), VolTrendSpan(swap_at, high)]
    )
    stayed = voltrend_schedule(candles, [VolTrendSpan(0, low)])

    np.testing.assert_array_equal(
        swapped.to_numpy()[:swap_at], stayed.to_numpy()[:swap_at]
    )
    assert not np.array_equal(swapped.to_numpy(), stayed.to_numpy())
    # the higher target must size at least as large after the swap
    assert (swapped.to_numpy()[swap_at:] >= stayed.to_numpy()[swap_at:] - 1e-12).all()


def test_schedule_respects_trade_start():
    candles = _trending(400)
    sig = voltrend_schedule(candles, [VolTrendSpan(0, DEFAULTS)], trade_start=220)
    assert (sig.iloc[:220] == 0.0).all()
    assert sig.iloc[220:].max() > 0.0


def test_schedule_rejects_a_span_starting_after_trading():
    candles = _trending(300)
    with pytest.raises(ValueError, match="every traded candle needs parameters"):
        voltrend_schedule(candles, [VolTrendSpan(100, DEFAULTS)], trade_start=50)


# --- engine integration ------------------------------------------------------


def test_engine_accepts_the_signal_and_the_dead_band_suppresses_chatter():
    """Fractional sizing without a dead-band would rebalance every candle."""
    candles = _noisy(500, seed=29)
    banded = backtest(
        candles, lambda c: voltrend(c, DEFAULTS), BacktestConfig(min_rebalance_delta=0.05),
        pair="XBTEUR",
    )
    unbanded = backtest(
        candles, lambda c: voltrend(c, DEFAULTS), BacktestConfig(min_rebalance_delta=0.0),
        pair="XBTEUR",
    )
    assert len(banded.fills) < len(unbanded.fills)
    assert banded.suppressed_rebalances > 0
    assert banded.total_fees < unbanded.total_fees


def test_books_reconcile_under_fractional_sizing():
    """The partial-sell accounting fix must hold for this strategy too."""
    candles = _noisy(500, seed=31)
    result = backtest(
        candles, lambda c: voltrend(c, DEFAULTS), BacktestConfig(), pair="XBTEUR"
    )
    accounted = result.realized_pnl + result.unrealized_pnl
    actual = result.final_equity - result.config.starting_capital
    assert accounted == pytest.approx(actual, abs=0.01)
