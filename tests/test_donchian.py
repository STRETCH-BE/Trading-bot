"""Donchian breakout: causality, warmup, and behavioural tests."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from trading_bot.data import schema
from trading_bot.strategies import DonchianParams, donchian_breakout

from .synthetic import from_closes, make_candles

DEFAULTS = DonchianParams()


# --- causality ---------------------------------------------------------------


def test_future_data_cannot_change_past_signals():
    """Shuffle everything after an arbitrary index; earlier signals must not move.

    This is the test that catches an accidental look-ahead — a centred rolling
    window, a backward fill, a global normalisation, anything that peeks.
    """
    rng = np.random.default_rng(7)
    closes = list(100 * np.cumprod(1 + rng.normal(0, 0.02, 400)))
    candles = from_closes(closes)

    cut = 250
    baseline = donchian_breakout(candles, DEFAULTS)

    for seed in range(5):
        shuffler = np.random.default_rng(seed)
        tampered = candles.copy()
        tail = tampered.iloc[cut:].copy()
        order = shuffler.permutation(len(tail))
        for col in (schema.OPEN, schema.HIGH, schema.LOW, schema.CLOSE, schema.VOLUME):
            tampered.loc[tampered.index[cut:], col] = tail[col].to_numpy()[order]

        tampered_signals = donchian_breakout(tampered, DEFAULTS)
        np.testing.assert_array_equal(
            baseline.to_numpy()[:cut],
            tampered_signals.to_numpy()[:cut],
            err_msg=f"future shuffle (seed {seed}) changed a past signal",
        )


def test_truncating_the_series_does_not_change_earlier_signals():
    """A prefix of the data must yield a prefix of the signals, exactly."""
    rng = np.random.default_rng(11)
    closes = list(100 * np.cumprod(1 + rng.normal(0, 0.015, 300)))
    candles = from_closes(closes)

    full = donchian_breakout(candles, DEFAULTS)
    for cut in (60, 120, 200):
        partial = donchian_breakout(candles.iloc[:cut].copy(), DEFAULTS)
        np.testing.assert_array_equal(full.to_numpy()[:cut], partial.to_numpy())


# --- warmup ------------------------------------------------------------------


def test_warmup_period_is_flat_and_never_nan():
    rng = np.random.default_rng(3)
    closes = list(100 * np.cumprod(1 + rng.normal(0.002, 0.02, 200)))
    signals = donchian_breakout(from_closes(closes), DEFAULTS)

    assert not signals.isna().any(), "warmup produced NaN instead of 0.0"
    warmup = max(DEFAULTS.entry_lookback, DEFAULTS.atr_period)
    assert (signals.iloc[:warmup] == 0.0).all(), "spurious signal during warmup"


def test_warmup_respects_custom_parameters():
    params = DonchianParams(entry_lookback=50, exit_lookback=5, atr_period=30)
    closes = [100.0 + i for i in range(120)]  # relentless rise
    signals = donchian_breakout(from_closes(closes), params)
    assert (signals.iloc[: params.warmup] == 0.0).all()
    assert signals.iloc[params.warmup :].max() == 1.0  # but it does eventually enter


def test_series_shorter_than_warmup_is_all_flat():
    signals = donchian_breakout(from_closes([100.0] * 10), DEFAULTS)
    assert (signals == 0.0).all()
    assert not signals.isna().any()


# --- output contract ---------------------------------------------------------


def test_signal_values_are_only_zero_or_one():
    rng = np.random.default_rng(5)
    closes = list(100 * np.cumprod(1 + rng.normal(0, 0.03, 500)))
    signals = donchian_breakout(from_closes(closes), DEFAULTS)
    assert set(np.unique(signals.to_numpy())) <= {0.0, 1.0}
    assert (signals >= 0).all() and (signals <= 1).all()


def test_signal_is_aligned_to_the_candle_index():
    candles = from_closes([100.0 + i for i in range(80)])
    candles.index = range(1000, 1000 + len(candles))  # deliberately odd index
    signals = donchian_breakout(candles, DEFAULTS)
    assert list(signals.index) == list(candles.index)


def test_row_labels_do_not_affect_the_result():
    """Only the monotonic timestamp order matters, not the DataFrame's labels."""
    candles = from_closes([100.0 + (i % 7) * 3 for i in range(120)])
    relabelled = candles.copy()
    relabelled.index = list(range(500, 500 + len(candles)))
    np.testing.assert_array_equal(
        donchian_breakout(candles, DEFAULTS).to_numpy(),
        donchian_breakout(relabelled, DEFAULTS).to_numpy(),
    )


def test_non_monotonic_candles_are_rejected():
    candles = from_closes([100.0] * 40).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="monotonically increasing"):
        donchian_breakout(candles, DEFAULTS)


# --- behaviour ---------------------------------------------------------------


def test_constant_price_never_enters():
    """No breakout can exist if the price never moves."""
    signals = donchian_breakout(from_closes([100.0] * 300), DEFAULTS)
    assert (signals == 0.0).all()


def test_step_function_enters_once_and_holds():
    """Flat, then a clean sustained rise: enter shortly after the break, stay long."""
    closes = [100.0] * 60 + [100.0 + 2.0 * i for i in range(1, 121)]
    candles = from_closes(closes)
    signals = donchian_breakout(candles, DEFAULTS)

    long_idx = np.flatnonzero(signals.to_numpy() == 1.0)
    assert len(long_idx) > 0, "never entered a clean sustained uptrend"

    entry = int(long_idx[0])
    assert 60 <= entry <= 66, f"entered at {entry}, expected just after the break at 60"

    # a single unbroken holding run: entered once, no chattering
    assert list(long_idx) == list(range(entry, entry + len(long_idx)))
    assert signals.iloc[-1] == 1.0, "exited a still-rising trend"

    transitions = np.abs(np.diff(signals.to_numpy())).sum()
    assert transitions == 1.0, f"expected exactly one entry, saw {transitions} flips"


def test_sawtooth_does_not_enter_and_exit_on_the_same_candle():
    """A whipsaw must never produce a same-candle round trip."""
    closes = [(100.0 if i % 2 == 0 else 140.0) for i in range(300)]
    candles = from_closes(closes)
    signals = donchian_breakout(candles, DEFAULTS)

    values = signals.to_numpy()
    for i in range(1, len(values)):
        if values[i] == 1.0 and values[i - 1] == 0.0:
            # the candle it entered on must remain long, not flip back same-bar
            assert values[i] == 1.0
    # entries and exits must land on distinct candles
    entries = {i for i in range(1, len(values)) if values[i] == 1 and values[i - 1] == 0}
    exits = {i for i in range(1, len(values)) if values[i] == 0 and values[i - 1] == 1}
    assert entries.isdisjoint(exits)


def test_trailing_stop_exits_a_reversal():
    """Rise then collapse: the ATR trail must get us out, not ride it down."""
    closes = [100.0] * 40 + [100.0 + 3.0 * i for i in range(1, 61)] + [
        280.0 - 12.0 * i for i in range(1, 25)
    ]
    signals = donchian_breakout(from_closes(closes), DEFAULTS)
    assert signals.iloc[-1] == 0.0, "still long after a sustained collapse"
    assert signals.max() == 1.0, "never entered the uptrend at all"


def test_tighter_atr_multiple_exits_no_later():
    """A tighter trailing stop cannot exit later than a looser one."""
    closes = [100.0] * 40 + [100.0 + 2.0 * i for i in range(1, 61)] + [
        220.0 - 6.0 * i for i in range(1, 21)
    ]
    candles = from_closes(closes)
    tight = donchian_breakout(candles, DonchianParams(atr_multiple=1.5)).to_numpy()
    loose = donchian_breakout(candles, DonchianParams(atr_multiple=4.0)).to_numpy()

    def last_long(arr):
        idx = np.flatnonzero(arr == 1.0)
        return int(idx[-1]) if len(idx) else -1

    assert last_long(tight) <= last_long(loose)


def test_no_pyramiding_only_one_position_at_a_time():
    """Signals are strictly binary, so exposure can never stack above 1.0."""
    closes = [100.0 + 5.0 * i for i in range(200)]  # continuous breakouts
    signals = donchian_breakout(from_closes(closes), DEFAULTS)
    assert signals.max() <= 1.0


# --- params ------------------------------------------------------------------


def test_params_are_frozen():
    params = DonchianParams()
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.entry_lookback = 99  # type: ignore[misc]


def test_invalid_params_rejected():
    with pytest.raises(ValueError, match="entry_lookback"):
        DonchianParams(entry_lookback=0)
    with pytest.raises(ValueError, match="atr_multiple"):
        DonchianParams(atr_multiple=-1.0)


def test_params_load_from_config_yaml(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "strategies:\n  donchian:\n    entry_lookback: 30\n    atr_multiple: 3.0\n"
    )
    params = DonchianParams.load(cfg)
    assert params.entry_lookback == 30
    assert params.atr_multiple == 3.0
    assert params.exit_lookback == 10  # default preserved


def test_repo_config_yaml_matches_declared_defaults():
    """config.yaml must not silently drift from the dataclass defaults."""
    assert DonchianParams.load("config.yaml") == DonchianParams()


def test_missing_config_raises_rather_than_defaulting(tmp_path):
    """FINDING 6: a mistyped path must never masquerade as success."""
    from trading_bot.backtest.config import ConfigNotFoundError

    with pytest.raises(ConfigNotFoundError, match="config file not found"):
        DonchianParams.load(tmp_path / "absent.yaml")


# --- purity ------------------------------------------------------------------


def test_strategy_is_pure_repeated_calls_are_identical():
    rng = np.random.default_rng(13)
    closes = list(100 * np.cumprod(1 + rng.normal(0, 0.02, 250)))
    candles = from_closes(closes)
    first = donchian_breakout(candles, DEFAULTS)
    second = donchian_breakout(candles, DEFAULTS)
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())


def test_strategy_does_not_mutate_its_input():
    candles = from_closes([100.0 + i for i in range(100)])
    before = candles.copy(deep=True)
    donchian_breakout(candles, DEFAULTS)
    pd.testing.assert_frame_equal(candles, before)


def test_strategy_output_is_accepted_by_the_engine():
    from trading_bot.backtest import BacktestConfig, backtest

    candles = from_closes([100.0 + (i % 30) for i in range(200)])
    result = backtest(
        candles,
        lambda c: donchian_breakout(c, DEFAULTS),
        BacktestConfig(min_order_units=0.0),
    )
    assert isinstance(result.equity, pd.Series)


def test_make_candles_helper_builds_valid_frames():
    candles = make_candles([(1.0, 2.0, 0.5, 1.5, 3.0)])
    assert candles[schema.TIMESTAMP].dt.tz is not None
