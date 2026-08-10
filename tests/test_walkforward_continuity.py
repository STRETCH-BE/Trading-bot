"""Walk-forward position continuity: parameters swap at boundaries, state never.

The mandated proof: a strategy holding continuously across three windows pays
exactly ONE entry — not three round trips.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest import BacktestConfig, backtest
from trading_bot.backtest.walkforward import (
    WindowSpec,
    make_windows,
    run_oos_concatenated,
)
from trading_bot.data import schema
from trading_bot.strategies.donchian import (
    DonchianParams,
    ParamSpan,
    donchian_breakout,
    donchian_breakout_schedule,
)

from .synthetic import from_closes

CONFIG = BacktestConfig(min_order_units=0.0, min_rebalance_delta=0.0)


def _three_windows(first_oos: int, oos_len: int) -> list[WindowSpec]:
    return [
        WindowSpec(
            idx=k,
            is_lo=first_oos - 180 + k * oos_len,
            is_hi=first_oos + k * oos_len,
            oos_lo=first_oos + k * oos_len,
            oos_hi=first_oos + (k + 1) * oos_len,
            partial=False,
        )
        for k in range(3)
    ]


PARAMS_A = DonchianParams(entry_lookback=20, exit_lookback=10, atr_multiple=2.5)
PARAMS_B = DonchianParams(entry_lookback=40, exit_lookback=5, atr_multiple=4.0)
PARAMS_C = DonchianParams(entry_lookback=10, exit_lookback=25, atr_multiple=1.5)


def test_hold_across_three_windows_pays_exactly_one_entry():
    """THE continuity proof, with an always-long scheduler."""
    candles = from_closes([100.0 + 0.1 * i for i in range(400)])
    windows = _three_windows(first_oos=220, oos_len=60)
    selections = [(w, p) for w, p in zip(windows, [PARAMS_A, PARAMS_B, PARAMS_C], strict=False)]

    def always_long(cs, spans, trade_start):
        sig = np.zeros(len(cs))
        sig[trade_start:] = 1.0
        return pd.Series(sig, index=cs.index)

    result = run_oos_concatenated(candles, selections, CONFIG, None, scheduler=always_long)

    # exactly one fill: the entry. No boundary closes, no boundary re-entries.
    assert [f.side for f in result.fills] == ["buy"]
    assert result.total_fees == pytest.approx(result.fills[0].fee)
    assert not result.ends_flat  # carried straight through to the end

    # and the fee equals ONE entry on the full capital, not three round trips
    entry_notional = CONFIG.starting_capital / (1 + CONFIG.fee_rate)
    assert result.fills[0].fee == pytest.approx(entry_notional * CONFIG.fee_rate)


def test_donchian_holds_a_trend_across_boundaries_with_different_params():
    """Same proof through the real strategy: an uptrend held across three
    windows with three different parameter sets still pays one entry."""
    closes = [100.0 * (1.02**i) for i in range(400)]  # relentless uptrend
    candles = from_closes(closes)
    windows = _three_windows(first_oos=220, oos_len=60)
    selections = [(w, p) for w, p in zip(windows, [PARAMS_A, PARAMS_B, PARAMS_C], strict=False)]

    result = run_oos_concatenated(candles, selections, CONFIG, None)

    buys = [f for f in result.fills if f.side == "buy"]
    sells = [f for f in result.fills if f.side == "sell"]
    assert len(buys) == 1, "re-entered at a boundary it never exited"
    assert sells == [], "a boundary forced an exit the strategy never chose"
    assert not result.ends_flat


def test_concatenated_equals_continuous_run_when_params_never_change():
    """Window boundaries with identical params must be invisible: the
    concatenated curve is byte-identical to one uninterrupted run."""
    rng = np.random.default_rng(21)
    closes = list(100 * np.cumprod(1 + rng.normal(0.001, 0.02, 400)))
    candles = from_closes(closes)
    windows = _three_windows(first_oos=220, oos_len=60)
    selections = [(w, PARAMS_A) for w in windows]

    concatenated = run_oos_concatenated(candles, selections, CONFIG, None)

    full_signal = donchian_breakout_schedule(
        candles, [ParamSpan(0, PARAMS_A)], trade_start=220
    )
    oos_candles = candles.iloc[220:400].reset_index(drop=True)
    oos_signal = pd.Series(full_signal.to_numpy()[220:400], index=oos_candles.index)
    continuous = backtest(oos_candles, lambda _c: oos_signal, CONFIG)

    pd.testing.assert_series_equal(concatenated.equity, continuous.equity)
    assert len(concatenated.fills) == len(continuous.fills)


def test_schedule_with_one_span_equals_plain_donchian():
    """The schedule core and the public single-params function are the same
    machine — donchian_breakout is literally a one-span schedule."""
    rng = np.random.default_rng(33)
    closes = list(100 * np.cumprod(1 + rng.normal(0, 0.025, 300)))
    candles = from_closes(closes)
    for params in (PARAMS_A, PARAMS_B, PARAMS_C):
        np.testing.assert_array_equal(
            donchian_breakout(candles, params).to_numpy(),
            donchian_breakout_schedule(candles, [ParamSpan(0, params)]).to_numpy(),
        )


def test_schedule_swaps_parameters_at_the_boundary():
    """A parameter swap must actually take effect (not just not break state):
    with a tight trail after the boundary, a decline exits; with a loose one
    it may not — the two schedules must diverge after the boundary and agree
    before it."""
    closes = (
        [100.0] * 60
        + [100.0 + 2.0 * i for i in range(1, 101)]  # rise to 300
        + [300.0 - 1.5 * i for i in range(1, 81)]  # slow bleed
    )
    candles = from_closes(closes)
    tight = DonchianParams(entry_lookback=20, exit_lookback=10, atr_multiple=1.5)
    loose = DonchianParams(entry_lookback=20, exit_lookback=10, atr_multiple=4.0)

    swap_at = 160
    swapped = donchian_breakout_schedule(
        candles, [ParamSpan(0, loose), ParamSpan(swap_at, tight)]
    )
    stayed = donchian_breakout_schedule(candles, [ParamSpan(0, loose)])

    np.testing.assert_array_equal(
        swapped.to_numpy()[:swap_at], stayed.to_numpy()[:swap_at]
    )
    assert not np.array_equal(swapped.to_numpy(), stayed.to_numpy()), (
        "the parameter swap had no effect at all — schedule is not being applied"
    )
    # the tighter trail must never exit LATER than the loose one
    tight_exit = np.argmin(swapped.to_numpy()[swap_at:] == 1.0)
    loose_exit = np.argmin(stayed.to_numpy()[swap_at:] == 1.0)
    assert tight_exit <= loose_exit


def test_make_windows_contiguous_oos_coverage():
    windows = make_windows(n_candles=1000, is_len=180, oos_len=60, step=60)
    assert windows[0].oos_lo == 180
    for a, b in zip(windows, windows[1:], strict=False):
        assert b.oos_lo == a.oos_hi, "hole or overlap between OOS windows"
    assert windows[-1].oos_hi == 1000  # tail is used (flagged partial if short)
    full = [w for w in windows if not w.partial]
    assert all(w.oos_hi - w.oos_lo == 60 for w in full)


def test_make_windows_rejects_step_not_equal_oos():
    with pytest.raises(ValueError, match="step"):
        make_windows(1000, 180, 60, 30)


def test_oos_run_starts_flat_with_starting_capital():
    candles = from_closes([100.0] * 400)
    windows = _three_windows(first_oos=220, oos_len=60)
    selections = [(w, PARAMS_A) for w in windows]
    result = run_oos_concatenated(candles, selections, CONFIG, None)
    assert float(result.equity.iloc[0]) == CONFIG.starting_capital
    assert result.equity.index[0] == candles[schema.TIMESTAMP].iloc[220]
