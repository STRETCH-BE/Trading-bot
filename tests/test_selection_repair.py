"""Repair of FINDINGS.md §3: fabricated-zero selection.

The defect: a combo whose warmup met or exceeded the in-sample window emitted
all-zero signals, received an in-sample Sharpe of 0.0 that meant "no evidence"
but was scored as "neutral evidence", and won a bare argmax whenever every
evaluable combo scored negative. In the recorded strategy #2 run this
deployed structurally unevaluated configurations in up to 74/145 windows.

Amendment 6 makes this repair a precondition of the strategy #3 gate run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtest import BacktestConfig
from trading_bot.backtest.walkforward import (
    GridPoint,
    NoEvaluableComboError,
    search_window,
    select_best,
    voltrend_grid,
)
from trading_bot.strategies.voltrend import VolTrendParams, voltrend

from .synthetic import from_closes

CONFIG = BacktestConfig(
    starting_capital=10_000.0, min_rebalance_delta=0.05, strategy_max_allocation=1.0
)


def _voltrend_factory(candles, params):
    return voltrend(candles, params)


# --- the regression: the exact recorded scenario -----------------------------


def test_trend_200_is_ineligible_in_a_180_candle_window():
    """The literal defect case: voltrend trend_lookback=200 in a 180-candle
    in-sample window (main_180_60 config). It used to score a fabricated 0.0;
    it must now be ineligible and carry -inf."""
    candles = from_closes([100.0 * (1.002**i) for i in range(180)])
    points = search_window(
        candles, voltrend_grid(), CONFIG, "XBTEUR", signal_factory=_voltrend_factory
    )

    dead = [p for p in points if p.params.trend_lookback == 200]
    live = [p for p in points if p.params.trend_lookback != 200]
    assert len(dead) == 4 and len(live) == 8

    for p in dead:
        assert not p.eligible
        assert p.sharpe == float("-inf")
        assert p.total_return == float("-inf")
    for p in live:
        assert p.eligible
        assert np.isfinite(p.sharpe)


def test_fabricated_zero_cannot_beat_a_negative_real_score():
    """The selection consequence, isolated: all evaluable combos negative,
    one unevaluable combo present. The old argmax picked the fabrication;
    select_best must pick the least-bad REAL evidence."""
    live = GridPoint(VolTrendParams(trend_lookback=50), sharpe=-0.8, total_return=-0.2)
    dead = GridPoint(
        VolTrendParams(trend_lookback=200),
        sharpe=float("-inf"), total_return=float("-inf"), eligible=False,
    )
    best = select_best([dead, live], "sharpe")
    assert best is live, "an unevaluated combo out-selected real negative evidence"
    assert select_best([dead, live], "total_return") is live


def test_raw_tuple_argmax_also_cannot_pick_minus_inf():
    """The run scripts reimplement the argmax over (sharpe, return) tuples
    rather than calling select_best (a seam noted in FINDINGS §3). -inf must
    lose ANY max() against any finite score, so the repair covers that path
    without requiring every caller to know about eligibility."""
    arr = [(-0.9, -0.3), (float("-inf"), float("-inf")), (-2.5, -0.9)]
    best = max(range(len(arr)), key=lambda i: arr[i][0])
    assert best == 0
    best_ret = max(range(len(arr)), key=lambda i: arr[i][1])
    assert best_ret == 0


def test_all_combos_unevaluable_raises_not_selects():
    """A grid/window mismatch must be a protocol error, never a selection."""
    candles = from_closes([100.0] * 40)  # shorter than every voltrend warmup>=50...
    # trend_lookback=50 -> warmup 50 >= 40; all 12 combos dead
    with pytest.raises(NoEvaluableComboError):
        search_window(
            candles, voltrend_grid(), CONFIG, "XBTEUR",
            signal_factory=_voltrend_factory,
        )
    with pytest.raises(NoEvaluableComboError):
        select_best(
            [GridPoint(VolTrendParams(), float("-inf"), float("-inf"), eligible=False)],
            "sharpe",
        )


def test_evaluable_but_flat_combo_keeps_its_real_zero():
    """The boundary the repair must NOT cross: a combo that COULD trade and
    chose not to has presented genuine evidence ('I would have sat out').
    Its 0.0 stays, and it stays eligible."""
    # dead-flat market: trend filter (close > SMA) never turns on
    candles = from_closes([100.0] * 120)
    points = search_window(
        candles,
        [VolTrendParams(trend_lookback=50, vol_lookback=20, target_vol=0.40)],
        CONFIG, "XBTEUR", signal_factory=_voltrend_factory,
    )
    assert points[0].eligible
    assert points[0].sharpe == 0.0  # real evidence, not fabrication


def test_partially_warmed_combo_is_still_evaluable():
    """warmup < window: reduced evidence is evidence. trend=100 in 120
    candles trades on the last ~20 — eligible, finite score."""
    candles = from_closes([100.0 * (1.01**i) for i in range(120)])
    points = search_window(
        candles,
        [VolTrendParams(trend_lookback=100, vol_lookback=20, target_vol=0.40)],
        CONFIG, "XBTEUR", signal_factory=_voltrend_factory,
    )
    assert points[0].eligible
    assert np.isfinite(points[0].sharpe)


def test_donchian_default_grid_untouched_by_the_repair():
    """Donchian's warmup (<= 60) fits every preregistered window, so the
    repair must not change its selection universe at all."""
    from trading_bot.backtest.walkforward import default_grid

    candles = from_closes(list(100 * np.cumprod(1 + np.random.default_rng(7).normal(0, 0.02, 90))))
    points = search_window(candles, default_grid(), CONFIG, "XBTEUR")
    assert all(p.eligible for p in points)
    assert all(np.isfinite(p.sharpe) for p in points)


# --- capital-matched benchmark (F6 repair, Amendment 4) ----------------------


def test_capital_matched_equals_buy_and_hold_at_allocation_one():
    from trading_bot.backtest.engine import buy_and_hold_equity, capital_matched_hold_equity

    candles = from_closes([100.0 * (1.005**i) for i in range(200)])
    config = BacktestConfig(strategy_max_allocation=1.0)
    pd.testing.assert_series_equal(
        capital_matched_hold_equity(candles, config),
        buy_and_hold_equity(candles, config),
        check_names=False,
    )


def test_capital_matched_identity_alloc_scaled_bh_plus_cash():
    """curve == (1-alloc)*C + alloc * (B&H at the same entry), exactly."""
    from trading_bot.backtest.engine import buy_and_hold_equity, capital_matched_hold_equity

    candles = from_closes([100.0 * (1.004**i) for i in range(150)])
    config = BacktestConfig(strategy_max_allocation=0.25)
    cm = capital_matched_hold_equity(candles, config)
    bh = buy_and_hold_equity(candles, config)
    C = config.starting_capital
    expected = 0.75 * C + 0.25 * bh
    np.testing.assert_allclose(cm.to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_metrics_report_both_benchmarks():
    from trading_bot.backtest import backtest, compute_metrics

    candles = from_closes([100.0 * (1.004**i) for i in range(150)])
    config = BacktestConfig(
        starting_capital=10_000.0, min_order_units=0.0, strategy_max_allocation=0.25
    )
    result = backtest(candles, lambda c: pd.Series([0.0] * len(c)), config)
    m = compute_metrics(result, config)

    # never-trade strategy: 0 return. On a rising market the uncapped B&H
    # dwarfs the capital-matched one; both must be present and ordered.
    assert m.buy_and_hold_return > m.capital_matched_return > 0
    assert m.excess_return < m.excess_vs_capital_matched < 0
    # identity: cm_return == alloc * bh_return (cash contributes zero return)
    assert m.capital_matched_return == pytest.approx(0.25 * m.buy_and_hold_return, rel=1e-9)


def test_capital_matched_is_the_harsher_benchmark_never_the_easier_one_at_full_alloc():
    """Guard against the obvious cheat: at allocation 1.0 the 'new' benchmark
    must not be softer than the old one — they are the same number."""
    from trading_bot.backtest import backtest, compute_metrics

    candles = from_closes([100.0 * (1.004**i) for i in range(150)])
    config = BacktestConfig(
        starting_capital=10_000.0, min_order_units=0.0, strategy_max_allocation=1.0
    )
    result = backtest(candles, lambda c: pd.Series([1.0] * len(c)), config)
    m = compute_metrics(result, config)
    assert m.capital_matched_return == pytest.approx(m.buy_and_hold_return, rel=1e-12)
    assert m.excess_vs_capital_matched == pytest.approx(m.excess_return, rel=1e-9, abs=1e-12)
