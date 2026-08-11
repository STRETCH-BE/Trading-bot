"""Strategy #3 mean reversion: the Amendment 6 specification, tested.

The causality proof itself lives in test_causality.py, which picks meanrev up
from the registry automatically. These tests pin the SEMANTICS: entry and
exit bands, hysteresis, warmup, schedule continuity, and the grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.data import schema
from trading_bot.strategies.meanrev import (
    MeanRevParams,
    MeanRevSpan,
    meanrev,
    meanrev_schedule,
)

from .synthetic import from_closes


def dip_market(n_flat: int = 40, dip: float = 0.85, recover_at: float = 1.0):
    """Flat at 100, one sharp dip, then recovery to the MA."""
    closes = [100.0] * n_flat + [100.0 * dip] + [100.0 * recover_at] * 10
    return from_closes(closes)


# --- entry and exit semantics ------------------------------------------------


def test_enters_on_a_k_sigma_dip_and_exits_on_recovery():
    # gentle noise so sigma is small but nonzero; then a violent dip
    rng = np.random.default_rng(11)
    base = list(100 + rng.normal(0, 0.3, 60))
    closes = base + [90.0] + [100.0] * 5  # dip >> K*sigma, then full recovery
    candles = from_closes(closes)
    sig = meanrev(candles, MeanRevParams(lookback=20, entry_k=1.5, exit_e=0.0))

    assert sig.iloc[60] == 1.0, "a many-sigma dip did not trigger entry"
    assert sig.iloc[61] == 0.0, "recovery to the MA did not trigger exit"
    assert (sig.iloc[:20] == 0.0).all(), "signal emitted during warmup"


def test_no_entry_without_a_dip():
    candles = from_closes([100.0 + 0.05 * i for i in range(80)])  # steady drift up
    sig = meanrev(candles, MeanRevParams(lookback=20, entry_k=1.0, exit_e=0.0))
    assert (sig == 0.0).all()


def test_hysteresis_holds_between_the_bands():
    """Price enters on a deep dip, then sits BETWEEN the entry and exit bands:
    the position must be held, not churned."""
    rng = np.random.default_rng(5)
    base = list(100 + rng.normal(0, 0.5, 40))
    # deep dip, then hover well below the MA but above the entry band
    closes = base + [92.0] + [96.0] * 6 + [101.0]
    candles = from_closes(closes)
    p = MeanRevParams(lookback=20, entry_k=1.5, exit_e=0.0)
    sig = meanrev(candles, p)

    assert sig.iloc[40] == 1.0  # entered on the dip
    # hovering: sigma exploded after the dip, so 96 stays below MA - 0*sigma
    held = sig.iloc[41:46]
    assert (held == 1.0).all(), f"position churned while between bands: {list(held)}"
    assert sig.iloc[-1] == 0.0  # full recovery exits


def test_exit_e_bites_earlier_than_e_zero():
    """E=0.5 exits when price is still half a sigma BELOW the MA, so its
    exit can never come later than E=0.0's."""
    rng = np.random.default_rng(9)
    base = list(100 + rng.normal(0, 1.0, 50))
    closes = base + [90.0] + list(np.linspace(90.5, 102, 12))
    candles = from_closes(closes)
    tight = meanrev(candles, MeanRevParams(lookback=20, entry_k=1.5, exit_e=0.5))
    loose = meanrev(candles, MeanRevParams(lookback=20, entry_k=1.5, exit_e=0.0))

    t_exit = next(i for i in range(51, len(candles)) if tight.iloc[i] == 0.0)
    l_exit = next(i for i in range(51, len(candles)) if loose.iloc[i] == 0.0)
    assert t_exit <= l_exit


def test_output_contract():
    rng = np.random.default_rng(3)
    candles = from_closes(list(100 * np.cumprod(1 + rng.normal(0, 0.03, 300))))
    sig = meanrev(candles, MeanRevParams())
    assert set(np.unique(sig.to_numpy())) <= {0.0, 1.0}, "long-or-flat only"
    assert not sig.isna().any()


def test_exit_band_must_sit_above_entry_band():
    with pytest.raises(ValueError, match="exit_e"):
        MeanRevParams(lookback=20, entry_k=1.0, exit_e=1.0)
    with pytest.raises(ValueError, match="exit_e"):
        MeanRevParams(lookback=20, entry_k=1.0, exit_e=1.5)


def test_warmup_is_the_lookback():
    for n in (10, 20, 30):
        assert MeanRevParams(lookback=n).warmup == n


# --- schedule: walk-forward continuity ---------------------------------------


def test_schedule_with_one_span_equals_plain_meanrev():
    rng = np.random.default_rng(21)
    candles = from_closes(list(100 * np.cumprod(1 + rng.normal(0, 0.04, 250))))
    for p in (MeanRevParams(10, 1.0, 0.0), MeanRevParams(30, 1.5, 0.5)):
        np.testing.assert_array_equal(
            meanrev(candles, p).to_numpy(),
            meanrev_schedule(candles, [MeanRevSpan(0, p)]).to_numpy(),
        )


def test_position_carries_across_a_span_boundary():
    """In a position at the boundary, with the new span's entry band never
    re-triggered: the position must carry, not force-close or re-enter."""
    rng = np.random.default_rng(2)
    base = list(100 + rng.normal(0, 0.4, 60))
    # Dip at 60, then a slow DECLINE (not a plateau: a constant price would
    # converge with the new span's short MA and legitimately trigger its
    # exit — price must stay below MA for the carry to be observable), then
    # a sharp recovery that exits.
    closes = base + [90.0] + [93.0 - 0.2 * j for j in range(15)] + [104.0] * 5
    candles = from_closes(closes)
    a = MeanRevParams(lookback=20, entry_k=1.5, exit_e=0.0)
    b = MeanRevParams(lookback=10, entry_k=1.0, exit_e=0.0)

    swapped = meanrev_schedule(candles, [MeanRevSpan(0, a), MeanRevSpan(70, b)])
    assert swapped.iloc[60] == 1.0, "did not enter on the dip"
    assert swapped.iloc[69] == 1.0, "exited before the boundary for no reason"
    assert swapped.iloc[70] == 1.0, (
        "the position did not carry across the parameter swap"
    )


def test_trade_start_pins_flat_before_the_oos_start():
    rng = np.random.default_rng(13)
    candles = from_closes(list(100 * np.cumprod(1 + rng.normal(0, 0.05, 200))))
    p = MeanRevParams(lookback=10, entry_k=1.0, exit_e=0.0)
    sig = meanrev_schedule(candles, [MeanRevSpan(0, p)], trade_start=120)
    assert (sig.iloc[:120] == 0.0).all()


# --- the preregistered grid --------------------------------------------------


def test_grid_is_exactly_the_amendment_6_twelve():
    from trading_bot.backtest.walkforward import meanrev_grid

    grid = meanrev_grid()
    assert len(grid) == 12
    assert {p.lookback for p in grid} == {10, 20, 30}
    assert {p.entry_k for p in grid} == {1.0, 1.5}
    assert {p.exit_e for p in grid} == {0.0, 0.5}
    assert len(set(grid)) == 12, "duplicate combos"
    # every combo is evaluable in the smallest preregistered window
    assert max(p.warmup for p in grid) < 90


def test_repo_config_yaml_matches_declared_defaults():
    assert MeanRevParams.load("config.yaml") == MeanRevParams()


def test_registered_for_the_causality_gate():
    from trading_bot.strategies.registry import STRATEGIES

    assert "meanrev" in STRATEGIES
    candles = from_closes([100.0] * 50)
    out = STRATEGIES["meanrev"](candles)
    assert isinstance(out, pd.Series) and len(out) == len(candles)


def test_signal_uses_only_data_through_bar_t():
    """Direct spot-check on top of the registry gate: changing candle t+1
    must not change the signal at t."""
    rng = np.random.default_rng(31)
    closes = list(100 + rng.normal(0, 2.0, 100))
    a = from_closes(closes)
    b = from_closes(closes[:-1] + [closes[-1] * 3])  # mutate the final close
    p = MeanRevParams(lookback=20, entry_k=1.0, exit_e=0.0)
    np.testing.assert_array_equal(
        meanrev(a, p).to_numpy()[:-1], meanrev(b, p).to_numpy()[:-1]
    )


def test_ties_to_schema_columns_not_positions():
    candles = from_closes([100.0] * 30)
    shuffled_cols = candles[[schema.CLOSE, schema.TIMESTAMP, schema.OPEN,
                             schema.HIGH, schema.LOW, schema.VOLUME]]
    np.testing.assert_array_equal(
        meanrev(candles).to_numpy(), meanrev(shuffled_cols).to_numpy()
    )
