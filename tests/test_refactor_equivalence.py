"""Stage 5a equivalence proof: the refactor changed no number.

Compares the post-refactor engine against a golden baseline captured BEFORE
the order seam and FillModel existed (`scripts/capture_equivalence_baseline.py`,
committed at 049ae93).

The comparison is bit-exact, not 8-decimal. json round-trips Python floats
losslessly via repr, so any difference at all — including a last-bit change
from reordering a floating-point expression — fails here.

If this test fails, the refactor is wrong. Find out why. Do NOT regenerate
the baseline and do NOT loosen the tolerance; either would convert a proof
into a rubber stamp.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading_bot.backtest import BacktestConfig, backtest
from trading_bot.backtest.walkforward import (
    make_windows,
    run_oos_concatenated,
    voltrend_grid,
    voltrend_scheduler,
)
from trading_bot.data import schema
from trading_bot.data.liquidity import FloorRule, sustained_floor_date
from trading_bot.data.store import ParquetStore
from trading_bot.strategies.voltrend import VolTrendParams, voltrend

GOLDEN = Path(__file__).parent / "golden" / "equivalence_baseline.json"
DATA_DIR = Path("data/parquet")
FLOOR_RULE = FloorRule(min_median_trades=500.0, max_missing=1, window_days=30)

CONFIG = BacktestConfig(
    starting_capital=10_000.0,
    maker_fee_bps=16.0,
    taker_fee_bps=26.0,
    fee_mode="taker",
    slippage_bps=5.0,
    min_rebalance_delta=0.05,
    periods_per_year=365.0,
)

pytestmark = pytest.mark.skipif(
    not GOLDEN.exists() or not (DATA_DIR / "XBTEUR" / "1d.parquet").exists(),
    reason="requires the ingested market data and the captured golden baseline",
)


def snapshot(result) -> dict:
    """Must stay identical to the capture script's snapshot()."""
    return {
        "equity": [float(v) for v in result.equity.to_numpy()],
        "final_equity": result.final_equity,
        "total_fees": result.total_fees,
        "realized_pnl": result.realized_pnl,
        "unrealized_pnl": result.unrealized_pnl,
        "open_units": result.open_units,
        "skipped_orders": result.skipped_orders,
        "skipped_costmin": result.skipped_costmin,
        "suppressed_rebalances": result.suppressed_rebalances,
        "fills": [
            {
                "time": f.time.isoformat(),
                "side": f.side,
                "price": f.price,
                "reference_price": f.reference_price,
                "units": f.units,
                "fee": f.fee,
                "cash_after": f.cash_after,
                "units_after": f.units_after,
                "forced": bool(f.forced),
            }
            for f in result.fills
        ],
        "trades": [
            {
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "units": t.units,
                "entry_fee": t.entry_fee,
                "exit_fee": t.exit_fee,
                "pnl": t.pnl,
            }
            for t in result.trades
        ],
    }


def assert_identical(got: dict, want: dict, label: str) -> None:
    assert len(got["equity"]) == len(want["equity"]), f"{label}: equity length changed"
    for i, (a, b) in enumerate(zip(got["equity"], want["equity"], strict=True)):
        assert a == b, (
            f"{label}: equity diverges at index {i}: {a!r} != {b!r} "
            f"(delta {a - b:.2e})"
        )
    for key in (
        "final_equity", "total_fees", "realized_pnl", "unrealized_pnl", "open_units",
        "skipped_orders", "skipped_costmin", "suppressed_rebalances",
    ):
        assert got[key] == want[key], f"{label}: {key} changed: {got[key]!r} != {want[key]!r}"

    assert len(got["fills"]) == len(want["fills"]), f"{label}: fill count changed"
    for i, (a, b) in enumerate(zip(got["fills"], want["fills"], strict=True)):
        assert a == b, f"{label}: fill {i} changed:\n  got  {a}\n  want {b}"

    assert len(got["trades"]) == len(want["trades"]), f"{label}: trade count changed"
    for i, (a, b) in enumerate(zip(got["trades"], want["trades"], strict=True)):
        assert a == b, f"{label}: trade {i} changed:\n  got  {a}\n  want {b}"


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text())


@pytest.fixture(scope="module")
def candles_by_pair() -> dict:
    store = ParquetStore(DATA_DIR)
    d1 = schema.TIMEFRAMES["1d"]
    out = {}
    for pair in ("XBTEUR", "ETHEUR"):
        full = store.read(schema.PAIRS[pair], d1)
        floor = sustained_floor_date(full, FLOOR_RULE)
        out[pair] = full[
            (full[schema.TIMESTAMP] >= floor)
            & (full[schema.TIMESTAMP] < CONFIG.holdout_ts)
        ].reset_index(drop=True)
    return out


@pytest.mark.parametrize("pair", ["XBTEUR", "ETHEUR"])
def test_walk_forward_oos_is_bit_identical(pair, golden, candles_by_pair):
    """The headline: the concatenated out-of-sample curve did not move."""
    candles = candles_by_pair[pair]
    wf = json.loads(Path(golden["source_walk_forward"]).read_text())
    windows = {w.idx: w for w in make_windows(len(candles), 180, 60, 60)}
    selections = [
        (
            windows[w["idx"]],
            VolTrendParams(
                trend_lookback=w["trend"], vol_lookback=w["vol"], target_vol=w["tvol"]
            ),
        )
        for w in wf["pairs"][pair]["runs"]["main_180_60"]["selections"]["sharpe"]["windows"]
    ]
    result = run_oos_concatenated(
        candles, selections, CONFIG, pair, scheduler=voltrend_scheduler
    )
    assert_identical(
        snapshot(result), golden["pairs"][pair]["walk_forward_oos"], f"{pair} walk-forward"
    )


@pytest.mark.parametrize("pair", ["XBTEUR", "ETHEUR"])
def test_every_static_grid_run_is_bit_identical(pair, golden, candles_by_pair):
    """All 12 grid combinations, covering paths the walk-forward may not hit."""
    candles = candles_by_pair[pair]
    want = golden["pairs"][pair]["static"]
    assert len(want) == 12
    for p in voltrend_grid():
        key = f"{p.trend_lookback}_{p.vol_lookback}_{p.target_vol}"
        result = backtest(candles, lambda c, q=p: voltrend(c, q), CONFIG, pair=pair)
        assert_identical(snapshot(result), want[key], f"{pair} static {key}")


def test_golden_baseline_predates_the_refactor():
    """Guard: the baseline must not be silently regenerated to force a pass."""
    import subprocess

    log = subprocess.run(
        ["git", "log", "--format=%H %s", "--", str(GOLDEN.relative_to(Path.cwd()))],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    assert log, "golden baseline is not committed — it cannot be trusted"
    assert len(log) == 1, (
        f"golden baseline has been modified {len(log)} times; it must be written "
        f"once, before the refactor. History:\n" + "\n".join(log)
    )
