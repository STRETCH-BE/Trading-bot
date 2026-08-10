#!/usr/bin/env python3
"""Capture the pre-refactor golden baseline for the Stage 5a equivalence proof.

Records full equity curves, fills and trades for:
  - the 4a walk-forward concatenated out-of-sample run, both pairs
  - all 12 static grid backtests, both pairs (4b)

Floats are stored via json's repr, which round-trips exactly in Python 3, so
the comparison in tests/test_refactor_equivalence.py is bit-exact rather than
merely 8-decimal. Run once BEFORE refactoring; never regenerate to make a
failing test pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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

FLOOR_RULE = FloorRule(min_median_trades=500.0, max_missing=1, window_days=30)
WF = "results/4a_strategy2_voltrend_20260810T230133Z_e611481.json"
OUT = Path("tests/golden/equivalence_baseline.json")

CONFIG = BacktestConfig(
    starting_capital=10_000.0,
    maker_fee_bps=16.0,
    taker_fee_bps=26.0,
    fee_mode="taker",
    slippage_bps=5.0,
    min_rebalance_delta=0.05,
    periods_per_year=365.0,
)


def snapshot(result) -> dict:
    """Everything the engine produced, at full float precision."""
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


def main() -> int:
    wf = json.loads(Path(WF).read_text())
    store = ParquetStore("data/parquet")
    d1 = schema.TIMEFRAMES["1d"]
    out: dict = {"source_walk_forward": WF, "pairs": {}}

    for pair in ("XBTEUR", "ETHEUR"):
        full = store.read(schema.PAIRS[pair], d1)
        floor = sustained_floor_date(full, FLOOR_RULE)
        candles = full[
            (full[schema.TIMESTAMP] >= floor)
            & (full[schema.TIMESTAMP] < CONFIG.holdout_ts)
        ].reset_index(drop=True)

        windows = {w.idx: w for w in make_windows(len(candles), 180, 60, 60)}
        selections = [
            (
                windows[w["idx"]],
                VolTrendParams(
                    trend_lookback=w["trend"],
                    vol_lookback=w["vol"],
                    target_vol=w["tvol"],
                ),
            )
            for w in wf["pairs"][pair]["runs"]["main_180_60"]["selections"]["sharpe"][
                "windows"
            ]
        ]

        oos = run_oos_concatenated(
            candles, selections, CONFIG, pair, scheduler=voltrend_scheduler
        )
        entry: dict = {"walk_forward_oos": snapshot(oos), "static": {}}
        print(f"{pair}: OOS final={oos.final_equity:.8f} fills={len(oos.fills)}")

        for p in voltrend_grid():
            key = f"{p.trend_lookback}_{p.vol_lookback}_{p.target_vol}"
            res = backtest(candles, lambda c, q=p: voltrend(c, q), CONFIG, pair=pair)
            entry["static"][key] = snapshot(res)
        print(f"{pair}: captured {len(entry['static'])} static runs")
        out["pairs"][pair] = entry

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    size_mb = OUT.stat().st_size / 1e6
    print(f"\nwrote {OUT} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
