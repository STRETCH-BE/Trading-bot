#!/usr/bin/env python3
"""Stage 4a: rolling walk-forward validation of the Donchian breakout.

Boundaries:
- start: the approved liquidity floor (trailing 30d median trades >= 500,
  <=1 missing daily candle / 30d, sustained), computed per pair from data;
- end: the last candle strictly before backtest.holdout_start.

Both are asserted before anything runs. The engine itself raises on holdout
data; the --unlock-holdout flag exists, is OFF by default, and is not used
by this stage.

Randomness: the ONLY stochastic element is the shuffled-returns control,
seeded per pair (printed). The grid, selection, and windows are deterministic.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot.backtest import BacktestConfig, buy_and_hold_equity, compute_metrics
from trading_bot.backtest.holdout import HoldoutUnlock
from trading_bot.backtest.metrics import Metrics
from trading_bot.backtest.walkforward import (
    GridPoint,
    default_grid,
    make_windows,
    parameter_stability,
    per_window_outcomes,
    run_oos_concatenated,
    search_window,
)
from trading_bot.data import schema
from trading_bot.data.liquidity import FloorRule, sustained_floor_date
from trading_bot.data.store import ParquetStore

FLOOR_RULE = FloorRule(min_median_trades=500.0, max_missing=1, window_days=30)
SHUFFLE_SEEDS = {"XBTEUR": 42, "ETHEUR": 43}
COMBOS = default_grid()

CONFIG = BacktestConfig(
    starting_capital=10_000.0,
    maker_fee_bps=16.0,
    taker_fee_bps=26.0,
    fee_mode="taker",  # next-open fills are marketable; maker never applies
    slippage_bps=5.0,
    min_rebalance_delta=0.05,
    periods_per_year=365.0,
)

WINDOW_CONFIGS = [("main_180_60", 180, 60), ("w_90_30", 90, 30), ("w_360_90", 360, 90)]


def _search_job(args):
    tag, widx, is_df, pair = args
    pts = search_window(is_df, COMBOS, CONFIG, pair)
    return tag, widx, [(p.sharpe, p.total_return) for p in pts]


def shuffled_candles(candles: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute daily returns (and candle shapes) — same distribution, no order."""
    rng = np.random.default_rng(seed)
    df = candles.reset_index(drop=True).copy()
    o = df[schema.OPEN].to_numpy(float)
    h = df[schema.HIGH].to_numpy(float)
    low = df[schema.LOW].to_numpy(float)
    c = df[schema.CLOSE].to_numpy(float)
    n = len(df)

    rets = c[1:] / c[:-1]
    shape_h = (h / np.maximum(o, c))[1:]
    shape_l = (low / np.minimum(o, c))[1:]
    perm = rng.permutation(n - 1)

    new_c = np.empty(n)
    new_c[0] = c[0]
    for i, r in enumerate(rets[perm]):
        new_c[i + 1] = new_c[i] * r
    new_o = np.concatenate([[o[0]], new_c[:-1]])
    new_h = np.maximum(new_o, new_c)
    new_l = np.minimum(new_o, new_c)
    new_h[1:] *= shape_h[perm]
    new_l[1:] *= shape_l[perm]

    df[schema.OPEN] = new_o
    df[schema.HIGH] = np.maximum(new_h, np.maximum(new_o, new_c))
    df[schema.LOW] = np.minimum(new_l, np.minimum(new_o, new_c))
    df[schema.CLOSE] = new_c
    return df


def metrics_dict(m: Metrics) -> dict:
    d = asdict(m)
    for k, v in d.items():
        if isinstance(v, pd.Timestamp):
            d[k] = v.isoformat()
        elif isinstance(v, pd.Timedelta):
            d[k] = v.total_seconds() / 86400.0
        elif isinstance(v, (np.floating, np.integer)):
            d[k] = float(v)
    return d


def run_pipeline(pair: str, candles: pd.DataFrame, is_len: int, oos_len: int,
                 tag: str, pool: ProcessPoolExecutor) -> dict:
    windows = make_windows(len(candles), is_len, oos_len, oos_len)
    jobs = [
        (tag, w.idx, candles.iloc[w.is_lo:w.is_hi].reset_index(drop=True), pair)
        for w in windows
    ]
    stats: dict[int, list[tuple[float, float]]] = {}
    for _tag, widx, pts in pool.map(_search_job, jobs, chunksize=2):
        stats[widx] = pts
        if widx % 20 == 0:
            print(f"    [{pair}/{tag}] window {widx}/{len(windows)}", flush=True)

    out = {"n_windows": len(windows), "selections": {}}
    for metric in ("sharpe", "total_return"):
        key = 0 if metric == "sharpe" else 1
        triples = []
        for w in windows:
            arr = stats[w.idx]
            best = max(range(len(arr)), key=lambda i: arr[i][key])
            triples.append(
                (w, COMBOS[best],
                 GridPoint(COMBOS[best], sharpe=arr[best][0], total_return=arr[best][1]))
            )
        oos = run_oos_concatenated(candles, [(w, p) for w, p, _ in triples], CONFIG, pair)
        outcomes = per_window_outcomes(candles, triples, oos, CONFIG)
        m = compute_metrics(oos, CONFIG)
        stab = parameter_stability([p for _, p, _ in triples])

        bh = buy_and_hold_equity(oos.candles, CONFIG)
        bh_rt = float(bh.iloc[-1]) * (1 - CONFIG.slippage_rate) * (1 - CONFIG.fee_rate)

        is_sharpes = [t[2].sharpe for t in triples]
        mean_is_sharpe = float(np.mean(is_sharpes))
        mean_oos_sharpe = float(np.mean([o.oos_sharpe for o in outcomes]))
        mean_is_ret = float(np.mean([t[2].total_return for t in triples]))
        mean_oos_ret = float(np.mean([o.oos_return for o in outcomes]))

        out["selections"][metric] = {
            "oos_metrics": metrics_dict(m),
            "oos_fills": len(oos.fills),
            "oos_closed_trades": len(oos.trades),
            "oos_suppressed_rebalances": oos.suppressed_rebalances,
            "oos_skipped_orders": oos.skipped_orders,
            "bh_after_round_trip_return": bh_rt / CONFIG.starting_capital - 1.0,
            "mean_is_sharpe_selected": mean_is_sharpe,
            "mean_is_return_selected": mean_is_ret,
            "mean_oos_sharpe_per_window": mean_oos_sharpe,
            "mean_oos_return_per_window": mean_oos_ret,
            "degradation_ratio": (m.sharpe / mean_is_sharpe) if mean_is_sharpe else None,
            "stability": asdict(stab),
            "windows": [
                {
                    "idx": o.spec.idx,
                    "is_start": o.is_start.date().isoformat(),
                    "oos_start": o.oos_start.date().isoformat(),
                    "oos_end": o.oos_end.date().isoformat(),
                    "partial": o.spec.partial,
                    "entry": o.params.entry_lookback,
                    "exit": o.params.exit_lookback,
                    "atr_mult": o.params.atr_multiple,
                    "is_sharpe": round(o.is_sharpe, 3),
                    "oos_return": round(o.oos_return, 5),
                    "oos_sharpe": round(o.oos_sharpe, 3),
                    "oos_fills": o.oos_fills,
                    "carried_in": o.carried_in,
                    "carried_out": o.carried_out,
                }
                for o in outcomes
            ],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/parquet"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--unlock-holdout", action="store_true",
        help="mint a one-shot holdout unlock (NOT used by stage 4a)",
    )
    args = parser.parse_args()

    unlock = HoldoutUnlock() if args.unlock_holdout else None
    if unlock is not None:
        print("WARNING: --unlock-holdout minted a token; stage 4a does not use it.",
              file=sys.stderr)

    git_hash = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    print(f"stage 4a | commit {git_hash} | shuffle seeds {SHUFFLE_SEEDS} | "
          f"grid {len(COMBOS)} combos | capital {CONFIG.starting_capital:,.0f} | "
          f"taker {CONFIG.taker_fee_bps}bps slip {CONFIG.slippage_bps}bps "
          f"delta {CONFIG.min_rebalance_delta}")

    store = ParquetStore(args.data_dir)
    d1 = schema.TIMEFRAMES["1d"]
    results: dict = {
        "meta": {
            "stage": "4a",
            "timestamp_utc": stamp,
            "git_hash": git_hash,
            "shuffle_seeds": SHUFFLE_SEEDS,
            "grid_size": len(COMBOS),
            "grid": {"entry_lookback": "10..60/5", "exit_lookback": "5..30/5",
                     "atr_multiple": "1.5..4.0/0.5", "atr_period": 14},
            "floor_rule": asdict(FLOOR_RULE),
            "config": {k: v for k, v in asdict(CONFIG).items()},
            "window_configs": WINDOW_CONFIGS,
        },
        "pairs": {},
    }

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for pair in ("XBTEUR", "ETHEUR"):
            full = store.read(schema.PAIRS[pair], d1)
            floor = sustained_floor_date(full, FLOOR_RULE)
            assert floor is not None, f"{pair}: liquidity floor never sustained"
            holdout = CONFIG.holdout_ts
            candles = full[
                (full[schema.TIMESTAMP] >= floor) & (full[schema.TIMESTAMP] < holdout)
            ].reset_index(drop=True)

            first, last = candles[schema.TIMESTAMP].iloc[0], candles[schema.TIMESTAMP].iloc[-1]
            assert first >= floor, "floor violated"
            assert last < holdout, "holdout violated"
            print(f"\n== {pair}: floor {floor.date()} -> using {first.date()} .. "
                  f"{last.date()} ({len(candles)} candles), holdout {holdout.date()} untouched")

            pair_out: dict = {
                "floor_date": floor.date().isoformat(),
                "range": [first.date().isoformat(), last.date().isoformat()],
                "n_candles": len(candles),
                "runs": {},
            }
            for tag, is_len, oos_len in WINDOW_CONFIGS:
                print(f"  -- {tag} ({is_len}/{oos_len})", flush=True)
                pair_out["runs"][tag] = run_pipeline(pair, candles, is_len, oos_len, tag, pool)

            print(f"  -- shuffled control (seed {SHUFFLE_SEEDS[pair]})", flush=True)
            shuffled = shuffled_candles(candles, SHUFFLE_SEEDS[pair])
            pair_out["runs"]["shuffled_180_60"] = run_pipeline(
                pair, shuffled, 180, 60, "shuffled", pool
            )
            results["pairs"][pair] = pair_out

    out_path = Path("results") / f"4a_{stamp}_{git_hash}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1, default=_json_default))
    print(f"\nwrote {out_path}")
    return 0


def _json_default(obj):
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


if __name__ == "__main__":
    sys.exit(main())
