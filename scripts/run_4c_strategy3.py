"""Stage 4 gate applied to STRATEGY #3: mean reversion (Amendment 6).

The final preregistration slot. Everything here was frozen in Amendment 6
BEFORE this script was written: grid (3x2x2 = 12), windows (180/60, 90/30,
360/90), selection by in-sample Sharpe on the REPAIRED selection layer,
allocation 0.25, criterion 1 against the CAPITAL-MATCHED benchmark
(Amendment 4), shuffle seeds 42/43, holdout sealed regardless of outcome.

Running this consumes slot 2 of 2. There is no slot 3.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot.backtest import BacktestConfig, buy_and_hold_equity, compute_metrics
from trading_bot.backtest.engine import capital_matched_hold_equity
from trading_bot.backtest.metrics import Metrics
from trading_bot.backtest.walkforward import (
    GridPoint,
    make_windows,
    meanrev_grid,
    meanrev_scheduler,
    parameter_stability,
    per_window_outcomes,
    run_oos_concatenated,
    search_window,
)
from trading_bot.data import schema
from trading_bot.data.liquidity import FloorRule, sustained_floor_date
from trading_bot.data.store import ParquetStore
from trading_bot.strategies.meanrev import meanrev

FLOOR_RULE = FloorRule(min_median_trades=500.0, max_missing=1, window_days=30)
SHUFFLE_SEEDS = {"XBTEUR": 42, "ETHEUR": 43}
COMBOS = meanrev_grid()
ALLOCATION = 0.25  # frozen by Amendment 6 — deliberately NOT a CLI flag

CONFIG = BacktestConfig(
    starting_capital=10_000.0,
    maker_fee_bps=16.0,
    taker_fee_bps=26.0,
    fee_mode="taker",
    slippage_bps=5.0,
    min_rebalance_delta=0.05,
    periods_per_year=365.0,
    strategy_max_allocation=ALLOCATION,
)

WINDOW_CONFIGS = [("main_180_60", 180, 60), ("w_90_30", 90, 30), ("w_360_90", 360, 90)]

# Amendment 6 criteria thresholds
SHUFFLE_MARGIN = 0.4
DEGRADATION_MIN = 0.5
STABILITY_JUMP_FRACTION = 0.30
MEANREV_GRID_STEPS = {"lookback": 10, "entry_k": 0.5, "exit_e": 0.5}


def _verify_preconditions() -> None:
    """Amendment 6: if either repair is absent, the run is VOID."""
    assert hasattr(GridPoint(None, 0.0, 0.0), "eligible"), (
        "selection layer not repaired — run is void"
    )
    from trading_bot.backtest.walkforward import NoEvaluableComboError  # noqa: F401

    m_fields = {f.name for f in Metrics.__dataclass_fields__.values()}
    assert "excess_vs_capital_matched" in m_fields, (
        "capital-matched benchmark not implemented — run is void"
    )


def _meanrev_signal(candles, params):
    return meanrev(candles, params)


def _search_job(args):
    tag, widx, is_df, pair, allocation = args
    assert CONFIG.strategy_max_allocation == allocation, (
        f"worker allocation drift: worker has {CONFIG.strategy_max_allocation}, "
        f"parent intended {allocation}"
    )
    pts = search_window(is_df, COMBOS, CONFIG, pair, signal_factory=_meanrev_signal)
    return tag, widx, [(p.sharpe, p.total_return, p.eligible) for p in pts]


def shuffled_candles(candles: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute daily returns (and candle shapes) — same distribution, no order.

    Identical construction to the strategy #1 and #2 controls.
    """
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
        (tag, w.idx, candles.iloc[w.is_lo:w.is_hi].reset_index(drop=True), pair,
         CONFIG.strategy_max_allocation)
        for w in windows
    ]
    stats: dict[int, list[tuple[float, float, bool]]] = {}
    for _tag, widx, pts in pool.map(_search_job, jobs, chunksize=2):
        stats[widx] = pts
        if widx % 20 == 0:
            print(f"    [{pair}/{tag}] window {widx}/{len(windows)}", flush=True)

    out = {
        "n_windows": len(windows),
        # Amendment 6 / instruction: per-combo, per-window numbers. Grid order
        # matches meanrev_grid(); entries are [is_sharpe, is_return, eligible].
        "per_window_grid_scores": {
            str(w.idx): [[s, r, e] for s, r, e in stats[w.idx]] for w in windows
        },
        "selections": {},
    }
    for metric in ("sharpe", "total_return"):
        key = 0 if metric == "sharpe" else 1
        triples = []
        for w in windows:
            arr = stats[w.idx]
            best = max(range(len(arr)), key=lambda i: arr[i][key])
            assert math.isfinite(arr[best][0]), (
                "selection returned a non-finite score — the repaired layer "
                "should have made this impossible; run is void"
            )
            triples.append(
                (w, COMBOS[best],
                 GridPoint(COMBOS[best], sharpe=arr[best][0], total_return=arr[best][1]))
            )
        oos = run_oos_concatenated(
            candles, [(w, p) for w, p, _ in triples], CONFIG, pair,
            scheduler=meanrev_scheduler,
        )
        outcomes = per_window_outcomes(candles, triples, oos, CONFIG)
        m = compute_metrics(oos, CONFIG)
        stab = parameter_stability([p for _, p, _ in triples], MEANREV_GRID_STEPS)

        bh = buy_and_hold_equity(oos.candles, CONFIG)
        cm = capital_matched_hold_equity(oos.candles, CONFIG)

        is_sharpes = [t[2].sharpe for t in triples]
        mean_is_sharpe = float(np.mean(is_sharpes))

        out["selections"][metric] = {
            "oos_metrics": metrics_dict(m),
            "oos_fills": len(oos.fills),
            "oos_closed_trades": len(oos.trades),
            "oos_suppressed_rebalances": oos.suppressed_rebalances,
            "oos_skipped_orders": oos.skipped_orders,
            "bh_uncapped_final": float(bh.iloc[-1]),
            "capital_matched_final": float(cm.iloc[-1]),
            "mean_is_sharpe_selected": mean_is_sharpe,
            "mean_is_return_selected": float(np.mean([t[2].total_return for t in triples])),
            "mean_oos_sharpe_per_window": float(np.mean([o.oos_sharpe for o in outcomes])),
            "mean_oos_return_per_window": float(np.mean([o.oos_return for o in outcomes])),
            "degradation_ratio": (m.sharpe / mean_is_sharpe) if mean_is_sharpe else None,
            "stability": asdict(stab),
            "windows": [
                {
                    "idx": o.spec.idx,
                    "is_start": o.is_start.isoformat(),
                    "oos_start": o.oos_start.isoformat(),
                    "oos_end": o.oos_end.isoformat(),
                    "partial": o.spec.partial,
                    "lookback": o.params.lookback,
                    "entry_k": o.params.entry_k,
                    "exit_e": o.params.exit_e,
                    "is_sharpe": o.is_sharpe,
                    "oos_sharpe": o.oos_sharpe,
                    "oos_return": o.oos_return,
                    "oos_fills": o.oos_fills,
                    "carried_in": o.carried_in,
                    "carried_out": o.carried_out,
                }
                for o in outcomes
            ],
        }
    return out


def btc_reference_return(store: ParquetStore, start, end) -> float:
    """Uncapped buy-and-hold BTC over [start, end] — REPORTED, never the gate."""
    d1 = schema.TIMEFRAMES["1d"]
    btc = store.read(schema.PAIRS["XBTEUR"], d1)
    btc = btc[(btc[schema.TIMESTAMP] >= start) & (btc[schema.TIMESTAMP] <= end)]
    btc = btc.reset_index(drop=True)
    if len(btc) < 2:
        return 0.0
    curve = buy_and_hold_equity(btc, CONFIG)
    return float(curve.iloc[-1]) / CONFIG.starting_capital - 1.0


def eur100_executability() -> dict:
    """The EUR 100 finding, restated against this strategy (binary, so the
    dead-band granularity problem does not apply — but ordermin does)."""
    prices = {"XBTEUR": 74_500.10, "ETHEUR": 2_525.99}  # last closes, 2025-12-31
    fee, slip = CONFIG.fee_rate, CONFIG.slippage_rate
    out = {"note": (
        "prices are the last closes in the data (2025-12-31); the sandbox has "
        "no route to live quotes. Binary long-or-flat needs no granularity, "
        "so only openability binds."
    )}
    for pair, px in prices.items():
        om = schema.min_order_units(pair)
        budget = 100.0 * ALLOCATION
        units = (budget / (1 + fee)) / (px * (1 + slip))
        out[pair] = {
            "full_position_eur": budget,
            "units_at_full_position": units,
            "ordermin_units": om,
            "openable_at_eur100": units >= om,
            "min_capital_to_open_eur": om * px * (1 + slip) * (1 + fee) / ALLOCATION,
        }
    return out


def evaluate_criteria(results: dict) -> dict:
    """Amendment 6's six criteria, evaluated mechanically. PASS needs 2-5."""
    crit: dict = {"per_pair": {}, "definitions": {
        "1_reporting": "capital-matched benchmark and uncapped B&H BTC present on every report",
        "2_shuffle": f"main OOS Sharpe - shuffled OOS Sharpe >= {SHUFFLE_MARGIN}",
        "3_degradation": f"main OOS Sharpe / mean selected IS Sharpe > {DEGRADATION_MIN}",
        "4_excess": "main concatenated OOS excess vs CAPITAL-MATCHED benchmark > 0",
        "5_windows": "criterion 4 holds in >= 2 of 3 window configurations",
        "6_stability": (
            f"<{STABILITY_JUMP_FRACTION:.0%} of transitions move a parameter "
            f">2 grid steps — VACUOUS on a 3x2x2 grid (no axis can move >2 "
            f"steps); reported descriptively, contributes nothing to PASS"
        ),
    }}
    for pair, p in results["pairs"].items():
        sel = {t: p["runs"][t]["selections"]["sharpe"] for t in p["runs"]}
        main = sel["main_180_60"]
        shuf = sel["shuffled_180_60"]

        margin = main["oos_metrics"]["sharpe"] - shuf["oos_metrics"]["sharpe"]
        degr = main["degradation_ratio"]
        excess_cm = main["oos_metrics"]["excess_vs_capital_matched"]
        excess_by_config = {
            t: sel[t]["oos_metrics"]["excess_vs_capital_matched"]
            for t in ("main_180_60", "w_90_30", "w_360_90")
        }
        positive_configs = sum(1 for v in excess_by_config.values() if v > 0)
        stab = main["stability"]
        jump_fraction = (
            sum(stab["jumps_gt_2_steps"].values()) / stab["n_transitions"]
            if stab["n_transitions"] else 0.0
        )
        crit["per_pair"][pair] = {
            "2_shuffle": {"margin": margin, "pass": margin >= SHUFFLE_MARGIN},
            "3_degradation": {"ratio": degr, "pass": degr is not None and degr > DEGRADATION_MIN},
            "4_excess_vs_capital_matched": {"value": excess_cm, "pass": excess_cm > 0},
            "5_windows": {
                "excess_by_config": excess_by_config,
                "positive_configs": positive_configs,
                "pass": positive_configs >= 2,
            },
            "6_stability": {
                "jump_fraction_gt2steps": jump_fraction,
                "boundaries_with_any_change": stab["n_changed"],
                "n_transitions": stab["n_transitions"],
                "verdict": "VACUOUS-PASS (uninformative on this grid)",
            },
        }
    both = crit["per_pair"]
    crit["overall"] = {
        "pass": all(
            both[pair][c]["pass"]
            for pair in both
            for c in ("2_shuffle", "3_degradation", "4_excess_vs_capital_matched", "5_windows")
        ),
        "rule": "PASS requires criteria 2, 3, 4, 5 on BOTH pairs; 6 is vacuous; 1 is reporting",
    }
    return crit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/parquet"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    _verify_preconditions()

    git_hash = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    print(f"stage 4 gate STRATEGY #3 meanrev | commit {git_hash} | "
          f"grid {len(COMBOS)} combos | allocation {ALLOCATION} "
          f"(signal 1.0 -> {ALLOCATION*100:.0f}% of equity) | "
          f"benchmark CAPITAL-MATCHED (Amendment 4) | "
          f"taker {CONFIG.taker_fee_bps}bps slip {CONFIG.slippage_bps}bps | "
          f"shuffle seeds {SHUFFLE_SEEDS}")
    print("THIS RUN CONSUMES SLOT 2 OF 2. There is no slot 3.", file=sys.stderr)

    store = ParquetStore(args.data_dir)
    d1 = schema.TIMEFRAMES["1d"]
    results: dict = {
        "meta": {
            "stage": "4_gate_strategy3_meanrev",
            "timestamp_utc": stamp,
            "git_hash": git_hash,
            "amendment": 6,
            "slot": "2 of 2 — CONSUMED by this run",
            "selection_layer": "repaired: warmup-ineligible combos carry -inf (FINDINGS §3 fix)",
            "benchmark": "capital-matched: alloc x B&H + (1-alloc) cash (Amendment 4)",
            "allocation": ALLOCATION,
            "shuffle_seeds": SHUFFLE_SEEDS,
            "grid_size": len(COMBOS),
            "grid": {"lookback": [10, 20, 30], "entry_k": [1.0, 1.5], "exit_e": [0.0, 0.5]},
            "floor_rule": asdict(FLOOR_RULE),
            "config": dict(asdict(CONFIG).items()),
            "window_configs": WINDOW_CONFIGS,
            "eur100_executability": eur100_executability(),
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
            assert first >= floor and last < holdout, "floor/holdout violated"
            print(f"\n== {pair}: floor {floor.date()} -> {first.date()} .. {last.date()} "
                  f"({len(candles)} candles), holdout {holdout.date()} untouched")

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

            # criterion 1 reporting: uncapped B&H BTC over this pair's OOS span
            main_windows = pair_out["runs"]["main_180_60"]["selections"]["sharpe"]["windows"]
            oos_start = pd.Timestamp(main_windows[0]["oos_start"])
            oos_end = pd.Timestamp(main_windows[-1]["oos_end"])
            pair_out["bh_btc_uncapped_return_same_span"] = btc_reference_return(
                store, oos_start, oos_end
            )
            results["pairs"][pair] = pair_out

    results["criteria"] = evaluate_criteria(results)

    out_path = Path("results") / f"4c_strategy3_meanrev_{stamp}_{git_hash}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1, default=_json_default))

    print("\n" + "=" * 72)
    verdict = results["criteria"]
    for pair, c in verdict["per_pair"].items():
        print(f"{pair}:")
        print(f"  2 shuffle margin      : {c['2_shuffle']['margin']:+.3f}  "
              f"{'PASS' if c['2_shuffle']['pass'] else 'FAIL'}")
        print(f"  3 degradation         : {c['3_degradation']['ratio']:.3f}   "
              f"{'PASS' if c['3_degradation']['pass'] else 'FAIL'}")
        print(f"  4 excess vs cap-match : {c['4_excess_vs_capital_matched']['value']:+.4f}  "
              f"{'PASS' if c['4_excess_vs_capital_matched']['pass'] else 'FAIL'}")
        print(f"  5 window robustness   : {c['5_windows']['positive_configs']}/3 positive  "
              f"{'PASS' if c['5_windows']['pass'] else 'FAIL'}")
        print(f"  6 stability           : {c['6_stability']['verdict']}")
    print("=" * 72)
    print(f"GATE VERDICT: {'PASS' if verdict['overall']['pass'] else 'FAIL'}")
    print(f"wrote {out_path}")
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
