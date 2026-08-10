#!/usr/bin/env python3
"""Stage 4b: parameter sensitivity for strategy #2 (volatility-targeted trend).

Static backtests — no walk-forward, no selection. Each of the 12 preregistered
parameter combinations is run once over the full non-holdout period, per pair,
with every cost active. Boundaries identical to 4a: liquidity floor T=500 per
pair, end at the last candle before holdout_start.

Report only. Nothing here tunes, and no combination discovered by this script
may be adopted — doing so would fit to the same data 4a was evaluated on.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot.backtest import BacktestConfig, backtest, compute_metrics
from trading_bot.backtest.walkforward import voltrend_grid
from trading_bot.data import schema
from trading_bot.data.liquidity import FloorRule, sustained_floor_date
from trading_bot.data.store import ParquetStore
from trading_bot.strategies.voltrend import voltrend

FLOOR_RULE = FloorRule(min_median_trades=500.0, max_missing=1, window_days=30)
COMBOS = voltrend_grid()
CONFIG = BacktestConfig(
    starting_capital=10_000.0,
    maker_fee_bps=16.0,
    taker_fee_bps=26.0,
    fee_mode="taker",
    slippage_bps=5.0,
    min_rebalance_delta=0.05,
    periods_per_year=365.0,
)

WF_RESULTS = "results/4a_strategy2_voltrend_20260810T230133Z_e611481.json"


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation without scipy."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/parquet"))
    args = ap.parse_args()

    assert len(COMBOS) == 12, f"grid cap is 12, got {len(COMBOS)}"
    git_hash = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    print(f"stage 4b | commit {git_hash} | {len(COMBOS)} static combos | "
          f"taker {CONFIG.taker_fee_bps}bps slip {CONFIG.slippage_bps}bps "
          f"delta {CONFIG.min_rebalance_delta}")

    wf = json.loads(Path(WF_RESULTS).read_text())
    store = ParquetStore(args.data_dir)
    d1 = schema.TIMEFRAMES["1d"]
    out: dict = {
        "meta": {
            "stage": "4b_strategy2_voltrend",
            "timestamp_utc": stamp,
            "git_hash": git_hash,
            "walk_forward_source": WF_RESULTS,
            "config": dict(asdict(CONFIG)),
            "floor_rule": asdict(FLOOR_RULE),
        },
        "pairs": {},
    }

    per_pair_rows: dict[str, list[dict]] = {}

    for pair in ("XBTEUR", "ETHEUR"):
        full = store.read(schema.PAIRS[pair], d1)
        floor = sustained_floor_date(full, FLOOR_RULE)
        assert floor is not None
        candles = full[
            (full[schema.TIMESTAMP] >= floor)
            & (full[schema.TIMESTAMP] < CONFIG.holdout_ts)
        ].reset_index(drop=True)
        first = candles[schema.TIMESTAMP].iloc[0]
        last = candles[schema.TIMESTAMP].iloc[-1]
        assert last < CONFIG.holdout_ts, "holdout violated"
        print(f"\n== {pair}: {first.date()} .. {last.date()} ({len(candles)} candles)")

        rows = []
        for p in COMBOS:
            res = backtest(candles, lambda c, q=p: voltrend(c, q), CONFIG, pair=pair)
            m = compute_metrics(res, CONFIG)
            rows.append({
                "trend_lookback": p.trend_lookback,
                "vol_lookback": p.vol_lookback,
                "target_vol": p.target_vol,
                "total_return": m.total_return,
                "sharpe": m.sharpe,
                "max_drawdown": m.max_drawdown,
                "trades": len(res.trades),
                "fills": len(res.fills),
                "fees": m.total_fees,
                "bh_return": m.buy_and_hold_return,
                "excess": m.excess_return,
            })
        rows.sort(key=lambda r: r["excess"], reverse=True)
        per_pair_rows[pair] = rows

        hdr = (f"{'trend':>6} {'vol':>4} {'tvol':>5} | {'return%':>11} {'sharpe':>7} "
               f"{'maxDD%':>7} {'trades':>7} {'fees':>10} | {'B&H%':>11} {'excess%':>12}")
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(f"{r['trend_lookback']:>6} {r['vol_lookback']:>4} {r['target_vol']:>5.2f} | "
                  f"{r['total_return']*100:>11.1f} {r['sharpe']:>7.2f} "
                  f"{r['max_drawdown']*100:>7.1f} {r['trades']:>7} {r['fees']:>10,.0f} | "
                  f"{r['bh_return']*100:>11.1f} {r['excess']*100:>12.1f}")

        pos = sum(1 for r in rows if r["excess"] > 0)
        print(f"  positive excess: {pos}/{len(rows)} = {pos/len(rows)*100:.0f}%")

        # 1-D marginals, averaging the metric over the other axes
        marg = {}
        for axis in ("trend_lookback", "vol_lookback", "target_vol"):
            marg[axis] = {}
            for val in sorted({r[axis] for r in rows}):
                sub = [r for r in rows if r[axis] == val]
                marg[axis][str(val)] = {
                    "mean_excess": float(np.mean([r["excess"] for r in sub])),
                    "mean_sharpe": float(np.mean([r["sharpe"] for r in sub])),
                    "mean_return": float(np.mean([r["total_return"] for r in sub])),
                    "n": len(sub),
                }
            print(f"  marginal {axis}: " + "  ".join(
                f"{k}->exc {v['mean_excess']*100:+.0f}pp/sh {v['mean_sharpe']:.2f}"
                for k, v in marg[axis].items()))

        # distance from the walk-forward selections
        wsel = wf["pairs"][pair]["runs"]["main_180_60"]["selections"]["sharpe"]["windows"]
        counts: dict[tuple, int] = {}
        for w in wsel:
            key = (w["trend"], w["vol"], w["tvol"])
            counts[key] = counts.get(key, 0) + 1
        best = rows[0]
        best_key = (best["trend_lookback"], best["vol_lookback"], best["target_vol"])
        by_key = {(r["trend_lookback"], r["vol_lookback"], r["target_vol"]): i
                  for i, r in enumerate(rows)}
        wf_ranks = [by_key[k] + 1 for k in counts]
        weighted_rank = sum((by_key[k] + 1) * n for k, n in counts.items()) / sum(counts.values())

        out["pairs"][pair] = {
            "range": [first.date().isoformat(), last.date().isoformat()],
            "n_candles": len(candles),
            "rows": rows,
            "positive_excess_fraction": pos / len(rows),
            "marginals": marg,
            "wf_selection_counts": {str(k): v for k, v in counts.items()},
            "wf_selection_ranks_static": {str(k): by_key[k] + 1 for k in counts},
            "wf_weighted_mean_static_rank": weighted_rank,
            "best_static": best_key,
        }
        print(f"  best static (by excess): trend={best_key[0]} vol={best_key[1]} "
              f"tvol={best_key[2]}  excess={best['excess']*100:+.0f}pp")
        print(f"  walk-forward picked {len(counts)} distinct combos; their static ranks "
              f"{sorted(wf_ranks)} of 12; window-weighted mean rank {weighted_rank:.1f}")

    # cross-pair rank agreement
    keys = [(r["trend_lookback"], r["vol_lookback"], r["target_vol"])
            for r in per_pair_rows["XBTEUR"]]
    x_by = {(r["trend_lookback"], r["vol_lookback"], r["target_vol"]): r
            for r in per_pair_rows["XBTEUR"]}
    e_by = {(r["trend_lookback"], r["vol_lookback"], r["target_vol"]): r
            for r in per_pair_rows["ETHEUR"]}
    for metric in ("excess", "sharpe", "total_return"):
        rho = spearman([x_by[k][metric] for k in keys], [e_by[k][metric] for k in keys])
        out.setdefault("cross_pair_rank_correlation", {})[metric] = rho
        print(f"\ncross-pair Spearman rank correlation ({metric}): {rho:+.3f}")

    x_rank = {k: i for i, k in enumerate(
        sorted(keys, key=lambda k: x_by[k]["excess"], reverse=True))}
    e_rank = {k: i for i, k in enumerate(
        sorted(keys, key=lambda k: e_by[k]["excess"], reverse=True))}
    flips = [(k, x_rank[k] + 1, e_rank[k] + 1) for k in keys
             if abs(x_rank[k] - e_rank[k]) >= 6]
    out["cross_pair_large_rank_flips"] = [
        {"combo": str(k), "xbt_rank": a, "eth_rank": b} for k, a, b in flips
    ]
    print(f"combos flipping >=6 rank places between pairs: {len(flips)}")
    for k, a, b in flips:
        print(f"  {k}: XBT #{a} vs ETH #{b}")

    path = Path("results") / f"4b_strategy2_voltrend_{stamp}_{git_hash}.json"
    path.write_text(json.dumps(out, indent=1, default=_jd))
    print(f"\nwrote {path}")
    return 0


def _jd(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    raise TypeError(type(o).__name__)


if __name__ == "__main__":
    sys.exit(main())
