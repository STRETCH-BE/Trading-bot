#!/usr/bin/env python3
"""Stage 4c: strategy #2 at EUR 100, 1,000 and 10,000.

Replays the EXACT walk-forward parameter schedule selected in 4a — no
re-selection, no re-optimisation — at three account sizes and two fee
schedules, with ordermin, costmin and min_rebalance_delta all active.

The question this answers is executability, not edge: at small size the
exchange's minimum order constraints can be larger than the strategy's own
rebalance dead-band, in which case the simulated strategy is not the one 4a
evaluated.
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

from trading_bot.backtest import BacktestConfig, buy_and_hold_equity, compute_metrics
from trading_bot.backtest.walkforward import (
    make_windows,
    run_oos_concatenated,
    voltrend_scheduler,
)
from trading_bot.data import schema
from trading_bot.data.liquidity import FloorRule, sustained_floor_date
from trading_bot.data.store import ParquetStore
from trading_bot.strategies.voltrend import VolTrendParams

FLOOR_RULE = FloorRule(min_median_trades=500.0, max_missing=1, window_days=30)
WF_RESULTS = "results/4a_strategy2_voltrend_20260810T230133Z_e611481.json"
SIZES = [100.0, 1_000.0, 10_000.0]
FEE_SCHEDULES = {"pro_16_26": (16.0, 26.0), "standard_25_40": (25.0, 40.0)}

BASE = dict(
    fee_mode="taker",
    slippage_bps=5.0,
    min_rebalance_delta=0.05,
    periods_per_year=365.0,
)


def load_selections(wf: dict, pair: str, n_candles: int):
    windows = make_windows(n_candles, 180, 60, 60)
    by_idx = {w.idx: w for w in windows}
    sel = []
    for w in wf["pairs"][pair]["runs"]["main_180_60"]["selections"]["sharpe"]["windows"]:
        params = VolTrendParams(
            trend_lookback=w["trend"], vol_lookback=w["vol"], target_vol=w["tvol"]
        )
        sel.append((by_idx[w["idx"]], params))
    return sel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/parquet"))
    args = ap.parse_args()

    git_hash = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    print(f"stage 4c | commit {git_hash} | walk-forward schedule from {WF_RESULTS}")

    wf = json.loads(Path(WF_RESULTS).read_text())
    store = ParquetStore(args.data_dir)
    d1 = schema.TIMEFRAMES["1d"]
    out: dict = {
        "meta": {
            "stage": "4c_strategy2_voltrend",
            "timestamp_utc": stamp,
            "git_hash": git_hash,
            "walk_forward_source": WF_RESULTS,
            "sizes": SIZES,
            "fee_schedules": FEE_SCHEDULES,
            "base_config": BASE,
        },
        "pairs": {},
    }

    for pair in ("XBTEUR", "ETHEUR"):
        p = schema.PAIRS[pair]
        full = store.read(p, d1)
        floor = sustained_floor_date(full, FLOOR_RULE)
        probe = BacktestConfig(starting_capital=10_000.0, **BASE)
        candles = full[
            (full[schema.TIMESTAMP] >= floor)
            & (full[schema.TIMESTAMP] < probe.holdout_ts)
        ].reset_index(drop=True)
        selections = load_selections(wf, pair, len(candles))
        oos_lo = selections[0][0].oos_lo
        oos = candles.iloc[oos_lo:].reset_index(drop=True)
        med_px = float(oos[schema.CLOSE].median())
        last_px = float(oos[schema.CLOSE].iloc[-1])

        print(f"\n{'='*94}\n{pair}  ordermin={p.ordermin} base units  costmin={p.costmin:.0f} EUR"
              f"  |  median close {med_px:,.0f}  last close {last_px:,.0f}\n{'='*94}")

        # --- executability: the binding minimum, in EUR and as % of equity
        exec_rows = []
        for size in SIZES:
            om_eur_med = p.ordermin * med_px
            om_eur_last = p.ordermin * last_px
            binding_med = max(om_eur_med, p.costmin)
            print(f"  at EUR {size:>8,.0f}: ordermin={om_eur_med:>8,.2f} EUR (median px) / "
                  f"{om_eur_last:>8,.2f} (last px), costmin={p.costmin:.2f} EUR "
                  f"-> binding {binding_med:>8,.2f} EUR = {binding_med/size*100:>7.2f}% of equity"
                  f"   vs min_rebalance_delta 5.00%")
            exec_rows.append({
                "size": size,
                "ordermin_eur_median_px": om_eur_med,
                "ordermin_eur_last_px": om_eur_last,
                "costmin_eur": p.costmin,
                "binding_min_eur": binding_med,
                "binding_pct_of_equity": binding_med / size,
                "exceeds_rebalance_delta": (binding_med / size) > 0.05,
            })

        pair_out: dict = {
            "ordermin": p.ordermin,
            "costmin": p.costmin,
            "oos_range": [str(oos[schema.TIMESTAMP].iloc[0].date()),
                          str(oos[schema.TIMESTAMP].iloc[-1].date())],
            "executability": exec_rows,
            "runs": {},
        }

        hdr = (f"  {'fees':<15} {'size':>9} {'fills':>6} {'blocked':>8} {'blk%':>6} "
               f"{'ordmin':>7} {'costmin':>8} {'supp':>6} {'trades':>7} "
               f"{'feesEUR':>10} {'fees%cap':>9} {'final':>13} {'B&H':>13}")
        print(f"\n{hdr}\n  {'-'*(len(hdr)-2)}")

        for fee_name, (maker, taker) in FEE_SCHEDULES.items():
            for size in SIZES:
                cfg = BacktestConfig(
                    starting_capital=size, maker_fee_bps=maker, taker_fee_bps=taker, **BASE
                )
                res = run_oos_concatenated(
                    candles, selections, cfg, pair, scheduler=voltrend_scheduler
                )
                m = compute_metrics(res, cfg)
                bh = buy_and_hold_equity(res.candles, cfg)
                intended = len(res.fills) + res.skipped_orders
                blocked_pct = res.skipped_orders / intended if intended else 0.0
                ordmin_only = res.skipped_orders - res.skipped_costmin

                # unconstrained twin: same schedule, no exchange minimums
                unc_cfg = BacktestConfig(
                    starting_capital=size, maker_fee_bps=maker, taker_fee_bps=taker,
                    min_order_units=0.0, costmin=0.0, **BASE
                )
                unc = run_oos_concatenated(
                    candles, selections, unc_cfg, None, scheduler=voltrend_scheduler
                )
                unc_m = compute_metrics(unc, unc_cfg)

                print(f"  {fee_name:<15} {size:>9,.0f} {len(res.fills):>6} "
                      f"{res.skipped_orders:>8} {blocked_pct*100:>5.1f}% {ordmin_only:>7} "
                      f"{res.skipped_costmin:>8} {res.suppressed_rebalances:>6} "
                      f"{len(res.trades):>7} {m.total_fees:>10,.0f} "
                      f"{m.fees_pct_of_capital*100:>8.0f}% {m.final_equity:>13,.2f} "
                      f"{float(bh.iloc[-1]):>13,.2f}")

                pair_out["runs"][f"{fee_name}_{int(size)}"] = {
                    "fee_schedule": fee_name, "size": size,
                    "fills": len(res.fills),
                    "blocked_orders": res.skipped_orders,
                    "blocked_by_ordermin": ordmin_only,
                    "blocked_by_costmin": res.skipped_costmin,
                    "intended_orders": intended,
                    "blocked_pct_of_intended": blocked_pct,
                    "suppressed_by_deadband": res.suppressed_rebalances,
                    "trades": len(res.trades),
                    "total_fees": m.total_fees,
                    "fees_pct_of_starting_capital": m.fees_pct_of_capital,
                    "final_equity": m.final_equity,
                    "total_return": m.total_return,
                    "sharpe": m.sharpe,
                    "max_drawdown": m.max_drawdown,
                    "buy_and_hold_final": float(bh.iloc[-1]),
                    "buy_and_hold_return": m.buy_and_hold_return,
                    "excess_return": m.excess_return,
                    "unconstrained_final_equity": unc_m.final_equity,
                    "unconstrained_total_return": unc_m.total_return,
                    "unconstrained_sharpe": unc_m.sharpe,
                    "unconstrained_fills": len(unc.fills),
                    "constraint_drag_vs_unconstrained": (
                        m.final_equity / unc_m.final_equity - 1.0
                    ),
                }

        print("\n  constrained vs UNCONSTRAINED (same schedule, exchange minimums removed):")
        for key, r in pair_out["runs"].items():
            print(f"    {key:<24} fills {r['fills']:>5} vs {r['unconstrained_fills']:>5}  "
                  f"final {r['final_equity']:>13,.2f} vs {r['unconstrained_final_equity']:>13,.2f}  "
                  f"drag {r['constraint_drag_vs_unconstrained']*100:>+8.2f}%")

        out["pairs"][pair] = pair_out

    path = Path("results") / f"4c_strategy2_voltrend_{stamp}_{git_hash}.json"
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
