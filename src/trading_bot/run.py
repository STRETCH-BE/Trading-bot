#!/usr/bin/env python3
"""Entry point. This is what systemd points at.

Defaults to PAPER in every code path. ``--live`` requires BOTH the flag and
``LIVE_TRADING=yes`` in the environment, and as of Stage 6c there is no live
broker to construct, so it refuses with an explicit message rather than
falling back to paper silently.

Flags: --paper (default) --live --dry-run --reconcile-only --once
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import signal
import sys
import time
from pathlib import Path

import pandas as pd

from trading_bot.backtest.config import BacktestConfig
from trading_bot.broker import PaperBroker, PaperBrokerConfig, StateStore
from trading_bot.data import schema
from trading_bot.data.store import ParquetStore
from trading_bot.execution import FillModel
from trading_bot.risk import RiskGate, RiskLimits
from trading_bot.risk.budget import check_allocation_budget, describe_budget
from trading_bot.scheduler.cycle import CycleContext, CycleHalted, run_cycle
from trading_bot.scheduler.reconcile import reconcile
from trading_bot.strategies.voltrend import VolTrendParams, voltrend

log = logging.getLogger("trading_bot.run")

_shutdown_requested = False


def _handle_sigterm(signum, frame) -> None:
    """Graceful: finish the current cycle, start no new one."""
    global _shutdown_requested
    _shutdown_requested = True
    log.warning("SIGTERM received — finishing the current cycle, then stopping")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trading-bot", description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--paper", action="store_true", help="paper trading (default)")
    mode.add_argument("--live", action="store_true", help="LIVE trading (guarded)")
    p.add_argument("--dry-run", action="store_true",
                   help="run the full cycle, log the order, submit nothing")
    p.add_argument("--reconcile-only", action="store_true",
                   help="reconcile and exit without trading")
    p.add_argument("--once", action="store_true", help="one cycle, then exit")
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--data-dir", type=Path, default=Path("data/parquet"))
    p.add_argument("--state-dir", type=Path, default=Path("data/state"))
    p.add_argument("--pair", action="append", choices=sorted(schema.PAIRS))
    p.add_argument("--interval-seconds", type=float, default=86_400.0)
    p.add_argument("--settle-delay-seconds", type=float, default=60.0,
                   help="wait after candle close before acting, so it finalises")
    p.add_argument("--halt-file", type=Path, default=Path("HALT"))
    p.add_argument("--log-level", default="INFO")
    return p


def resolve_mode(args) -> str:
    """PAPER unless BOTH --live and LIVE_TRADING=yes. Never silently live."""
    if not args.live:
        return "paper"
    if os.environ.get("LIVE_TRADING") != "yes":
        raise SystemExit(
            "--live requires LIVE_TRADING=yes in the environment. Refusing to "
            "start: the flag alone is not enough, by design."
        )
    raise SystemExit(
        "--live was requested and authorised, but no live broker exists yet "
        "(Stage 6b is not implemented). Refusing to fall back to paper "
        "silently — that would look like live trading and would not be."
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    mode = resolve_mode(args)

    # FINDING 6: raises if the path is wrong; prints what is actually in use.
    config = BacktestConfig.load(args.config)
    limits = RiskLimits.load(args.config)
    params = VolTrendParams.load(args.config)
    print(config.banner(args.config))
    print(f"  mode               : {mode.upper()}"
          f"{'  (DRY RUN)' if args.dry_run else ''}")
    print(f"  strategy           : voltrend {params}")
    print(f"  risk limits        : {limits}")
    print(describe_budget(config, limits))
    # Refuse to start if a full strategy signal could never be approved.
    check_allocation_budget(config, limits).raise_if_violated()
    print(f"  halt file          : {args.halt_file.resolve()}")
    print("=" * 68)

    args.state_dir.mkdir(parents=True, exist_ok=True)
    local = StateStore(args.state_dir / "local.db")
    venue = StateStore(args.state_dir / "paper_venue.db")  # the simulated venue
    parquet = ParquetStore(args.data_dir)
    pairs = tuple(args.pair or ["XBTEUR"])
    fill_model = FillModel.from_config(config)

    def price_of(pair: str) -> float:
        df = parquet.read(schema.PAIRS[pair], schema.TIMEFRAMES["1d"])
        return float(df[schema.CLOSE].iloc[-1])

    broker = PaperBroker(
        venue, fill_model, price_of,
        config=PaperBrokerConfig(), rng=random.Random(),
    )
    if local.get_cash() == 0.0 and not local.all_orders():
        local.set_cash(config.starting_capital)
        venue.set_cash(config.starting_capital)

    if args.reconcile_only:
        report = reconcile(local, broker)
        print(report.render())
        return 0 if report.agrees else 1

    ctx = CycleContext(
        local=local, broker=broker,
        gate=RiskGate(limits, fill_model, halt_file=args.halt_file),
        fill_model=fill_model, config=config, strategy_name="voltrend",
        signal_fn=lambda c: voltrend(c, params),
        candles_for=lambda pair: parquet.read(
            schema.PAIRS[pair], schema.TIMEFRAMES["1d"]
        ),
        pairs=pairs, dry_run=args.dry_run,
    )

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    first = True
    while True:
        try:
            run_cycle(ctx, now=pd.Timestamp.now(tz="UTC"), reconcile_first=first)
        except CycleHalted as exc:
            log.critical("HALTED: %s", exc)
            return 2
        except Exception:
            # Any unhandled exception halts. Never continue in an unknown state.
            log.exception("unhandled exception in cycle — halting")
            ctx.gate.engage_halt("unhandled exception in cycle")
            return 3
        first = False

        if args.once or _shutdown_requested:
            log.info("stopping (%s)", "--once" if args.once else "SIGTERM")
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    sys.exit(main())
