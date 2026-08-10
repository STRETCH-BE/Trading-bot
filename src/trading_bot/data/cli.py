"""Command-line entry point: ``tbot-data`` (or ``python -m trading_bot.data``).

Subcommands:
    download-dump  fetch a Kraken OHLCVT archive from a URL (incl. Google Drive)
    ingest         load dump CSVs (zip / directory / single csv) into parquet
    update         top up from the ccxt Kraken REST API
    validate       run data-quality checks; exit 1 on errors
    report         print coverage + quality summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trading_bot.data import dumps, schema
from trading_bot.data.report import render
from trading_bot.data.store import ParquetStore
from trading_bot.data.update import GapError, make_kraken_exchange, update
from trading_bot.data.validate import validate_store

DEFAULT_DATA_DIR = Path("data/parquet")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tbot-data", description=__doc__)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("download-dump", help="download a Kraken OHLCVT archive")
    p.add_argument("url", help="direct or Google Drive URL of the quarterly archive")
    p.add_argument("dest", type=Path, help="where to save the archive")
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("ingest", help="ingest dump CSVs into the parquet store")
    p.add_argument("source", type=Path, help="zip archive, directory, or single CSV")
    _add_common(p)
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("update", help="top up from the ccxt Kraken REST API")
    _add_common(p)
    p.add_argument(
        "--allow-gap",
        action="store_true",
        help="append even if REST history no longer reaches back to the stored data",
    )
    p.set_defaults(func=_cmd_update)

    p = sub.add_parser("validate", help="run data-quality checks (exit 1 on errors)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("report", help="print coverage and quality summary")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("audit", help="full data-quality audit incl. gap runs")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--top-gaps", type=int, default=10)
    p.set_defaults(func=_cmd_audit)

    return parser


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument(
        "--pair",
        action="append",
        choices=sorted(schema.PAIRS),
        help="restrict to a pair (repeatable; default: all)",
    )
    p.add_argument(
        "--timeframe",
        action="append",
        choices=sorted(schema.TIMEFRAMES),
        help="restrict to a timeframe (repeatable; default: all)",
    )


def _selected(args: argparse.Namespace) -> tuple[list[schema.Pair], list[schema.Timeframe]]:
    pairs = [schema.PAIRS[p] for p in (args.pair or schema.PAIRS)]
    timeframes = [schema.TIMEFRAMES[t] for t in (args.timeframe or schema.TIMEFRAMES)]
    return pairs, timeframes


def _cmd_download(args: argparse.Namespace) -> int:
    path = dumps.download_dump(args.url, args.dest)
    print(f"saved {path} ({path.stat().st_size:,} bytes)")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    store = ParquetStore(args.data_dir)
    pairs, timeframes = _selected(args)
    results = dumps.ingest(args.source, store, pairs, timeframes)
    if not results:
        print("no matching dump files found in source", file=sys.stderr)
        return 1
    for (pair, timeframe), added in sorted(results.items()):
        print(f"{pair}/{timeframe}: +{added} candles")
    return 0


def _cmd_update(args: argparse.Namespace) -> int:
    store = ParquetStore(args.data_dir)
    exchange = make_kraken_exchange()
    pairs, timeframes = _selected(args)
    status = 0
    for pair in pairs:
        for timeframe in timeframes:
            try:
                result = update(
                    store, exchange, pair, timeframe, allow_gap=args.allow_gap
                )
            except GapError as exc:
                print(f"GAP: {exc}", file=sys.stderr)
                status = 1
                continue
            print(
                f"{result.pair}/{result.timeframe}: fetched {result.fetched}, "
                f"appended {result.appended}, now at {result.last_after}"
            )
            for warning in result.warnings:
                print(f"  warning: {warning}", file=sys.stderr)
    return status


def _cmd_validate(args: argparse.Namespace) -> int:
    store = ParquetStore(args.data_dir)
    reports = validate_store(store)
    print(render(reports))
    if not reports:
        return 1  # nothing to validate is a failure for CI purposes
    return 0 if all(rep.ok for rep in reports) else 1


def _cmd_audit(args: argparse.Namespace) -> int:
    from trading_bot.data.audit import audit_store, render

    store = ParquetStore(args.data_dir)
    audits = audit_store(store)
    print(render(audits, top_gaps=args.top_gaps))
    return 0 if audits and all(a.integrity_ok for a in audits) else 1


def _cmd_report(args: argparse.Namespace) -> int:
    store = ParquetStore(args.data_dir)
    reports = validate_store(store)
    print(render(reports))
    return 0
