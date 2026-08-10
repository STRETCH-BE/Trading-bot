#!/usr/bin/env python3
"""Regenerate the deterministic test fixtures in tests/fixtures/.

Produces files in the exact wire formats the data layer consumes:

- ``<PAIR>_<minutes>.csv``  — Kraken OHLCVT dump format: no header,
  ``timestamp_s,open,high,low,close,volume,trades``.
- ``ccxt_kraken_xbteur_1h.json`` — ccxt ``fetch_ohlcv`` rows:
  ``[timestamp_ms, open, high, low, close, volume]``, continuing seamlessly
  after the end of the XBTEUR 1h dump fixture.

The values are a seeded random walk (synthetic, not market data); the formats
are what the parsers are tested against. See tests/fixtures/README.md.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

START = datetime(2024, 1, 1, tzinfo=UTC)

SPECS = [
    # (pair, start price, hourly rows, daily rows)
    ("XBTEUR", 40_000.0, 48, 30),
    ("ETHEUR", 2_100.0, 48, 30),
]

API_ROWS = 12  # hourly candles served by the fake exchange after the dump end


def make_candles(
    rng: random.Random, start: datetime, step: timedelta, count: int, price: float
) -> list[tuple[int, float, float, float, float, float, int]]:
    rows = []
    for i in range(count):
        ts = start + i * step
        drift = rng.gauss(0, price * 0.004)
        open_, close = price, max(price + drift, price * 0.5)
        high = max(open_, close) * (1 + abs(rng.gauss(0, 0.001)))
        low = min(open_, close) * (1 - abs(rng.gauss(0, 0.001)))
        volume = round(rng.uniform(1, 80), 8)
        trades = rng.randint(20, 900)
        rows.append(
            (
                int(ts.timestamp()),
                round(open_, 1),
                round(high, 1),
                round(low, 1),
                round(close, 1),
                volume,
                trades,
            )
        )
        price = close
    return rows


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()  # supports --help
    FIXTURES.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    api_continuation: list[list[float]] = []
    for pair, price, hourly_n, daily_n in SPECS:
        hourly = make_candles(rng, START, timedelta(hours=1), hourly_n + API_ROWS, price)
        daily = make_candles(rng, START, timedelta(days=1), daily_n, price)

        # Dump CSVs cover only the first hourly_n/daily_n candles; the tail of
        # the hourly XBTEUR series becomes the "recorded" API continuation so
        # updater tests splice seamlessly.
        for name, rows in ((f"{pair}_60.csv", hourly[:hourly_n]), (f"{pair}_1440.csv", daily)):
            out = FIXTURES / name
            out.write_text("".join(",".join(str(v) for v in row) + "\n" for row in rows))
            print(f"wrote {out} ({len(rows)} rows)")

        if pair == "XBTEUR":
            api_continuation = [
                [ts * 1000, o, h, lo, c, v] for ts, o, h, lo, c, v, _ in hourly[hourly_n:]
            ]

    out = FIXTURES / "ccxt_kraken_xbteur_1h.json"
    out.write_text(json.dumps(api_continuation, indent=1) + "\n")
    print(f"wrote {out} ({len(api_continuation)} rows)")


if __name__ == "__main__":
    main()
