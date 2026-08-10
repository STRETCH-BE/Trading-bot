# Test fixtures

Small recorded-format fixtures used by the test suite. **No test touches the
network** — the ccxt exchange is replaced by `tests/helpers.py::FakeExchange`,
which replays these files.

| file | format | contents |
|---|---|---|
| `XBTEUR_60.csv`, `ETHEUR_60.csv` | Kraken OHLCVT dump CSV (no header: `timestamp_s,open,high,low,close,volume,trades`) | 48 hourly candles from 2024‑01‑01 00:00 UTC |
| `XBTEUR_1440.csv`, `ETHEUR_1440.csv` | same | 30 daily candles from 2024‑01‑01 |
| `ccxt_kraken_xbteur_1h.json` | ccxt `fetch_ohlcv` rows: `[timestamp_ms, open, high, low, close, volume]` | 12 hourly candles continuing exactly where `XBTEUR_60.csv` ends |

## Provenance

The *formats* are exact copies of the two wire formats the data layer parses
(Kraken's downloadable OHLCVT archives, and ccxt's unified OHLCV rows for
`kraken`). The *values* are a deterministic seeded random walk, regenerable
with:

```sh
python scripts/make_fixtures.py
```

They were synthesised rather than captured because this project's CI/dev
sandbox has no route to api.kraken.com. If you want real recorded data here,
run `scripts/make_fixtures.py --help` for the layout, capture a real dump
slice + `fetch_ohlcv` response locally, and drop them in with the same names —
the tests only assume format, continuity between the two XBTEUR hourly files,
and clean (gap-free, positive-volume) series.
