# Candle data schema

Canonical OHLCV storage for the bot. Code source of truth:
`src/trading_bot/data/schema.py`.

## Storage layout

```
data/parquet/<PAIR>/<TIMEFRAME>.parquet
```

- `PAIR` — Kraken pair name: `XBTEUR`, `ETHEUR`
- `TIMEFRAME` — `1h` (60‑minute candles), `1d` (1440‑minute candles)

One parquet file per dataset, always sorted by `timestamp`, unique
timestamps, written atomically (tmp file + rename). `data/` is not committed.

## Columns

| column | arrow type | meaning |
|---|---|---|
| `timestamp` | `timestamp[us, tz=UTC]` | Candle **open** time. UTC, aligned to the timeframe grid. |
| `open` | `float64` | First trade price in the interval (quote currency, EUR). |
| `high` | `float64` | Highest trade price. |
| `low` | `float64` | Lowest trade price. |
| `close` | `float64` | Last trade price. |
| `volume` | `float64` | Base‑asset volume (BTC / ETH) traded in the interval. |
| `trades` | `int64`, nullable | Trade count. Present for rows from modern 7‑column dumps; **null** for API‑sourced rows (ccxt OHLCV has no trade count) and for rows from the legacy 6‑column dump variant. |
| `source` | `string` | Provenance: `kraken_dump` or `ccxt_api`. |

## Invariants

1. `timestamp` strictly increasing, no nulls, no duplicates (enforced on
   write; `tbot-data validate` re-checks).
2. All timestamps on the epoch-anchored UTC grid: on the hour for `1h`,
   midnight UTC for `1d` (matching how Kraken aligns candles).
3. Candles are only stored once **closed** — the updater drops the
   still-forming candle (`open_time + step > now`).
4. Timestamp-collision policy on merge: a dump-sourced row replaces an
   API-sourced one (dumps are final and carry `trades`); otherwise the
   existing row wins, so re-running ingest or update is idempotent.

## Source semantics worth knowing

- **Kraken dumps omit intervals with zero trades.** Gaps flagged by
  validation in early, illiquid history are a property of the source, not
  corruption. Validation still reports every gap (as errors) so the call is
  made explicitly per dataset.
- **Kraken's REST OHLC endpoint serves only ~720 most recent candles** per
  timeframe (~30 days of `1h`, ~2 years of `1d`). If the store's tail is
  older than that window, the updater raises `GapError` instead of silently
  writing a hole; ingest a newer quarterly dump first (or pass
  `--allow-gap` to accept the hole knowingly).
- The dumps' quarterly archives are distributed via rotating Google Drive
  links from <https://support.kraken.com/articles/360047124832>;
  `tbot-data download-dump <url> <dest>` handles the Drive confirmation
  dance, or download manually and point `ingest` at the zip.

## Validation rules (`tbot-data validate`)

| check | severity |
|---|---|
| null timestamps | error |
| duplicate timestamps | error |
| non-monotonic order | error |
| timestamps off the epoch-anchored grid | error |
| gaps (missing candles between first and last) | error |
| zero-volume candle inside an active period | warning |
| empty dataset | warning |

"Active period" for the zero-volume check: at least half of the up-to-24
nearest candles on **each** side traded (candles near the series edges see
correspondingly smaller windows; the very first/last candle has no
before/after window and can never be flagged). Leading/trailing dead zones
of an illiquid listing are not flagged; a silent hour between two busy days
is.
