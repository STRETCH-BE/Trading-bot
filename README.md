# Trading-bot

Crypto trading bot for Kraken (XBTEUR, ETHEUR), built in stages.
**Stage 1 — data layer** is what exists today: no strategies, no order
execution, just clean candle data.

## What Stage 1 does

- **Ingests Kraken's historical OHLCVT dumps** (quarterly zip archives from
  [Kraken support](https://support.kraken.com/articles/360047124832)) for
  XBTEUR and ETHEUR, 1h and 1d candles.
- **Normalises to parquet** under `data/parquet/<PAIR>/<TF>.parquet` with a
  documented schema — see [docs/data-schema.md](docs/data-schema.md).
- **Tops up incrementally** from Kraken's REST API via ccxt for candles newer
  than the dump, storing only closed candles, refusing to write silent holes
  (Kraken's REST OHLC history is ~720 candles deep).
- **Validates**: duplicate timestamps, ordering, grid alignment, gaps, and
  zero-volume candles inside active trading periods.

## Quick start

```sh
uv venv --python 3.11 .venv
uv pip install -p .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest    # offline test suite

# real data (needs network):
.venv/bin/tbot-data download-dump "<drive-url-from-kraken-support-page>" data/raw/dump.zip
.venv/bin/tbot-data ingest data/raw/dump.zip
.venv/bin/tbot-data update
.venv/bin/tbot-data report
```

## Repo map

```
src/trading_bot/data/   schema, dump ingestion, ccxt updater, parquet store,
                        validation, reporting, CLI
docs/data-schema.md     the storage contract
tests/                  offline tests + recorded-format fixtures
scripts/make_fixtures.py  regenerates the fixtures deterministically
```
