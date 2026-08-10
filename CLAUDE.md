# Trading-bot

Crypto trading bot for Kraken, built in stages. **Currently at Stage 1: the
data layer only.** There is deliberately no strategy, signal, backtest, or
order-execution code yet — do not add any without an explicit request.

## Stage roadmap

1. **Data layer** (done) — Kraken OHLCVT dump ingestion, ccxt REST top-up,
   parquet store, validation. Pairs: XBTEUR, ETHEUR; timeframes: 1h, 1d.
2. Backtesting engine (not started)
3. Strategy layer (not started)
4. Paper / live execution (not started)

## Commands

```sh
uv venv --python 3.11 .venv && uv pip install -p .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest          # tests (no network needed, ever)
.venv/bin/ruff check src tests scripts
.venv/bin/tbot-data --help          # data CLI (or: python -m trading_bot.data)
```

Typical data workflow:

```sh
tbot-data download-dump "<google-drive-url>" data/raw/Kraken_OHLCVT.zip
tbot-data ingest data/raw/Kraken_OHLCVT.zip     # dump -> data/parquet/
tbot-data update                                # top up via ccxt REST
tbot-data validate                              # exit 1 on data errors
tbot-data report                                # coverage + quality table
```

## Layout & conventions

- `src/` layout, package `trading_bot`; Python ≥ 3.11; hatchling build; uv
  for envs. Line length 100 (ruff, rules E/F/I/UP/B).
- Candle schema is documented in `docs/data-schema.md` and defined once in
  `trading_bot/data/schema.py`. New code must consume those constants, not
  restate column names.
- `data/` (raw dumps + parquet) is gitignored — never commit market data.
- **Tests must never touch the network.** The ccxt exchange is injected
  (`update()` takes any object with `fetch_ohlcv`); tests use
  `tests/helpers.py::FakeExchange` replaying `tests/fixtures/` (regenerate
  via `scripts/make_fixtures.py`).
- Timestamps are tz-aware UTC everywhere; candle timestamps are open times;
  only closed candles are stored.

## Environment note

The remote sandbox this repo is often developed in has no route to
api.kraken.com or Google Drive (proxy policy). Live `download-dump` /
`update` runs must happen on a machine with normal network access; the test
suite is designed to pass without it.
