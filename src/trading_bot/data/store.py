"""Parquet-backed candle store.

Layout: ``<root>/<PAIR>/<timeframe>.parquet`` (e.g. ``data/parquet/XBTEUR/1h.parquet``).
Files are written atomically (tmp file + rename) and always sorted by timestamp.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trading_bot.data import schema
from trading_bot.data.normalize import finalize

_ARROW_SCHEMA = pa.schema(
    [
        pa.field(schema.TIMESTAMP, pa.timestamp("us", tz="UTC")),
        pa.field(schema.OPEN, pa.float64()),
        pa.field(schema.HIGH, pa.float64()),
        pa.field(schema.LOW, pa.float64()),
        pa.field(schema.CLOSE, pa.float64()),
        pa.field(schema.VWAP, pa.float64()),
        pa.field(schema.VOLUME, pa.float64()),
        pa.field(schema.TRADES, pa.int64()),
        pa.field(schema.SOURCE, pa.string()),
    ]
)


class ParquetStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, pair: schema.Pair, timeframe: schema.Timeframe) -> Path:
        return self.root / pair.kraken_name / f"{timeframe.name}.parquet"

    def read(self, pair: schema.Pair, timeframe: schema.Timeframe) -> pd.DataFrame:
        """Read a dataset; returns an empty canonical frame if absent."""
        path = self.path(pair, timeframe)
        if not path.exists():
            return schema.empty_frame()
        df = pq.read_table(path).to_pandas()
        df[schema.TRADES] = df[schema.TRADES].astype("Int64")
        df[schema.SOURCE] = df[schema.SOURCE].astype("string")
        df[schema.TIMESTAMP] = df[schema.TIMESTAMP].astype("datetime64[us, UTC]")
        return df[schema.COLUMNS]

    def write(self, pair: schema.Pair, timeframe: schema.Timeframe, df: pd.DataFrame) -> None:
        """Atomically replace a dataset with ``df`` (finalized to canonical order)."""
        missing = [c for c in schema.COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"frame is missing canonical columns: {missing}")
        df = finalize(df[schema.COLUMNS])
        path = self.path(pair, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, schema=_ARROW_SCHEMA, preserve_index=False)
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.stem}.", suffix=".parquet.tmp"
        )
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            pq.write_table(table, tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def upsert(
        self,
        pair: schema.Pair,
        timeframe: schema.Timeframe,
        new: pd.DataFrame,
    ) -> int:
        """Merge ``new`` rows into the dataset; returns the growth in row count.

        Timestamp-collision policy: a dump-sourced row replaces an API-sourced
        one (dumps are final and carry the trade count, so a quarterly dump may
        enrich candles the updater fetched earlier); otherwise the existing row
        wins, which keeps re-runs of both ingest and update idempotent.
        """
        existing = self.read(pair, timeframe)
        if existing.empty:
            merged = finalize(new[schema.COLUMNS])
        else:
            combined = pd.concat([existing, new[schema.COLUMNS]], ignore_index=True)
            priority = (combined[schema.SOURCE] != schema.SOURCE_DUMP).astype(int)
            order = combined.assign(_priority=priority).sort_values(
                [schema.TIMESTAMP, "_priority"], kind="stable"
            )
            merged = finalize(order.drop(columns="_priority"))
        added = len(merged) - len(existing)
        self.write(pair, timeframe, merged)
        return added

    def last_timestamp(
        self, pair: schema.Pair, timeframe: schema.Timeframe
    ) -> pd.Timestamp | None:
        ts = self.read(pair, timeframe)[schema.TIMESTAMP].dropna()
        if ts.empty:
            return None
        return ts.iloc[-1]

    def datasets(self) -> Iterator[tuple[schema.Pair, schema.Timeframe]]:
        """Yield (pair, timeframe) for every dataset present on disk, registry order."""
        for pair in schema.PAIRS.values():
            for timeframe in schema.TIMEFRAMES.values():
                if self.path(pair, timeframe).exists():
                    yield pair, timeframe
