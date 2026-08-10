"""Full data-quality audit of an ingested store.

Reports coverage, missing intervals (with the longest runs), OHLC sanity
counts, source provenance, and staleness versus now.

On missing intervals: per Kraken's documentation a missing candle means NO
TRADES OCCURRED in that interval, not that data is corrupt. Nothing in this
codebase fills them — no interpolation, no forward-fill. They are reported
as absent and left absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from trading_bot.data import schema
from trading_bot.data.store import ParquetStore


@dataclass
class GapRun:
    start: pd.Timestamp  # last present candle before the hole
    end: pd.Timestamp  # first present candle after the hole
    missing: int  # candles absent between them

    @property
    def first_missing(self) -> pd.Timestamp:
        return self.start

    def describe(self, step: pd.Timedelta) -> str:
        gap_from = self.start + step
        gap_to = self.end - step
        span = f"{gap_from:%Y-%m-%d %H:%M} .. {gap_to:%Y-%m-%d %H:%M}"
        return f"{self.missing:>7,}  {span}"


@dataclass
class DatasetAudit:
    dataset: str
    rows: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    expected_rows: int
    missing_intervals: int
    gap_runs: list[GapRun] = field(default_factory=list)
    high_lt_low: int = 0
    close_out_of_range: int = 0
    open_out_of_range: int = 0
    negative_volume: int = 0
    non_positive_price: int = 0
    duplicate_timestamps: int = 0
    null_timestamps: int = 0
    off_grid: int = 0
    source_counts: dict[str, int] = field(default_factory=dict)
    staleness: pd.Timedelta | None = None
    filled_rows: int = 0  # always 0: nothing in this pipeline fills

    @property
    def integrity_ok(self) -> bool:
        return (
            self.high_lt_low
            == self.close_out_of_range
            == self.open_out_of_range
            == self.negative_volume
            == self.non_positive_price
            == self.duplicate_timestamps
            == self.null_timestamps
            == self.off_grid
            == 0
        )

    @property
    def coverage_pct(self) -> float:
        return 100.0 * self.rows / self.expected_rows if self.expected_rows else 0.0


def audit_frame(
    df: pd.DataFrame,
    timeframe: schema.Timeframe,
    *,
    dataset: str,
    now: datetime | None = None,
) -> DatasetAudit:
    now = now or datetime.now(UTC)
    if df.empty:
        return DatasetAudit(dataset, 0, None, None, 0, 0)

    ts = df[schema.TIMESTAMP]
    step = pd.Timedelta(timeframe.step)
    first, last = ts.iloc[0], ts.iloc[-1]
    expected = int((last - first) / step) + 1

    diffs = ts.diff()
    runs: list[GapRun] = []
    for idx in diffs[diffs > step].index:
        pos = df.index.get_loc(idx)
        prev_ts, next_ts = ts.iloc[pos - 1], ts.iloc[pos]
        runs.append(GapRun(prev_ts, next_ts, int((next_ts - prev_ts) / step) - 1))

    price_cols = [schema.OPEN, schema.HIGH, schema.LOW, schema.CLOSE]
    epoch = pd.Timestamp(0, tz="UTC")

    return DatasetAudit(
        dataset=dataset,
        rows=len(df),
        first=first,
        last=last,
        expected_rows=expected,
        missing_intervals=sum(r.missing for r in runs),
        gap_runs=runs,
        high_lt_low=int((df[schema.HIGH] < df[schema.LOW]).sum()),
        close_out_of_range=int(
            ((df[schema.CLOSE] < df[schema.LOW]) | (df[schema.CLOSE] > df[schema.HIGH])).sum()
        ),
        open_out_of_range=int(
            ((df[schema.OPEN] < df[schema.LOW]) | (df[schema.OPEN] > df[schema.HIGH])).sum()
        ),
        negative_volume=int((df[schema.VOLUME] < 0).sum()),
        non_positive_price=int((df[price_cols] <= 0).any(axis=1).sum()),
        duplicate_timestamps=int(ts.duplicated().sum()),
        null_timestamps=int(ts.isna().sum()),
        off_grid=int((((ts - epoch) % step) != pd.Timedelta(0)).sum()),
        source_counts={
            str(k): int(v) for k, v in df[schema.SOURCE].value_counts().items()
        },
        staleness=pd.Timestamp(now) - last,
    )


def audit_store(store: ParquetStore, *, now: datetime | None = None) -> list[DatasetAudit]:
    return [
        audit_frame(
            store.read(pair, tf),
            tf,
            dataset=f"{pair.kraken_name}/{tf.name}",
            now=now,
        )
        for pair, tf in store.datasets()
    ]


def render(audits: list[DatasetAudit], *, top_gaps: int = 10) -> str:
    if not audits:
        return "store is empty — nothing to audit"

    out: list[str] = []
    tf_by_name = {tf.name: tf for tf in schema.TIMEFRAMES.values()}

    header = (
        f"{'dataset':<13} {'first':<17} {'last':<17} {'rows':>9} "
        f"{'expected':>9} {'missing':>8} {'cover':>7}"
    )
    out += ["=" * len(header), "COVERAGE", "=" * len(header), header, "-" * len(header)]
    for a in audits:
        out.append(
            f"{a.dataset:<13} {a.first:%Y-%m-%d %H:%M} {a.last:%Y-%m-%d %H:%M} "
            f"{a.rows:>9,} {a.expected_rows:>9,} {a.missing_intervals:>8,} "
            f"{a.coverage_pct:>6.2f}%"
        )

    out += ["", "=" * len(header), "INTEGRITY (all must be zero)", "=" * len(header)]
    ihdr = (
        f"{'dataset':<13} {'high<low':>9} {'close_oob':>10} {'open_oob':>9} "
        f"{'vol<0':>7} {'price<=0':>9} {'dupes':>7} {'nulls':>7} {'off_grid':>9}"
    )
    out += [ihdr, "-" * len(ihdr)]
    for a in audits:
        out.append(
            f"{a.dataset:<13} {a.high_lt_low:>9} {a.close_out_of_range:>10} "
            f"{a.open_out_of_range:>9} {a.negative_volume:>7} {a.non_positive_price:>9} "
            f"{a.duplicate_timestamps:>7} {a.null_timestamps:>7} {a.off_grid:>9}"
        )

    out += ["", "=" * len(header), "PROVENANCE & STALENESS", "=" * len(header)]
    for a in audits:
        sources = ", ".join(f"{k}={v:,}" for k, v in sorted(a.source_counts.items()))
        days = a.staleness.total_seconds() / 86400 if a.staleness is not None else 0
        out.append(
            f"  {a.dataset:<13} sources: {sources:<28} "
            f"last candle is {days:,.1f} days old"
        )
        out.append(f"  {'':<13} rows filled/interpolated: {a.filled_rows}")

    for a in audits:
        if not a.gap_runs:
            continue
        step = pd.Timedelta(tf_by_name[a.dataset.split("/")[1]].step)
        out += [
            "",
            "=" * len(header),
            f"{a.dataset}: {len(a.gap_runs):,} gap runs, "
            f"{a.missing_intervals:,} missing intervals — "
            f"{min(top_gaps, len(a.gap_runs))} longest",
            "=" * len(header),
            f"{'missing':>7}  absent range (candles that do not exist)",
            "-" * len(header),
        ]
        for run in sorted(a.gap_runs, key=lambda r: r.missing, reverse=True)[:top_gaps]:
            out.append(run.describe(step))

    return "\n".join(out)
