"""Data-quality validation for canonical candle frames.

Checks, per dataset:

- null timestamps                 -> error
- duplicate timestamps            -> error
- non-monotonic timestamp order   -> error
- timestamps off the epoch-anchored UTC grid (on the hour for 1h,
  midnight UTC for 1d)            -> error
- gaps (missing candles between the first and last timestamp) -> error.
  Note: Kraken's dumps simply omit intervals with no trades, so early
  low-liquidity history legitimately triggers this; the report includes gap
  locations so the call can be made per dataset.
- zero-volume candles in the middle of an active period -> warning.
  "Active" means trading is happening on BOTH sides of the candle: at least
  ``min_active_fraction`` of the ``window`` candles before it AND after it
  have volume. Leading/trailing dead zones therefore don't get flagged, a
  dead hour between two busy days does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trading_bot.data import schema
from trading_bot.data.data_quality_only import zero_volume_in_active_period
from trading_bot.data.store import ParquetStore

_MAX_EXAMPLES = 5  # cap listed examples per issue so reports stay readable

DEFAULT_ACTIVE_WINDOW = 24
DEFAULT_MIN_ACTIVE_FRACTION = 0.5


@dataclass
class Issue:
    severity: str  # "error" | "warning"
    code: str
    message: str


@dataclass
class ValidationReport:
    dataset: str
    issues: list[Issue] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]


def validate_frame(
    df: pd.DataFrame,
    timeframe: schema.Timeframe,
    *,
    dataset: str = "",
    active_window: int = DEFAULT_ACTIVE_WINDOW,
    min_active_fraction: float = DEFAULT_MIN_ACTIVE_FRACTION,
) -> ValidationReport:
    report = ValidationReport(dataset=dataset)
    if df.empty:
        report.issues.append(Issue("warning", "empty", "dataset is empty"))
        report.stats = {"rows": 0}
        return report

    ts = df[schema.TIMESTAMP]

    null_count = int(ts.isna().sum())
    if null_count:
        report.issues.append(
            Issue("error", "null_timestamps", f"{null_count} row(s) with null timestamps")
        )
        df = df[ts.notna()]
        if df.empty:
            report.stats = {"rows": null_count}
            return report
        ts = df[schema.TIMESTAMP]

    dupes = ts[ts.duplicated(keep=False)]
    if not dupes.empty:
        examples = ", ".join(str(t) for t in dupes.unique()[:_MAX_EXAMPLES])
        report.issues.append(
            Issue(
                "error",
                "duplicate_timestamps",
                f"{dupes.duplicated().sum()} duplicated timestamp(s), e.g. {examples}",
            )
        )

    if not ts.is_monotonic_increasing:
        report.issues.append(
            Issue("error", "non_monotonic", "timestamps are not in increasing order")
        )

    # Work on a sorted, deduplicated view for the remaining checks.
    clean = df.sort_values(schema.TIMESTAMP, kind="stable").drop_duplicates(
        subset=schema.TIMESTAMP
    )
    cts = clean[schema.TIMESTAMP].reset_index(drop=True)
    first, last = cts.iloc[0], cts.iloc[-1]
    step = pd.Timedelta(timeframe.step)

    # Kraken candles are anchored to the epoch: 1h opens on the hour, 1d at
    # midnight UTC. Anchoring here (not at the first row) means a shifted
    # first candle is itself flagged instead of redefining the grid.
    epoch = pd.Timestamp(0, tz="UTC")
    off_grid_mask = (cts - epoch) % step != pd.Timedelta(0)
    if off_grid_mask.any():
        offenders = cts[off_grid_mask]
        examples = ", ".join(str(t) for t in offenders.head(_MAX_EXAMPLES))
        report.issues.append(
            Issue(
                "error",
                "off_grid",
                f"{len(offenders)} timestamp(s) not aligned to the {timeframe.name} "
                f"grid, e.g. {examples}",
            )
        )

    expected_rows = int((last - first) / step) + 1
    gap_runs: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    diffs = cts.diff()
    for idx in diffs[diffs > step].index:
        prev_ts = cts.iloc[idx - 1]
        next_ts = cts.iloc[idx]
        # for an off-grid (fractional-step) diff, `whole` on-grid candles were
        # skipped; for an on-grid diff it's whole - 1
        whole, remainder = divmod(next_ts - prev_ts, step)
        missing = int(whole) - 1 if remainder == pd.Timedelta(0) else int(whole)
        gap_runs.append((prev_ts, next_ts, missing))
    missing_total = sum(n for _, _, n in gap_runs)
    if gap_runs:
        largest = max(gap_runs, key=lambda g: g[2])
        examples = "; ".join(
            f"{n} candle(s) missing between {a} and {b}" for a, b, n in gap_runs[:_MAX_EXAMPLES]
        )
        report.issues.append(
            Issue(
                "error",
                "gaps",
                f"{missing_total} missing candle(s) across {len(gap_runs)} gap(s) "
                f"(largest: {largest[2]} between {largest[0]} and {largest[1]}). {examples}",
            )
        )

    zero_active = zero_volume_in_active_period(
        clean, window=active_window, min_active_fraction=min_active_fraction
    )
    zero_total = int((clean[schema.VOLUME] == 0).sum())
    if zero_active:
        examples = ", ".join(str(t) for t in zero_active[:_MAX_EXAMPLES])
        report.issues.append(
            Issue(
                "warning",
                "zero_volume_active",
                f"{len(zero_active)} zero-volume candle(s) inside active periods, "
                f"e.g. {examples}",
            )
        )

    report.stats = {
        "rows": len(df),
        "first": first,
        "last": last,
        "expected_rows": expected_rows,
        "missing_rows": missing_total,
        "gap_runs": len(gap_runs),
        "zero_volume_rows": zero_total,
        "zero_volume_active_rows": len(zero_active),
    }
    return report


def validate_store(
    store: ParquetStore,
    *,
    active_window: int = DEFAULT_ACTIVE_WINDOW,
    min_active_fraction: float = DEFAULT_MIN_ACTIVE_FRACTION,
) -> list[ValidationReport]:
    reports = []
    for pair, timeframe in store.datasets():
        df = store.read(pair, timeframe)
        reports.append(
            validate_frame(
                df,
                timeframe,
                dataset=f"{pair.kraken_name}/{timeframe.name}",
                active_window=active_window,
                min_active_fraction=min_active_fraction,
            )
        )
    return reports
