"""Human-readable summary of the store: coverage, counts, data quality."""

from __future__ import annotations

from trading_bot.data.store import ParquetStore
from trading_bot.data.validate import ValidationReport, validate_store


def render(reports: list[ValidationReport]) -> str:
    if not reports:
        return "store is empty — no datasets found"

    header = f"{'dataset':<14} {'first':<20} {'last':<20} {'candles':>9} {'quality':<8}"
    lines = [header, "-" * len(header)]
    for rep in reports:
        first = _fmt_ts(rep.stats.get("first"))
        last = _fmt_ts(rep.stats.get("last"))
        rows = rep.stats.get("rows", 0)
        quality = "OK" if rep.ok and not rep.warnings else (
            "ERRORS" if not rep.ok else "warnings"
        )
        lines.append(f"{rep.dataset:<14} {first:<20} {last:<20} {rows:>9} {quality:<8}")

    issue_lines = []
    for rep in reports:
        for issue in rep.issues:
            issue_lines.append(f"  [{issue.severity}] {rep.dataset}: {issue.message}")
    if issue_lines:
        lines += ["", "issues:", *issue_lines]
    return "\n".join(lines)


def report_store(store: ParquetStore) -> str:
    return render(validate_store(store))


def _fmt_ts(value: object) -> str:
    if value is None:
        return "-"
    return str(value)[:19]  # "YYYY-MM-DD HH:MM:SS", drop tz suffix for the table
