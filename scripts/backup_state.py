#!/usr/bin/env python3
"""Nightly backup of the SQLite state, via the online-backup API.

Why not `cp`: the bot writes in WAL mode and may be mid-transaction when the
timer fires. Copying the .db file alone loses everything still in the -wal, and
copying the three files non-atomically can capture an inconsistent set. The
``Connection.backup()`` API takes a consistent snapshot of a LIVE database
without stopping the writer, which is the whole point of running it nightly
against a bot that never stops.

The backup is verified with ``PRAGMA integrity_check`` before it is allowed to
count as a backup, and the previous good copy is only ever removed by the
retention sweep, never by this run.

Exit codes: 0 all backed up, 1 a backup or verification failed.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

RETENTION_DAYS = 30
STAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def backup_one(source: Path, dest_dir: Path, stamp: str) -> Path:
    """Snapshot one live database. Returns the verified backup path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{source.stem}_{stamp}.db"

    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(dest)
        try:
            src.backup(dst)  # the online-backup API, not a file copy
        finally:
            dst.close()
    finally:
        src.close()

    # A backup nobody has opened is a rumour. Verify before trusting it.
    check = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
    try:
        verdict = check.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        check.close()
    if verdict != "ok":
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{source.name}: backup failed integrity_check: {verdict}")
    return dest


def sweep(dest_dir: Path, days: int, now: datetime) -> list[Path]:
    """Delete backups older than the retention window. Returns what went."""
    cutoff = now - timedelta(days=days)
    removed = []
    for path in sorted(dest_dir.glob("*.db")):
        stem = path.stem.rsplit("_", 1)
        if len(stem) != 2:
            continue
        try:
            when = datetime.strptime(stem[1], STAMP_FORMAT).replace(tzinfo=UTC)
        except ValueError:
            continue  # not ours; leave it alone
        if when < cutoff:
            path.unlink()
            removed.append(path)
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=Path("data/state"))
    parser.add_argument("--backup-dir", type=Path, default=Path("data/backups"))
    parser.add_argument("--retention-days", type=int, default=RETENTION_DAYS)
    args = parser.parse_args(argv)

    sources = sorted(args.state_dir.glob("*.db"))
    if not sources:
        print(f"no databases in {args.state_dir} — nothing to back up", file=sys.stderr)
        return 1

    now = datetime.now(UTC)
    stamp = now.strftime(STAMP_FORMAT)
    failures = 0
    for source in sources:
        try:
            dest = backup_one(source, args.backup_dir, stamp)
            print(f"ok  {source.name} -> {dest} ({dest.stat().st_size:,} bytes)")
        except Exception as exc:  # noqa: BLE001 - report every failure, not the first
            failures += 1
            print(f"FAIL {source.name}: {exc}", file=sys.stderr)

    for path in sweep(args.backup_dir, args.retention_days, now):
        print(f"pruned {path.name} (older than {args.retention_days}d)")

    if failures:
        print(f"{failures} backup(s) FAILED", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
