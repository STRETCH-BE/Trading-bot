#!/usr/bin/env python3
"""Is the bot actually alive? Run by a systemd timer, OUTSIDE the bot process.

A heartbeat the bot checked itself would be worthless — a wedged process cannot
notice that it is wedged, and a crashed one certainly cannot. So the bot only
WRITES ``heartbeat.json`` (see ``trading_bot.run.write_heartbeat``) and this
runs separately to read it.

``systemctl is-active`` is not a substitute: a process can be running, holding
its file descriptors, and stuck in a socket read forever. Staleness catches
that; liveness does not.

Exit codes: 0 fresh, 1 stale or unreadable, 2 the bot reported it had halted.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# The bot's default cycle interval is one day. A heartbeat is late long before
# it is a whole cycle late, so the default tolerance is a cycle plus a margin.
DEFAULT_MAX_AGE_SECONDS = 26 * 3600.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heartbeat-file", type=Path,
                        default=Path("data/state/heartbeat.json"))
    parser.add_argument("--max-age-seconds", type=float,
                        default=DEFAULT_MAX_AGE_SECONDS)
    args = parser.parse_args(argv)

    if not args.heartbeat_file.exists():
        print(f"NO HEARTBEAT: {args.heartbeat_file} does not exist. The bot has "
              f"not completed a cycle since it was last started.", file=sys.stderr)
        return 1

    try:
        payload = json.loads(args.heartbeat_file.read_text())
        written = datetime.fromisoformat(payload["written_at"])
    except Exception as exc:  # noqa: BLE001
        print(f"UNREADABLE HEARTBEAT: {args.heartbeat_file}: {exc}", file=sys.stderr)
        return 1

    if written.tzinfo is None:
        written = written.replace(tzinfo=UTC)
    age = (datetime.now(UTC) - written).total_seconds()

    if payload.get("halted"):
        print(f"HALTED: the last cycle ({payload.get('cycle_id')}) reported a halt.",
              file=sys.stderr)
        return 2

    if age > args.max_age_seconds:
        print(f"STALE HEARTBEAT: last cycle {payload.get('cycle_id')} completed "
              f"{age / 3600:.1f}h ago (limit {args.max_age_seconds / 3600:.1f}h). "
              f"The process may be running but wedged.", file=sys.stderr)
        return 1

    print(f"ok: cycle {payload.get('cycle_id')} completed {age / 3600:.1f}h ago "
          f"(mode {payload.get('mode')}, submitted {payload.get('orders_submitted')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
