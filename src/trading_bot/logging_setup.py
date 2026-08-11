"""Structured logging: JSON to a rotating file, plain text to journald.

Two sinks, on purpose:

* **stderr** — systemd captures it into the journal. Human-readable, because
  that is what ``journalctl -u trading-bot -f`` is for at 3am.
* **a rotating JSON file** — one object per line, machine-readable, 50 MB x 10
  so a week of noisy cycles cannot fill the disk.

Both sinks pass through :class:`RedactingFilter`. There are no API keys in the
codebase today (no live broker exists), but the redaction is installed BEFORE
the keys arrive rather than after the first leak. ``tests/test_redaction.py``
asserts it works.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
from pathlib import Path

MAX_BYTES = 50 * 1024 * 1024
BACKUP_COUNT = 10

# Environment variables whose VALUES must never reach a log sink. Matched
# case-insensitively as substrings, so KRAKEN_API_SECRET matches "secret".
SECRET_NAME_HINTS = ("key", "secret", "token", "password", "passwd", "nonce", "otp")

REDACTED = "***REDACTED***"

# Kraken API keys are base64-ish and long; secrets are longer still. Matching on
# shape as well as on name means a key pasted into a message with no label
# ("auth failed for AbCd...") is still caught.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    # key=value / key: value / "key": "value" for any secret-ish name
    re.compile(
        r"(?i)\b([A-Za-z_][A-Za-z0-9_]*(?:" + "|".join(SECRET_NAME_HINTS) + r")"
        r"[A-Za-z0-9_]*)\b(\s*[=:]\s*)(\"?)([^\s,;}\"']+)(\3)"
    ),
    # a bare long base64-ish run: Kraken private keys are 88 chars, public 56
    re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),
)


def redact(text: str) -> str:
    """Replace anything that looks like a credential. Never raises."""
    def _keep_label(m: re.Match[str]) -> str:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}{m.group(5)}"

    return _PATTERNS[1].sub(REDACTED, _PATTERNS[0].sub(_keep_label, text))


class RedactingFilter(logging.Filter):
    """Redacts the fully rendered message and any exception text.

    Attached to each HANDLER rather than baked into a formatter, so the two
    sinks cannot disagree about what a credential is. Note the limit: a handler
    added later by other code does not inherit this, so anything adding a sink
    must add the filter too — ``configure`` is the only place that should.

    The record is rendered BEFORE redaction and its args cleared. Redacting the
    format string instead would eat the ``%s`` placeholders themselves — a
    secret is often interpolated right after its label, so ``key=%s`` matches
    the name/value pattern and the surviving args then blow up formatting.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:  # pragma: no cover - malformed call site
            rendered = str(record.msg)
        record.msg = redact(rendered)
        record.args = None
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Never raises — a logging failure must not
    take down a trading process."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        try:
            return json.dumps(payload, default=str)
        except Exception:  # pragma: no cover - defensive
            return json.dumps({"level": "ERROR", "message": "unserialisable log record"})


def configure(
    level: str = "INFO",
    log_file: Path | None = None,
    *,
    stream: bool = True,
) -> None:
    """Install the sinks on the root logger. Idempotent per handler type."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    redactor = RedactingFilter()

    if stream:
        console = logging.StreamHandler()
        console.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        console.addFilter(redactor)
        root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
        )
        rotating.setFormatter(JsonFormatter())
        rotating.addFilter(redactor)
        root.addHandler(rotating)

    # Also on the root logger itself. This covers only records logged directly
    # on root (logger-level filters do NOT apply to records propagated up from
    # child loggers) — the handler filters above are what do the real work.
    root.addFilter(redactor)
