"""API keys must never reach a log sink. The mandated redaction test.

There is no live broker yet, so no key exists to leak. That is exactly when to
install this: after the first leak the credential is already burned.
"""

from __future__ import annotations

import json
import logging

import pytest

from trading_bot.logging_setup import (
    JsonFormatter,
    RedactingFilter,
    configure,
    redact,
)

# Shaped like real Kraken credentials, but not real: a 56-char public key and
# an 88-char private key are what the API issues.
FAKE_PUBLIC = "A" * 40 + "bcdefghijklmnop"
FAKE_PRIVATE = "x" * 60 + "YZ0123456789abcdefghijklmno+/=="


@pytest.mark.parametrize(
    "text",
    [
        f"KRAKEN_API_KEY={FAKE_PUBLIC}",
        f"kraken_api_secret: {FAKE_PRIVATE}",
        f'{{"apiKey": "{FAKE_PUBLIC}"}}',
        f"password={FAKE_PUBLIC}",
        f"auth token = {FAKE_PUBLIC}",
        f"nonce={FAKE_PUBLIC}",
        f"submitting with {FAKE_PRIVATE} failed",  # unlabelled, shape only
    ],
)
def test_credentials_never_survive_redaction(text):
    out = redact(text)
    assert FAKE_PUBLIC not in out, f"public key leaked: {out}"
    assert FAKE_PRIVATE not in out, f"private key leaked: {out}"
    assert "REDACTED" in out


def test_ordinary_log_lines_are_left_alone():
    """A redactor that eats normal output is a redactor people turn off."""
    for text in (
        "cycle 9ed4384766ac done: settled=1 submitted=1",
        "XBTEUR buy 0.00251234 @ 97234.50 (filled)",
        "risk[max_order_size_pct] order is 31.2% of equity",
        "reconciliation agrees: cash 10000.00, 1 position",
    ):
        assert redact(text) == text


def test_filter_redacts_the_message_and_its_args(caplog):
    logger = logging.getLogger("test.redaction.args")
    logger.addFilter(RedactingFilter())
    with caplog.at_level(logging.INFO, logger="test.redaction.args"):
        logger.info("connecting with key=%s", FAKE_PUBLIC)

    rendered = caplog.records[0].getMessage()
    assert FAKE_PUBLIC not in rendered
    assert "REDACTED" in rendered


def test_filter_redacts_exception_text():
    record = logging.LogRecord(
        "t", logging.ERROR, __file__, 1, "boom", None, None
    )
    record.exc_text = f"AuthError: rejected key {FAKE_PRIVATE}"
    RedactingFilter().filter(record)
    assert FAKE_PRIVATE not in record.exc_text


def test_json_formatter_emits_one_object_per_line():
    record = logging.LogRecord(
        "trading_bot.run", logging.INFO, __file__, 1, "cycle done", None, None
    )
    line = JsonFormatter().format(record)
    assert "\n" not in line
    payload = json.loads(line)
    assert payload["level"] == "INFO"
    assert payload["logger"] == "trading_bot.run"
    assert payload["message"] == "cycle done"
    assert payload["ts"]


def test_configured_file_sink_is_redacted_json(tmp_path):
    """End to end: what actually lands on disk carries no credential."""
    log_file = tmp_path / "logs" / "trading-bot.jsonl"
    configure(level="INFO", log_file=log_file, stream=False)
    try:
        logging.getLogger("trading_bot.test").warning(
            "auth failed for KRAKEN_API_KEY=%s", FAKE_PUBLIC
        )
        for handler in logging.getLogger().handlers:
            handler.flush()

        lines = [ln for ln in log_file.read_text().splitlines() if ln.strip()]
        assert lines, "nothing was written to the JSON sink"
        payload = json.loads(lines[-1])
        assert FAKE_PUBLIC not in json.dumps(payload)
        assert "REDACTED" in payload["message"]
    finally:
        configure(level="INFO", log_file=None, stream=False)


def test_rotation_is_configured_at_50mb_times_10(tmp_path):
    from logging.handlers import RotatingFileHandler

    configure(level="INFO", log_file=tmp_path / "bot.jsonl", stream=False)
    try:
        rotating = [
            h for h in logging.getLogger().handlers
            if isinstance(h, RotatingFileHandler)
        ]
        assert len(rotating) == 1
        assert rotating[0].maxBytes == 50 * 1024 * 1024
        assert rotating[0].backupCount == 10
    finally:
        configure(level="INFO", log_file=None, stream=False)
