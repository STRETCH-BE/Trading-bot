"""The RUNBOOK must not describe commands that do not exist.

A runbook read at 3am that names an imaginary flag is worse than no runbook:
it burns the one thing you are short of. These tests parse RUNBOOK.md and check
every flag, script, and unit file it mentions against the actual codebase.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from trading_bot.run import build_parser

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "RUNBOOK.md"
DEPLOY = ROOT / "deploy"

# Commands the user explicitly said do NOT exist (Stage 6d was never built).
# Documenting them would send someone hunting for a chat bot at 3am.
FORBIDDEN_COMMANDS = ("/halt", "/flatten", "/status")


@pytest.fixture(scope="module")
def runbook() -> str:
    assert RUNBOOK.exists(), "RUNBOOK.md is missing"
    return RUNBOOK.read_text()


def test_every_documented_flag_exists(runbook):
    """Any --flag in the runbook must be a real argparse option."""
    real = set()
    for action in build_parser()._actions:
        real.update(action.option_strings)

    # flags belonging to the helper scripts, not to run.py
    script_flags = {"--heartbeat-file", "--max-age-seconds", "--state-dir",
                    "--backup-dir", "--retention-days", "--since", "-p", "-f",
                    "-header", "-column", "-r"}

    documented = set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]+", runbook))
    unknown = documented - real - script_flags
    assert not unknown, f"RUNBOOK documents flags that do not exist: {sorted(unknown)}"


def test_stage_6d_commands_are_only_ever_named_as_missing(runbook):
    """The Stage 7 amendment: /halt, /flatten and /status do not exist.

    They may appear ONLY inside the 'What is NOT available yet' section, where
    naming them is the point.
    """
    _, _, missing_section = runbook.partition("## What is NOT available yet")
    assert missing_section, "the RUNBOOK has no 'What is NOT available yet' section"
    body = runbook[: runbook.index("## What is NOT available yet")]

    for command in FORBIDDEN_COMMANDS:
        assert command not in body, (
            f"RUNBOOK documents {command!r} as if it works. Stage 6d was never "
            f"built; it belongs only in the 'not available' section."
        )
        assert command in missing_section, (
            f"{command!r} is not listed as missing — the reader will not learn "
            f"it is absent until they try it."
        )


def test_the_missing_section_names_the_stage_for_each_gap(runbook):
    _, _, missing = runbook.partition("## What is NOT available yet")
    for expected in ("Stage 6b", "Stage 6d"):
        assert expected in missing, f"{expected} is never named as the source of a gap"
    assert "KrakenBroker" in missing, "the absence of a live broker is not stated"


def test_every_referenced_script_exists(runbook):
    for match in re.findall(r"scripts/([A-Za-z0-9_]+\.py)", runbook):
        assert (ROOT / "scripts" / match).exists(), f"RUNBOOK names a missing script: {match}"


def test_every_referenced_systemd_unit_exists(runbook):
    units = set(re.findall(r"\b(trading-bot[a-z-]*\.(?:service|timer))\b", runbook))
    assert units, "the RUNBOOK references no systemd units at all"
    for unit in units:
        assert (DEPLOY / unit).exists(), f"RUNBOOK names a unit we do not ship: {unit}"


def test_documented_working_methods_are_all_present(runbook):
    """The four the user named as the real kill and diagnostic methods."""
    for required in ("touch /var/lib/trading-bot/HALT", "systemctl stop trading-bot",
                     "--reconcile-only", "--dry-run"):
        assert required in runbook, f"RUNBOOK does not document {required!r}"


def test_the_no_flatten_warning_is_present(runbook):
    """The most dangerous gap: a halt does not exit the market."""
    assert "does not sell anything" in runbook.lower() or \
           "does NOT sell anything" in runbook
    assert "flatten" in runbook.lower()


def test_units_reference_only_paths_the_deploy_script_creates():
    """A unit pointing at a directory nobody creates fails at 03:00, not now."""
    deploy_sh = (DEPLOY / "deploy.sh").read_text()
    for unit_file in sorted(DEPLOY.glob("*.service")):
        text = unit_file.read_text()
        for path in re.findall(r"(/var/(?:lib|log)/trading-bot[A-Za-z0-9/_.-]*)", text):
            root = "/".join(path.split("/")[:4])
            assert root in deploy_sh, (
                f"{unit_file.name} uses {path}, but deploy.sh never creates {root}"
            )


def _directives(unit_file: Path) -> str:
    """The unit's actual directives, with comment lines dropped.

    Comments legitimately discuss --live and LIVE_TRADING (explaining why they
    are absent); a check that cannot tell a comment from a directive would push
    that explanation out of the file, which is the opposite of what we want.
    """
    return "\n".join(
        line for line in unit_file.read_text().splitlines()
        if not line.lstrip().startswith("#")
    )


def test_the_service_unit_never_enables_live_trading():
    """PAPER MODE ONLY. The unit must not set LIVE_TRADING or pass --live."""
    directives = _directives(DEPLOY / "trading-bot.service")
    assert "--live" not in directives
    assert "--paper" in directives
    assert not re.search(r"^\s*Environment=.*LIVE_TRADING", directives, re.MULTILINE), (
        "the unit sets LIVE_TRADING — that removes one of the two live-trading guards"
    )


def test_the_deploy_script_never_writes_live_trading():
    text = (DEPLOY / "deploy.sh").read_text()
    assert not re.search(r"^\s*LIVE_TRADING\s*=\s*yes", text, re.MULTILINE)
    assert "echo 'LIVE_TRADING=yes'" not in text


def test_secrets_stay_out_of_the_repo_and_working_directory():
    """The user's standing constraint, asserted rather than assumed."""
    unit = (DEPLOY / "trading-bot.service").read_text()
    assert "EnvironmentFile=/etc/trading-bot/env" in unit

    deploy_sh = (DEPLOY / "deploy.sh").read_text()
    assert "chmod 600" in deploy_sh
    assert "chown root:root" in deploy_sh

    # and no credential-looking assignment is committed anywhere in the repo
    for path in (DEPLOY / "trading-bot.service", DEPLOY / "deploy.sh", RUNBOOK):
        text = path.read_text()
        assert not re.search(
            r"(?i)\b(api_?key|api_?secret|private_?key)\s*=\s*[A-Za-z0-9+/]{16,}", text
        ), f"{path.name} appears to contain a credential"


def test_the_unit_does_not_restart_loop_on_a_halt():
    """Exit 2 is a decision. Restarting it would fight the operator."""
    text = (DEPLOY / "trading-bot.service").read_text()
    assert "SuccessExitStatus=2" in text, (
        "a halt exits 2; without SuccessExitStatus systemd treats it as a "
        "failure and restarts into the same halt"
    )
    assert "Restart=on-failure" in text
    assert "StartLimitBurst" in text


def test_the_deploy_script_refuses_unreproducible_deploys():
    text = (DEPLOY / "deploy.sh").read_text()
    assert "git status --porcelain" in text, "does not check for a dirty tree"
    assert "merge-base --is-ancestor" in text, "does not check HEAD is pushed"
    assert "NTPSynchronized" in text, "does not verify time synchronisation"
    assert "set -euo pipefail" in text
