"""Ban bare ``.transaction().__enter__()`` anywhere in the codebase.

Why this is worth a dedicated guard: an unreferenced
``_GeneratorContextManager`` is garbage-collected the instant the expression
ends. CPython throws ``GeneratorExit`` into the suspended generator, which
runs the ``except BaseException: ROLLBACK`` branch — so the transaction ends
immediately and the connection silently returns to AUTOCOMMIT. Every write
the caller believes is staged commits one at a time instead.

This actually happened while writing the Stage 6a crash harness: a write
meant to be stranded mid-transaction survived a SIGKILL, because there was no
transaction. It is invisible in review and produces no error — exactly the
kind of bug that corrupts live trading state under a crash.

The rule: always ``with store.transaction():``.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import trading_bot
from trading_bot.broker.state import StateStore

SOURCE_ROOTS = [
    pathlib.Path(trading_bot.__file__).parent,
    pathlib.Path(__file__).parent,
    pathlib.Path(__file__).parent.parent / "scripts",
]

# The crash harness must strand a real transaction, which is the one legitimate
# use — and it holds the context manager in a variable, which is the safe form.
ALLOWED = {"crash_harness.py"}


def _offenders(tree: ast.AST, filename: str) -> list[str]:
    """Find `<expr>.transaction().__enter__()` where the CM is not bound."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "__enter__"):
            continue
        inner = func.value
        # bare form: something.transaction().__enter__()
        if (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "transaction"
        ):
            found.append(f"{filename}:{node.lineno}")
    return found


def test_no_bare_transaction_enter_anywhere():
    offenders: list[str] = []
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.name in ALLOWED or path.name == pathlib.Path(__file__).name:
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            offenders += _offenders(tree, str(path))

    assert not offenders, (
        "bare `.transaction().__enter__()` found — the context manager is "
        "unreferenced, so it is collected immediately, rolled back, and the "
        "connection drops to autocommit. Use `with store.transaction():`. "
        f"Offenders: {offenders}"
    )


def test_the_detector_actually_fires():
    """Control: a harness that never fails is indistinguishable from a pass."""
    bad = ast.parse("store.transaction().__enter__()\n")
    assert _offenders(bad, "fake.py") == ["fake.py:1"]

    good = ast.parse("with store.transaction() as c:\n    c.execute('x')\n")
    assert _offenders(good, "fake.py") == []

    held = ast.parse("cm = store.transaction()\ncm.__enter__()\n")
    assert _offenders(held, "fake.py") == [], "the safe held form must be allowed"


# --- the behaviour itself, pinned ------------------------------------------


def test_bare_enter_silently_drops_to_autocommit(tmp_path):
    """The bug, demonstrated. If this ever stops being true, the ban can go."""
    store = StateStore(tmp_path / "s.db")
    store.set_cash(1.0)

    store.transaction().__enter__()  # noqa: the exact anti-pattern, on purpose
    assert not store._conn.in_transaction, (
        "an unreferenced context manager stayed open — CPython behaviour "
        "changed; re-derive this guard before relaxing it"
    )


def test_held_context_manager_really_opens_a_transaction(tmp_path):
    store = StateStore(tmp_path / "s.db")
    store.set_cash(1.0)

    cm = store.transaction()
    cm.__enter__()
    assert store._conn.in_transaction
    cm.__exit__(None, None, None)
    assert not store._conn.in_transaction


def test_with_statement_commits(tmp_path):
    store = StateStore(tmp_path / "s.db")
    with store.transaction() as c:
        c.execute("INSERT INTO account(key, value) VALUES('probe', 7.0)")
    row = store._conn.execute("SELECT value FROM account WHERE key='probe'").fetchone()
    assert row["value"] == 7.0


def test_with_statement_rolls_back_on_exception(tmp_path):
    store = StateStore(tmp_path / "s.db")
    with pytest.raises(RuntimeError):
        with store.transaction() as c:
            c.execute("INSERT INTO account(key, value) VALUES('ghost', 1.0)")
            raise RuntimeError("boom")
    assert store._conn.execute(
        "SELECT COUNT(*) n FROM account WHERE key='ghost'"
    ).fetchone()["n"] == 0
