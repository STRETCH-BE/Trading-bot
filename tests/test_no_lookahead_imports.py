"""The forward-looking module must never be reachable from a signal path.

``trading_bot.data.data_quality_only`` intentionally reads future candles.
That is fine for data-quality reporting and fatal in a strategy, so this
test enforces the boundary structurally rather than by convention.
"""

from __future__ import annotations

import subprocess
import sys

FORBIDDEN = "trading_bot.data.data_quality_only"

SIGNAL_PATH_MODULES = [
    "trading_bot.strategies",
    "trading_bot.strategies.registry",
    "trading_bot.strategies.donchian",
    "trading_bot.backtest",
    "trading_bot.backtest.engine",
    "trading_bot.backtest.metrics",
]


def _imports_of(module: str) -> set[str]:
    """Modules loaded by importing ``module`` in a FRESH interpreter.

    A fresh process matters: inside the test session other tests have already
    imported the validation stack, so checking this process's sys.modules
    would produce a false positive.
    """
    code = (
        f"import importlib, sys; importlib.import_module({module!r}); "
        "print('\\n'.join(m for m in sys.modules if m.startswith('trading_bot')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return set(out.stdout.split())


def test_forward_looking_module_is_not_in_the_signal_import_graph():
    for module in SIGNAL_PATH_MODULES:
        loaded = _imports_of(module)
        assert FORBIDDEN not in loaded, (
            f"{module} transitively imports {FORBIDDEN}, which reads FUTURE "
            f"candles. That is look-ahead bias in a signal path. Reformulate "
            f"the calculation causally instead of importing it."
        )


def test_the_boundary_check_can_actually_detect_an_import():
    """Control: the detector must see the module when it IS imported."""
    loaded = _imports_of(FORBIDDEN)
    assert FORBIDDEN in loaded


def test_validation_stack_still_uses_it():
    """It is not dead code — the data-quality path is its legitimate consumer."""
    loaded = _imports_of("trading_bot.data.validate")
    assert FORBIDDEN in loaded
