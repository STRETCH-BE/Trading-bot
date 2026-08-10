"""The holdout boundary: candles on/after ``holdout_start`` are untouchable.

Every path that produces a performance number (backtest, walk-forward,
sensitivity sweep, simulation) runs through ``backtest()``, which raises
``HoldoutViolation`` if the candles reach the boundary. There is no warning
mode. The only way through is a ``HoldoutUnlock`` token, which:

- is never constructed by any default code path — only an explicit
  ``--unlock-holdout`` flag on a runner script may mint one;
- authorises exactly ONE run, then is spent;
- announces itself loudly on stderr when consumed.
"""

from __future__ import annotations

import sys


class HoldoutViolation(RuntimeError):
    """A performance run touched the holdout period without authorisation."""


class HoldoutUnlock:
    """One-shot authorisation to run on holdout data.

    Mint one only from an explicit command-line flag. Never store one in
    config, never construct one in library code.
    """

    def __init__(self) -> None:
        self._used = False

    @property
    def used(self) -> bool:
        return self._used

    def consume(self, *, context: str) -> None:
        if self._used:
            raise HoldoutViolation(
                "this HoldoutUnlock is already spent — one unlock authorises "
                "exactly one run. Mint a new one (explicitly) if you truly "
                "intend a second holdout run."
            )
        self._used = True
        banner = "!" * 74
        print(
            f"{banner}\n"
            f"!!  HOLDOUT UNLOCKED: {context}\n"
            f"!!  This run reads data the strategy was never allowed to see\n"
            f"!!  during development. Results on it are FINAL — re-running\n"
            f"!!  after changes makes the holdout worthless.\n"
            f"{banner}",
            file=sys.stderr,
        )
