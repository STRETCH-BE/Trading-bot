"""Rolling walk-forward: in-sample selection, out-of-sample evaluation.

POSITION CONTINUITY is the design constraint everything here serves. The
out-of-sample result is ONE continuous engine run over the whole OOS span,
with a parameter schedule that swaps at window boundaries
(``donchian_breakout_schedule``). The "concatenated equity curve" is not
stitched from per-window runs — per-window numbers are read back off the
single continuous run by slicing. Therefore, by construction:

- positions carry across boundaries;
- no forced close, no forced re-entry, no boundary fee the strategy did not
  choose;
- only the parameters change at a boundary, never position state.

Windows are defined in CANDLE COUNTS (daily data: 180 candles ~ 180 days;
missing candles are absent, never filled). Contiguity requires step == the
out-of-sample length, which is asserted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pandas as pd

from trading_bot.backtest.config import BacktestConfig
from trading_bot.backtest.engine import BacktestResult, backtest
from trading_bot.backtest.metrics import _sharpe
from trading_bot.data import schema
from trading_bot.strategies.donchian import (
    DonchianParams,
    ParamSpan,
    donchian_breakout,
    donchian_breakout_schedule,
)
from trading_bot.strategies.meanrev import (
    MeanRevParams,
    MeanRevSpan,
    meanrev_schedule,
)
from trading_bot.strategies.voltrend import (
    VolTrendParams,
    VolTrendSpan,
    voltrend_schedule,
)

PPY_DAILY = 365.0


def default_grid() -> list[DonchianParams]:
    """The Stage 4a grid: 11 x 6 x 6 = 396 combinations, atr_period fixed."""
    return [
        DonchianParams(
            entry_lookback=entry,
            exit_lookback=exit_,
            atr_period=14,
            atr_multiple=mult / 10.0,
        )
        for entry in range(10, 61, 5)
        for exit_ in range(5, 31, 5)
        for mult in range(15, 41, 5)  # 1.5 .. 4.0 step 0.5, integer-safe
    ]


@dataclass(frozen=True)
class WindowSpec:
    """Index bounds into the candle frame. IS = [is_lo, is_hi); OOS = [oos_lo, oos_hi)."""

    idx: int
    is_lo: int
    is_hi: int
    oos_lo: int
    oos_hi: int
    partial: bool  # OOS shorter than the nominal length (tail of the data)


def make_windows(n_candles: int, is_len: int, oos_len: int, step: int) -> list[WindowSpec]:
    if step != oos_len:
        raise ValueError(
            f"step ({step}) must equal oos_len ({oos_len}) or the concatenated "
            f"out-of-sample periods would overlap or leave holes"
        )
    windows: list[WindowSpec] = []
    k = 0
    while True:
        is_lo = k * step
        is_hi = is_lo + is_len
        oos_hi = min(is_hi + oos_len, n_candles)
        if oos_hi <= is_hi:  # no OOS candle left
            break
        windows.append(
            WindowSpec(
                idx=k,
                is_lo=is_lo,
                is_hi=is_hi,
                oos_lo=is_hi,
                oos_hi=oos_hi,
                partial=(oos_hi - is_hi) < oos_len,
            )
        )
        if oos_hi == n_candles:
            break
        k += 1
    return windows


class NoEvaluableComboError(ValueError):
    """Every combo's warmup meets or exceeds the in-sample window.

    Selection would be a pure fabrication, so the protocol refuses to select.
    Reachable only by a grid/window mismatch that the preregistered configs
    do not contain — but 'cannot happen' is not a definition, this is.
    """


@dataclass
class GridPoint:
    params: DonchianParams
    sharpe: float
    total_return: float
    # False when the combo could not be evaluated AT ALL in this window
    # (warmup >= window length): its signals are all-zero by construction,
    # not by evidence. FINDINGS.md §3: such combos used to score a fabricated
    # Sharpe of 0.0 and win the argmax whenever every real score was
    # negative. Ineligible combos carry -inf so that ANY argmax — including
    # a caller's raw tuple max, not just select_best — cannot pick them.
    eligible: bool = True


def voltrend_grid() -> list[VolTrendParams]:
    """Strategy #2 grid: exactly 12 combinations (preregistration rule 2 cap)."""
    return [
        VolTrendParams(trend_lookback=trend, vol_lookback=vol, target_vol=tv)
        for trend in (50, 100, 200)
        for vol in (20, 60)
        for tv in (0.40, 0.60)
    ]


def meanrev_grid() -> list[MeanRevParams]:
    """Strategy #3 grid: exactly 12 combinations (Amendment 6, rule 2 cap)."""
    return [
        MeanRevParams(lookback=n, entry_k=k, exit_e=e)
        for n in (10, 20, 30)
        for k in (1.0, 1.5)
        for e in (0.0, 0.5)
    ]


def meanrev_scheduler(
    candles: pd.DataFrame, spans: list, trade_start: int
) -> pd.Series:
    """Scheduler for strategy #3; ``spans`` carry MeanRevParams."""
    converted = [MeanRevSpan(start=s.start, params=s.params) for s in spans]
    return meanrev_schedule(candles, converted, trade_start=trade_start)


def _donchian_signal(candles: pd.DataFrame, params):
    return donchian_breakout(candles, params)


def search_window(
    is_candles: pd.DataFrame,
    combos: Sequence,
    config: BacktestConfig,
    pair: str,
    signal_factory: Callable[[pd.DataFrame, object], pd.Series] = _donchian_signal,
) -> list[GridPoint]:
    """Backtest every combo on the in-sample slice, standalone (starts flat).

    Same engine, same costs as the out-of-sample run — no cost-free variant.

    THE WARMUP RULE (repair of FINDINGS.md §3): a combo whose ``warmup``
    meets or exceeds the window length cannot present any in-sample evidence
    — its signals are all-zero by construction. It is returned INELIGIBLE
    with sharpe and total_return of -inf, so no argmax anywhere can select
    it. A combo that IS evaluable but happened never to trade keeps its real
    score (a flat 0.0 is genuine evidence: "it would have sat out").
    """
    n = len(is_candles)
    points = []
    for params in combos:
        warmup = getattr(params, "warmup", 0)
        if warmup >= n:
            points.append(
                GridPoint(
                    params=params,
                    sharpe=float("-inf"),
                    total_return=float("-inf"),
                    eligible=False,
                )
            )
            continue
        result = backtest(
            is_candles,
            lambda c, p=params: signal_factory(c, p),
            config,
            pair=pair,
        )
        returns = result.equity.pct_change().dropna()
        points.append(
            GridPoint(
                params=params,
                sharpe=_sharpe(returns, PPY_DAILY, config.risk_free_rate),
                total_return=result.final_equity / config.starting_capital - 1.0,
            )
        )
    if not any(p.eligible for p in points):
        raise NoEvaluableComboError(
            f"no combo in the grid is evaluable in a {n}-candle window; "
            f"minimum warmup in the grid is "
            f"{min(getattr(p.params, 'warmup', 0) for p in points)}"
        )
    return points


def select_best(points: Sequence[GridPoint], metric: str) -> GridPoint:
    """Deterministic argmax over ELIGIBLE points; ties resolve to the
    earliest grid entry. Raises rather than select a combo that was never
    evaluated (FINDINGS.md §3)."""
    eligible = [p for p in points if p.eligible]
    if not eligible:
        raise NoEvaluableComboError("selection over a set with no evaluable combo")
    if metric == "sharpe":
        best = max(eligible, key=lambda p: p.sharpe)
    elif metric == "total_return":
        best = max(eligible, key=lambda p: p.total_return)
    else:
        raise ValueError(f"unknown selection metric {metric!r}")
    assert best.sharpe != float("-inf"), "an ineligible combo escaped the filter"
    return best


SignalScheduler = Callable[[pd.DataFrame, list[ParamSpan], int], pd.Series]


def _donchian_scheduler(
    candles: pd.DataFrame, spans: list[ParamSpan], trade_start: int
) -> pd.Series:
    return donchian_breakout_schedule(candles, spans, trade_start=trade_start)


def voltrend_scheduler(
    candles: pd.DataFrame, spans: list, trade_start: int
) -> pd.Series:
    """Scheduler for strategy #2; ``spans`` carry VolTrendParams."""
    converted = [VolTrendSpan(start=s.start, params=s.params) for s in spans]
    return voltrend_schedule(candles, converted, trade_start=trade_start)


def run_oos_concatenated(
    candles: pd.DataFrame,
    selections: list[tuple[WindowSpec, DonchianParams]],
    config: BacktestConfig,
    pair: str | None,
    *,
    scheduler: SignalScheduler = _donchian_scheduler,
) -> BacktestResult:
    """ONE continuous engine run over the full out-of-sample span.

    The signal is computed over the entire candle history (bands may look back
    into the past, which is legitimate), forced flat before the first OOS
    candle, and parameter-swapped at each window boundary with position state
    carried across. The engine then runs over the OOS slice only, so the
    equity curve starts at the true start, flat, with starting capital.
    """
    if not selections:
        raise ValueError("no windows selected")
    first_oos = selections[0][0].oos_lo
    oos_end = selections[-1][0].oos_hi
    spans = [ParamSpan(start=w.oos_lo, params=p) for w, p in selections]

    full_signal = scheduler(candles, spans, first_oos)

    oos_candles = candles.iloc[first_oos:oos_end].reset_index(drop=True)
    oos_signal = pd.Series(
        full_signal.to_numpy()[first_oos:oos_end], index=oos_candles.index
    )
    return backtest(oos_candles, lambda _c: oos_signal, config, pair=pair)


@dataclass
class WindowOutcome:
    spec: WindowSpec
    params: DonchianParams
    is_sharpe: float
    is_return: float
    oos_return: float
    oos_sharpe: float
    oos_fills: int
    oos_closed_trades: int
    carried_in: bool
    carried_out: bool
    is_start: pd.Timestamp = field(default=None)  # type: ignore[assignment]
    oos_start: pd.Timestamp = field(default=None)  # type: ignore[assignment]
    oos_end: pd.Timestamp = field(default=None)  # type: ignore[assignment]


def per_window_outcomes(
    candles: pd.DataFrame,
    selections: list[tuple[WindowSpec, DonchianParams, GridPoint]],
    oos_result: BacktestResult,
    config: BacktestConfig,
) -> list[WindowOutcome]:
    """Read per-window numbers back off the single continuous OOS run."""
    ts = candles[schema.TIMESTAMP]
    equity = oos_result.equity
    first_oos = selections[0][0].oos_lo
    outcomes = []
    for spec, params, point in selections:
        lo = spec.oos_lo - first_oos  # position within the OOS slice
        hi = spec.oos_hi - first_oos
        eq_before = float(equity.iloc[lo - 1]) if lo > 0 else config.starting_capital
        eq_after = float(equity.iloc[hi - 1])
        window_eq = equity.iloc[max(lo - 1, 0) : hi]
        rets = window_eq.pct_change().dropna()

        w_start = ts.iloc[spec.oos_lo]
        w_end = ts.iloc[spec.oos_hi - 1]
        fills = [f for f in oos_result.fills if w_start <= f.time <= w_end]
        closed = [t for t in oos_result.trades if w_start <= t.exit_time <= w_end]

        # position DURING the first candle of the window = signal of the
        # previous candle; read it from the fills-reconstructed equity instead:
        # carried flags come from the engine's own units_after bookkeeping.
        carried_in = _position_open_at(oos_result, lo)
        carried_out = _position_open_at(oos_result, hi)

        outcomes.append(
            WindowOutcome(
                spec=spec,
                params=params,
                is_sharpe=point.sharpe,
                is_return=point.total_return,
                oos_return=eq_after / eq_before - 1.0,
                oos_sharpe=_sharpe(rets, PPY_DAILY, config.risk_free_rate),
                oos_fills=len(fills),
                oos_closed_trades=len(closed),
                carried_in=carried_in,
                carried_out=carried_out,
                is_start=ts.iloc[spec.is_lo],
                oos_start=w_start,
                oos_end=w_end,
            )
        )
    return outcomes


def _position_open_at(result: BacktestResult, slice_pos: int) -> bool:
    """Units held at the OPEN of OOS-slice position ``slice_pos``."""
    ts = result.candles[schema.TIMESTAMP]
    if slice_pos <= 0:
        return False
    if slice_pos >= len(ts):
        return result.open_units > 0
    boundary_time = ts.iloc[slice_pos]
    units = 0.0
    for f in result.fills:
        if f.time >= boundary_time:
            break
        units = f.units_after
    return units > 1e-12


@dataclass
class StabilityStats:
    mean_abs_change: dict[str, float]
    jumps_gt_2_steps: dict[str, int]
    n_transitions: int
    n_changed: int  # boundaries where at least one parameter moved


GRID_STEPS = {"entry_lookback": 5, "exit_lookback": 5, "atr_multiple": 0.5}
# Strategy #2's grid is coarse by design (12-combination cap), so its "steps"
# are the minimum gap along each axis. NOTE: on a 3x2x2 grid a move of more
# than 2 steps is only reachable on trend_lookback (50 -> 200); the >2-step
# stability criterion is therefore far weaker here than on a 396-point grid
# and must not be read as strong evidence of stability.
VOLTREND_GRID_STEPS = {"trend_lookback": 50, "vol_lookback": 40, "target_vol": 0.20}


def parameter_stability(
    params_seq: list, grid_steps: dict[str, float] | None = None
) -> StabilityStats:
    grid_steps = grid_steps or GRID_STEPS
    fields_ = list(grid_steps)
    changes: dict[str, list[float]] = {f: [] for f in fields_}
    n_changed = 0
    for a, b in zip(params_seq, params_seq[1:], strict=False):
        moved = False
        for f in fields_:
            d = abs(getattr(b, f) - getattr(a, f))
            changes[f].append(d)
            moved = moved or d > 0
        n_changed += int(moved)
    n = max(len(params_seq) - 1, 1)
    return StabilityStats(
        mean_abs_change={f: sum(v) / n for f, v in changes.items()},
        jumps_gt_2_steps={
            f: sum(1 for d in changes[f] if d > 2 * grid_steps[f]) for f in fields_
        },
        n_transitions=len(params_seq) - 1,
        n_changed=n_changed,
    )
