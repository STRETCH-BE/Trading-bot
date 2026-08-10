# Preregistration

A binding record, written **before** the results it governs exist. Its
purpose is to remove the freedom to change the standard after seeing a
number. Nothing in this document may be revised to accommodate a
disappointing result; a revision is a new experiment and is subject to the
budget in rule 1.

---

## Context

Strategy #1 — Donchian channel breakout, long-only with ATR trailing stop,
daily XBTEUR and ETHEUR — **failed the Stage 4 gate on 2026-08-10**, at
commit `846b5ec`, recorded in `results/4a_20260810T222700Z_846b5ec.json`.

It failed on two of the pre-agreed criteria:

- **Excess over buy-and-hold:** negative in 5 of 6 pair × window-configuration
  cells. The single positive cell (ETHEUR at 90/30 windows) appeared at one
  window configuration only and is treated as an artifact.
- **Parameter stability:** selected parameters moved at 62 of 70 boundaries
  (XBTEUR) and 48 of 57 (ETHEUR) — no stable parameter regime was being
  discovered.

It **cleared** the shuffled-returns control: out-of-sample Sharpe 0.91
(XBTEUR) and 1.20 (ETHEUR) against a shuffled-data control of 0.36 and 0.54
respectively. A real trend effect therefore exists in the price series and is
not an artifact of the pipeline. It is simply insufficient to overcome
trading costs relative to holding the asset.

---

## Measured base rates (from the strategy #1 run)

These are not literature estimates. They were measured on this data, with
this pipeline, and they are the empirical justification for rule 2.

Under a **396-combination** grid searched per in-sample window:

| | XBTEUR | ETHEUR |
|---|---|---|
| Selected in-sample Sharpe on **shuffled** returns (pure noise) | 1.46 | 1.69 |
| Selected in-sample Sharpe on **real** returns | 1.90 | 1.98 |

**Roughly three quarters of the in-sample Sharpe was manufactured by the
parameter search itself**, not by any property of the market. Any future
in-sample figure must be read against a noise benchmark of this kind, never
against zero.

---

## Binding rules for all further work

### 1. Strategy budget

At most **2 further strategies** will be tested. Strategy #2 is
cross-sectional momentum. If both fail, the project concludes that
**buy-and-hold wins and no live trading occurs.**

### 2. Grid cap

No strategy may be optimised over more than **12 parameter combinations,
total, per window**.

Justification: the measured selection inflation above. A 396-combination
search manufactured ~0.4–0.5 Sharpe from noise alone; a 12-combination
search reduces that inflation by roughly an order of magnitude in effective
multiplicity.

### 3. Holdout

Data from **2026-01-01** onward is unlocked **exactly once**, at the very
end, on the **single** strategy that has already passed every other gate.

- Not once per strategy. Once for the project.
- If the holdout is unlocked, **the project ends with that result either way
  — pass or fail. There is no second look.**

### 4. No re-runs for taste

A walk-forward that has been run **may not be re-run** with a modified grid,
window configuration, or universe because the result was disappointing.

Any such change constitutes a **new experiment** and consumes one of the 2
strategy slots in rule 1. Every run is logged with a timestamp and the commit
hash of the code that produced it.

### 5. Gate criteria — fixed now, unchangeable

A strategy passes only if **all** of the following hold:

1. **Excess return.** Concatenated out-of-sample excess return over
   buy-and-hold, after all costs, must be **positive**. For a multi-asset
   strategy the benchmark is *both* an equal-weight hold of the
   point-in-time universe *and* buy-and-hold BTC. **It must beat both.**
2. **Degradation ratio.** Out-of-sample Sharpe ÷ in-sample Sharpe **> 0.5**.
3. **Shuffled control.** Out-of-sample Sharpe must exceed the
   shuffled-returns control by **at least 0.4**.
4. **Window robustness.** The result must hold across **at least 2 of 3**
   window configurations.
5. **Parameter stability.** Fewer than **30%** of boundary transitions may
   move a parameter by more than 2 grid steps.

### 6. Live trading precondition

**No live capital** is committed until a strategy passes all of rule 5 **and**
the holdout of rule 3.

---

## Signed

**Michael Nicasens**, 2026-08-10.

Committed before any strategy #2 code was written. The binding record is the
commit that introduces this file; its parent is `ac43dfc` (Stage 4a results),
and the repository contains no cross-sectional momentum implementation at
that point in history — verifiable with `git log` and `git show`.

---

# Amendments

Amendments are appended, never rewritten in place. Each is committed alone
and before the work it governs, so the record shows the decision was taken
in advance of any result rather than in response to one.

## Amendment 1 — 2026-08-10

**Binding record:** the commit that introduces this amendment. Its parent is
`33ab9ce` (the original preregistration). A commit cannot contain its own
hash, so the hash is recorded in the git history rather than in this text;
the ordering claim below is verifiable with `git log` and `git show`.

**Cross-sectional momentum was designated strategy #2. It is abandoned
before any backtest was run, on data grounds.**

The dataset contains 2 EUR pairs against a required 10. More fundamentally,
survivorship bias in free exchange archives is **identifiable but not
correctable**: delisted pairs' price series through their decline are
unrecoverable from this source, and the strategy's edge claim lives entirely
inside that region. A delisting list would name what is missing; it would not
supply the missing returns.

Investigation session only. **No results produced, no gate reached.**

### Ruling

**This does not consume a strategy slot.** Rule 4 guards against re-running
after seeing a disappointing result; no result exists. Nothing was fitted,
nothing was observed, and there is therefore nothing to be tempted by.

**Strategy budget remains 2.**

### Redesignation

Strategy #2 is hereby redesignated as: **[to be filled after selection]**

This amendment is committed **BEFORE** that strategy is chosen or written.
The placeholder above is deliberate and is itself part of the record: at the
time of this commit the replacement strategy was not yet selected. Filling it
in requires a further amendment, committed before that strategy's code.

## Amendment 2 — 2026-08-10

**Binding record:** the commit that introduces this amendment. Its parent is
`9d0fae8` (amendment 1). Committed **before any strategy #2 code exists** —
verifiable with `git log` and `git show`: at this commit the repository
contains no volatility-targeted trend implementation.

Filling the placeholder left by amendment 1:

> **Strategy #2 is hereby redesignated as: volatility-targeted trend on
> daily XBTEUR and ETHEUR.**
>
> Rationale: strategy #1's binary sizing paid full round-trip fees at every
> whipsaw and held full exposure through high-volatility periods where
> drawdowns were made. Vol-targeting addresses both mechanisms while using
> the trend effect already demonstrated real by the shuffle control (OOS
> Sharpe 0.91/1.20 vs 0.36/0.54 shuffled).

### Specification, fixed before any result

- **Direction** from a trend filter; **size** from inverse realized
  volatility. Target position is continuous in [0.0, 1.0]: zero when the
  trend filter is off, otherwise `min(target_vol / realized_vol, 1.0)`.
- **Grid, exactly 12 combinations** (rule 2 cap, at the limit, not over):
  `trend_lookback ∈ {50, 100, 200}` × `vol_lookback ∈ {20, 60}` ×
  `target_vol ∈ {0.40, 0.60}`.
- Costs, boundaries and gate criteria are unchanged from rule 5 and the
  Stage 4a protocol: liquidity floor T=500, end at holdout minus one day,
  180/60/60 main plus 90/30 and 360/90, selection by in-sample Sharpe,
  shuffled-returns control, position continuity across window boundaries.

### Slot accounting

This consumes **strategy slot 1 of the remaining 2**. One slot remains after
this run, whatever its outcome. Rule 4 applies from the moment it is run: the
grid and window configurations above may not be revised in response to the
result.
