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

## Amendment 3 — 2026-08-10

**Binding record:** the commit that introduces this amendment. Parent is
`c5c823e` (strategy #2 gate results).

### Grid discrepancy, surfaced not resolved silently

A later-authored specification for strategy #2 proposed a different
12-combination grid — `trend_lookback [50,100,150,200]` ×
`target_vol [0.40,0.60,0.80]` with `vol_lookback` fixed at 30 — from the one
actually run at `e611481` (`trend_lookback [50,100,200]` ×
`vol_lookback [20,60]` × `target_vol [0.40,0.60]`). That specification was
written **before** the strategy #2 results were seen, but would have been
**adopted after**.

**Ruling: the completed run at `e611481` / `c5c823e` stands as strategy #2,
final.** Re-running under the alternative grid would be a new experiment
under rule 4 and would consume the final strategy slot. It was declined.

**Slot accounting unchanged: 1 of 2 consumed, 1 remains.**

### Holdout: not unlocked

An instruction to unlock the holdout was declined on two independent grounds:

1. **Rule 3 bars it.** The holdout unlocks only on a strategy that has passed
   every other gate. Strategy #2 failed criteria 1 and 4 on both pairs.
2. **The holdout is empty.** Data ends 2025-12-31; candles on/after
   2026-01-01 number **zero** across all four datasets. The REST top-up that
   would fetch them remains blocked by the sandbox egress proxy. Any holdout
   "result" would have been fabricated.

`HoldoutUnlock` was never minted. **The single permitted use remains
available.**

### Strategy #2 outcome (recorded)

Gate **FAILED**. Criteria 2, 3 and 5 passed; criteria 1 and 4 failed on both
pairs. Concatenated out-of-sample excess over buy-and-hold: **−7,219 pp
(XBTEUR), −16,833 pp (ETHEUR)**; 0 of 3 window configurations beat
buy-and-hold on either pair. Full record:
`results/4a_strategy2_voltrend_20260810T230133Z_e611481.json`.

Note on criterion 5: it passed only because a 3×2×2 grid makes a >2-step move
reachable on one axis alone. Parameters changed at 49/70 and 41/57 boundaries.
**This PASS is not evidence of a stable parameter regime.**

Supporting analyses, report-only, no selection and no slot consumed:
`results/4b_strategy2_voltrend_20260810T230858Z_c5c823e.json` (sensitivity),
`results/4c_strategy2_voltrend_20260810T231014Z_c5c823e.json` (EUR 100).

## Amendment 4 — 2026-08-11

**Binding record:** the commit that introduces this amendment. Parent is
`c4c634a` (Stage 7 deploy artifacts). Committed **alone and before any
re-run of 4a**, verifiable with `git log` and `git show`.

### Benchmark redefinition, forward-looking only

Criterion 1 (positive excess over buy-and-hold) and criterion 4 compare a
strategy against **100% buy-and-hold**. For any strategy capped below 100%
allocation this is **unpassable by construction** over a period where the
asset rose ~160x.

This is demonstrable from arithmetic alone and is independent of any result:
a portfolio holding at most `k` of equity, with the remainder in cash earning
zero, has terminal wealth bounded below a portfolio holding 1.0 of the same
asset whenever that asset appreciates. No skill in timing recovers a 160x
gap from a 0.25 cap. The criterion measured **allocation, not strategy**. The
defect would have been equally present had a strategy passed it.

**Benchmark redefined for FUTURE runs to CAPITAL-MATCHED:**

    benchmark = strategy_max_allocation x buy-and-hold
              + (1 - strategy_max_allocation) x cash earning zero

Criteria **2, 3 and 5 are ratio-based and unchanged**. Sharpe, the
degradation ratio and the shuffled-returns control are invariant under a
uniform rescaling of position size — mean and standard deviation both scale
by `k`, so the ratio cancels — and they therefore never carried this defect.

### SCOPE — this redefinition does NOT apply retroactively

Run `4a_strategy2_voltrend_20260810T230133Z_e611481` executed at an
**effective allocation of 1.0**: `strategy_max_allocation` did not exist at
that commit, and the order translator's `max_allocation` defaulted to 1.0.
Criterion 1 was therefore a **fair test at 100% vs 100%**, and strategy #2
failed it by **-7,219 pp (XBTEUR)** and **-16,833 pp (ETHEUR)**.

That result **stands**. This redefinition:

- does **not** rehabilitate it;
- does **not** permit a re-run of strategy #2 under the new benchmark;
- does **not** restore its consumed slot.

### Slot status

| Strategy | Outcome | Slot |
|---|---|---|
| Cross-sectional momentum | abandoned pre-result (Amendment 1) | **none consumed** |
| #2 — volatility-targeted trend | **FAILED** (criteria 1 and 4) | **1 consumed** |

**1 of 2 consumed. 1 slot remains.**

### Ruling on this amendment

**Does not itself consume a slot.** It corrects a defective criterion for
future use and explicitly preserves the failed result it might otherwise be
read as revising. The correction rests on outcome-independent arithmetic, and
the amendment forecloses the only benefit a post-hoc revision could confer —
a second look at strategy #2 — by barring that re-run in the scope clause
above.

## Amendment 5 — 2026-08-11

**Binding record:** the commit that introduces this amendment. Parent is
`57efb14` (FINDINGS.md). This is the terminal amendment.

### Project concluded — ruling (b)

The preregistered question is answered: **buy-and-hold wins on these assets
at this horizon and cost structure.** Two strategies were tested under the
gate; both failed criterion 1 on both pairs across every window
configuration. The full record, the disclosure of the walk-forward selection
defect found during post-hoc verification, the corrected degradation
figures, and the unfixed defect register are in `FINDINGS.md` at `57efb14`.

Rule 1 anticipated this outcome: "If both fail, the project concludes that
buy-and-hold wins and no live trading occurs." Both tested strategies
failed. The project concludes accordingly. **No live trading occurs.**

### Slot accounting — final

| Strategy | Outcome | Slot |
|---|---|---|
| Cross-sectional momentum | abandoned pre-result (Amendment 1) | none consumed |
| #2 — volatility-targeted trend | FAILED (criteria 1 and 4) | 1 consumed |
| — | — | **slot 2 of 2 BANKED UNSPENT** |

**Banking the final slot was a deliberate ruling, not an oversight.** A
candidate for it existed and was argued in the record (hold-by-default with
a volatility-regime exit, mechanism stated before any backtest, 12-combo
grid specified): it was declined at an honestly estimated ~25–30% pass
probability, against a deployable benefit at current capital of
approximately zero, and with the selection layer disqualified until
repaired. The slot remains available to a future reopening of the project
under the conditions in `FINDINGS.md` §9 (capital above ~EUR 500, a
reconsidered dead-band, a repaired selection layer and capital-matched
benchmark, or a different data source). Spending it before those conditions
hold would contradict the reasoning recorded here.

### Holdout — final

The 2026-01-01 holdout was **never unlocked**. Rule 3's single permitted use
expires with the project, unused. The holdout region contains zero candles
(data ends 2025-12-31), so no unlock could have produced a result in any
case.

### Standing constraints on any revival

A revival inherits this preregistration in full: the amendments are
append-only, completed runs are final (Amendments 3 and 4), the grid cap is
12, and the banked slot is single-use. The walk-forward selection layer and
the capital-matched benchmark (FINDINGS.md §3 and F6) must be repaired and
their repairs tested before any new gate run is trustworthy.

## Amendment 6 — 2026-08-11

**Binding record:** the commit that introduces this amendment. Parent is
`9cb57ef` (Amendment 5). Committed **alone and before any strategy #3 code
exists** — verifiable with `git log` and `git show`: at this commit the
repository contains no mean-reversion implementation.

### Reopening — condition invoked

Amendment 5 concluded the project and barred spending the banked slot
"before those conditions hold" (FINDINGS.md §9). This amendment reopens the
project under **§9 condition 3**, which is satisfied as a precondition of
the run itself:

- the walk-forward selection layer (FINDINGS.md §3: `search_window` /
  `select_best`) will be repaired and regression-tested **before** the gate
  runs — no combo whose warmup meets or exceeds the in-sample window may
  enter selection, so no fabricated 0.0 Sharpe can win an argmax;
- the capital-matched benchmark (Amendment 4; defect F6) will be
  implemented and tested **before** the gate runs, and criterion 1 is
  evaluated against it.

If either repair is not in place and tested, the gate run is void and the
slot is NOT consumed. Once the gate runs, **slot 2 of 2 is CONSUMED,
whatever the outcome. There is no slot 3.**

### Strategy #3 designation: mean reversion, long-or-flat

Filling the final slot: **mean reversion on daily XBTEUR and ETHEUR** —
the "buy dips, sell recovery" hypothesis. Mechanism stated before any
backtest: liquidation cascades and panic selling push price transiently
below short-horizon fair value; a buyer of K-sigma dips who exits on
reversion to the mean harvests the rebound. This is the mirror of the two
failed trend strategies: they bought strength; this buys weakness.

### Specification, fixed before any result

All choices below are frozen now. Nothing may be revised after a number is
seen (rule 4).

- **Signal** (pure function, candles in → target in {0.0, 1.0}, no I/O,
  no look-ahead; the signal at close of bar t uses data through bar t
  only; the engine fills at the open of t+1 as always):
  - MA(t) = simple moving average of close over the trailing N bars,
    **including bar t**; σ(t) = sample standard deviation (ddof=1) of the
    same N closes. Both undefined during warmup.
  - **K is in standard deviations** (chosen and fixed now, per the
    instruction to pick one). E is in the same units.
  - ENTRY: flat and close(t) < MA(t) − K·σ(t) → target 1.0.
  - EXIT: long and close(t) ≥ MA(t) − E·σ(t) → target 0.0.
    E = 0.0 therefore means "exit when close reaches the MA".
  - Otherwise hold the previous target (hysteresis — the strategy is a
    two-state machine; position state carries across walk-forward
    boundaries exactly as Donchian's did, via a parameter schedule).
  - Warmup = N bars; signal 0.0 during warmup, never NaN.
  - Long-or-flat, no leverage, no shorting.
- **Grid — exactly 12 combinations** (rule 2 cap, at the limit, not over):
  N ∈ {10, 20, 30} × K ∈ {1.0, 1.5} × E ∈ {0.0, 0.5}. No fourth axis.
- **Windows:** the same three configurations as strategies #1 and #2 —
  180/60 (main), 90/30, 360/90. Selection metric: in-sample Sharpe, as
  before. Grid steps for the stability criterion: N: 10, K: 0.5, E: 0.5.
- **Allocation:** the run executes at `strategy_max_allocation = 0.25`,
  the deployed configuration. Criterion 1's benchmark is capital-matched
  at the same 0.25 (Amendment 4). Uncapped buy-and-hold BTC is REPORTED
  alongside on every output, never used as the pass/fail benchmark.
- **Costs:** unchanged from rule 5 — taker 26 bps, slippage 5 bps,
  `min_rebalance_delta` 0.05 (inert for a binary strategy: entries move
  0.25 of equity; full exits are exempt).
- **Data:** real Kraken daily, liquidity floors 2013-12-03 (XBT) /
  2016-01-27 (ETH), end strictly before 2026-01-01. **The holdout stays
  sealed.** It does not unlock on this run regardless of outcome.
- **Shuffle control:** same construction and pipeline as before, seeds
  fixed now: XBTEUR = 42, ETHEUR = 43 (single draw per pair, same n=1
  limitation as prior runs, on record).
- **Output:** `results/4c_strategy3_meanrev_<UTCstamp>_<githash>.json`
  with per-combo, per-window, per-asset numbers and an explicit PASS/FAIL
  per criterion. (Prefix `4c_strategy3_meanrev` per the instruction; note
  the prior gate runs used a `4a_` prefix — recorded to prevent confusion.)

### Gate criteria for this run (mapping to rule 5 + Amendment 4)

1. **Reporting:** capital-matched benchmark AND uncapped buy-and-hold BTC
   on every report (requirement, not a pass/fail test).
2. **Shuffle:** selected OOS Sharpe − shuffled OOS Sharpe ≥ 0.4 (rule 5.3).
3. **Degradation:** OOS Sharpe ÷ mean in-sample Sharpe of selected combos
   > 0.5, computed on the REPAIRED selection layer (rule 5.2).
4. **Excess:** concatenated OOS excess return over the CAPITAL-MATCHED
   benchmark, after all costs, must be positive (rule 5.1 as amended).
5. **Window robustness:** criterion 4 must hold in ≥ 2 of 3 window
   configurations (rule 5.4).
6. **Parameter stability** (rule 5.5): recorded honestly as follows. On a
   3×2×2 grid no axis CAN move more than 2 grid steps (max moves: N = 2
   steps, K = 1, E = 1), so the ">2 steps in <30% of transitions" test is
   **vacuously passable and will be reported as VACUOUS, not as evidence
   of stability** — extending Amendment 3's precedent. The fraction of
   boundaries where any parameter changed is reported descriptively
   alongside.

**PASS requires criteria 2, 3, 4, 5 all true.** Criterion 6 cannot
contribute a pass on this grid and can only be reported. `live_approved`
additionally requires rule 6 (holdout), which this run cannot trigger.

### Bindings restated

No re-run with a modified grid, window, or universe after seeing the
result. No cherry-picking a window configuration. The outcome — pass or
fail — is recorded as a further amendment. The €100 executability finding
for ETHEUR (unopenable below ~€101 at 0.25) is reported with the results;
the gate itself runs at the standard 10,000 starting capital as before.

## Amendment 7 — 2026-08-11

**Binding record:** the commit that introduces this amendment. Parent is
`67754aa` (strategy #3 gate results). This is the terminal amendment; the
strategy budget is exhausted and no further amendment can authorise a run.

### Strategy #3 outcome (recorded)

Gate **FAILED — every operative criterion, on both pairs.** Run at
`652beb0`, allocation 0.25, capital-matched benchmark (Amendment 4), on the
repaired selection layer (Amendment 6 preconditions verified: repairs landed
and tested at `1f8849a` before the run; zero ineligible selections occurred;
the finite-score assertion never fired). Full record:
`results/4c_strategy3_meanrev_20260811T185700Z_652beb0.json`.

Main configuration, concatenated out-of-sample:

- **Criterion 4 (excess vs capital-matched): −4,034 pp (XBTEUR),
  −5,054 pp (ETHEUR).** The strategy lost 36.2% / 38.3% in absolute terms
  over a span where its own 25%-capped benchmark returned +3,998% / +5,015%.
- **Criterion 2 (shuffle): margins −0.832 / −0.846** against a required
  ≥ +0.4. The strategy EARNED +64% / +78% on the shuffled series and lost on
  the real ones — dip-buying profits on a random walk by construction, and
  the real series actively punishes it.
- **Criterion 3 (degradation): −0.248 / −0.288** (OOS Sharpe negative while
  selected in-sample Sharpe averaged +1.0 to +1.9).
- **Criterion 5 (window robustness): 0/3 configurations positive, both
  pairs.**
- Criterion 6 (stability): VACUOUS-PASS as pre-declared in Amendment 6;
  contributes nothing.

### The project's complete answer

Three strategies now bracket the hypothesis space from both sides. The trend
family (#1, #2) beat its shuffle controls — the momentum information in
daily BTC/ETH is real — and still lost thousands of points to holding after
costs. The mean-reversion family (#3) lost to its own shuffle control —
betting against that same structure loses outright. **Daily BTC/ETH exhibits
positive autocorrelation that is too weak to trade profitably with and
costly to trade against. Buy-and-hold wins. No live trading occurs.**

### Slot accounting — final and exhausted

| Strategy | Outcome | Slot |
|---|---|---|
| Cross-sectional momentum | abandoned pre-result (Amendment 1) | none consumed |
| #2 — volatility-targeted trend | FAILED (criteria 1, 4) | 1 consumed |
| #3 — mean reversion | FAILED (criteria 2, 3, 4, 5) | **2 consumed** |

**2 of 2 consumed. The strategy budget is exhausted. There is no slot 3,
and rule 4 makes every completed run final.**

### Holdout — permanently sealed

Rule 3 unlocks the holdout exactly once, on the single strategy that passed
every other gate. No such strategy exists and, with the budget exhausted,
none can. **The holdout is permanently sealed.** (It also contains zero
candles; both facts are independently sufficient.)

### Terminal state

The conclusion of Amendment 5 stands, strengthened: reopened once under its
own recorded conditions, the project reached the same answer by the opposite
route. FINDINGS.md carries the dated addendum. This preregistration is now
closed to further strategy work; the append-only record remains the
authoritative history of what was claimed, when, and what happened.
