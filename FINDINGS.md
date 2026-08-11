# FINDINGS — project concluded

**Ruling (b), 2026-08-11: buy-and-hold wins on these assets at this horizon
and cost structure. The project concludes with that answer.**

This document is the terminal record. Every number in it is traceable to a
committed results file and a commit hash. Nothing reported here was fixed,
re-run, or reweighted after being seen; where the machinery itself was found
defective, the defect is disclosed below rather than repaired into silence.

---

## 1. The question and the answer

**Preregistered question** (PREREGISTRATION.md, commit `33ab9ce`): can a
rule-based strategy on daily XBTEUR/ETHEUR spot, long-only, unlevered, after
Kraken fees and slippage, beat buying and holding — demonstrated by a
walk-forward gate fixed before any result was seen?

**Answer: no.** Two strategies, six window configurations, two assets —
every concatenated out-of-sample excess return over buy-and-hold was
negative, by margins between −1,194 and −21,034 percentage points. This held
under a hostile post-hoc audit of the machinery that produced it (§3): the
configuration least affected by the one genuine harness defect found failed
worst.

The answer is scoped, not universal: **these** assets (BTC, ETH in EUR),
**this** horizon (daily bars, 2013–2025), **this** cost structure (16/26 bps
+ 5 bps slippage; a EUR 100 account would in reality pay the 25/40 bps tier,
widening every failure), **this** constraint set (long-only spot, no
leverage). §9 lists what would have to change before the question deserves
reopening.

---

## 2. The record

| Strategy | Implementation | Gate run | Results file | Verdict |
|---|---|---|---|---|
| #1 Donchian breakout (ATR trail) | `670c658` | `846b5ec` | `results/4a_20260810T222700Z_846b5ec.json` | **FAILED** — criteria 1, 2 (XBT), 4, 5 |
| #2a Cross-sectional momentum | — (never written) | — | — | **ABANDONED pre-result** (Amendment 1, `9d0fae8`); no slot consumed |
| #2 Volatility-targeted trend | `e611481` | `c5c823e` | `results/4a_strategy2_voltrend_20260810T230133Z_e611481.json` | **FAILED** — criteria 1, 4 |

Supporting analyses (report-only, no selection): `results/4b_…c5c823e.json`
(sensitivity), `results/4c_…c5c823e.json` (EUR 100 / fee-tier sensitivity).

Per-criterion detail, main configuration (180/60 windows, selection by
in-sample Sharpe), concatenated OOS:

| | Donchian XBT | Donchian ETH | voltrend XBT | voltrend ETH |
|---|---|---|---|---|
| 1. Excess vs B&H | −119.17 (−11,917 pp) ✗ | −11.94 (−1,194 pp) ✗ | −72.19 (−7,219 pp) ✗ | −168.33 (−16,833 pp) ✗ |
| 2. Degradation > 0.5 | 0.477 ✗ | 0.607 ✓ | 0.937* ✓ | 0.836* ✓ |
| 3. Shuffle margin ≥ 0.4 | 0.549 ✓ | 0.667 ✓ | 1.293 ✓ | 0.559 ✓ |
| 4. ≥ 2 of 3 configs | 0/3 ✗ | 1/3 ✗ | 0/3 ✗ | 0/3 ✗ |
| 5. Param stability | 31.4% jumps ✗ | 31.6% jumps ✗ | ✓ (weak; Amendment 3) | ✓ (weak; Amendment 3) |

\* Recorded values; overstated by the defect disclosed in §3. Corrected
figures there. The voltrend run predates `strategy_max_allocation` and
executed at an effective allocation of 1.0, so its benchmark comparison was
a fair 100%-vs-100% (Amendment 4 records this; the result is final and not
re-runnable under the redefined benchmark).

The single positive cell anywhere in the record — Donchian ETHEUR at 90/30,
+831.03 — is one cell out of twelve strategy-pair-config combinations,
produced by the config with the most selection noise (145 windows of 30
days), and is treated as artifact per the preregistration's
multiple-comparison stance.

The 2026 holdout was **never unlocked**. Its single permitted use expires
with the project, unused. (The holdout region also contains zero candles —
data ends 2025-12-31 — so no result could have come from it either way.)

---

## 3. DISCLOSURE — the walk-forward selection defect

Found 2026-08-11 during the independent verification that preceded this
conclusion, i.e. **after** both gate runs. Disclosed, not repaired: the
harness at `846b5ec`/`e611481` is what produced the recorded numbers.

**The defect.** `search_window` (`src/trading_bot/backtest/walkforward.py`)
evaluates each grid combo on the in-sample slice **alone**, with no prior
history for indicator warmup. A combo whose warmup meets or exceeds the
window length — voltrend `trend_lookback=200` in a 180-candle window;
`trend_lookback∈{100,200}` in a 90-candle window — emits all-zero signals,
produces a flat equity curve, and receives a **fabricated in-sample Sharpe
of 0.0** ("no evidence" scored as "neutral evidence"). `select_best` is a
bare argmax, so whenever every evaluable combo scored negative, a fabricated
zero **won the selection**, and the walk-forward deployed a configuration
whose in-sample evaluation bore no relation to its out-of-sample behaviour
(which is computed with full history and trades actively).

Fingerprint, from the committed results: in the 90/30 config, voltrend
selected `trend_lookback=100` — structurally unevaluable there — in 47/145
(XBT) and 38/118 (ETH) windows, and `trend_lookback=200` exactly zero times,
which is precisely the argmax tie-break landing on the first dead combo in
grid order. 8 of 12 combos were unevaluable in that config; 4 of 12 in the
main config.

**Effect on recorded numbers.** The fabricated zeros deflate the mean
in-sample Sharpe that forms the degradation-ratio denominator, so **recorded
degradation ratios are overstated**. Corrected (zero-Sharpe selections
excluded from the denominator):

| config | voltrend XBT | voltrend ETH | Donchian XBT | Donchian ETH |
|---|---|---|---|---|
| main_180_60 | 0.937 → **0.633** | 0.836 → **0.577** | 0.477 → **0.436** | 0.607 → **0.586** |
| w_90_30 | 1.070 → **0.524** | 1.127 → **0.535** | 0.532 → **0.400** | 0.775 → **0.591** |
| w_360_90 | 0.845 → **0.790** | 0.794 → **0.751** | 0.544 (0 zeros) | 0.655 (0 zeros) |

(Donchian had no structurally unevaluable combos — its warmup ≤ 60 fits
every window — but the same argmax pathology applied to combos that simply
never triggered in-window; its corrections above use the same method.)

**Why the verdicts survive.** Criterion 2's passes for voltrend survive
correction in all six cells (minimum 0.524), and Donchian's criterion-2
failure becomes *more* decisive (0.400). Criterion 1 — the criterion both
strategies died on — does not touch the defective denominator at all; and
the 360/90 configuration, in which **all twelve combos were genuinely
evaluable** (0–3 zero-selections), produced the **worst** failures in the
record: −17,986 pp (XBT) and −21,034 pp (ETH) for voltrend. No correction of
the selection layer recovers gaps of that size. Criterion 5's recorded
instability is partly artificial (bouncing between live and dead combos at
boundaries) — direction: harsher on the strategy — and both strategies'
criterion-5 outcomes were already discounted or failed.

The selection layer is **disqualified for any future gate run** until
repaired (§9). It is not repaired in this repository.

---

## 4. The structural finding: the dead-band caps every capital level at 4–5 position levels

`min_rebalance_delta` is 0.05 **of equity**. Under the risk-budget mapping,
the strategy's expressible range is `strategy_max_allocation` × equity =
0.20–0.25 of equity. The smallest permitted adjustment is therefore a
constant **20–25% of the expressible range at every capital level, forever**
— at EUR 100 or EUR 5,000,000, a fractional-allocation strategy has at most
4–5 expressible position levels. Capital only decides whether exchange
minimums bind *harder* than that floor (below ~EUR 505 they do).

The consequence for the record: the walk-forward validated voltrend at an
effective allocation of 1.0, where the dead-band was 5% of the expressible
range (~20 levels of granularity). The deployed configuration permits 4–5.
**The validated object cannot be expressed by the deployed config at any
capital.** Changing the dead-band is a deliberate re-specification requiring
re-validation, not a tweak — recorded here so it is decided, if ever, with
open eyes.

---

## 5. Executability at small capital

Prices are the last closes in the data (2025-12-31: XBT 74,500.10 EUR, ETH
2,525.99 EUR — the development sandbox has no route to Kraken for live
prices). Constants from `schema.py`: ordermin 0.0001 BTC / 0.01 ETH, costmin
1 EUR; taker 26 bps + 5 bps slippage.

| capital | alloc | pair | full position | openable? | smallest change | % of range | levels |
|---:|---:|---|---:|---|---:|---:|---:|
| 100 | 0.20 | XBTEUR | 20.00 € | yes | 7.45 € (ordermin) | 37.3% | 2.7 |
| 100 | 0.20 | ETHEUR | 20.00 € | **NO** | — | — | 0 |
| 100 | 0.25 | XBTEUR | 25.00 € | yes | 7.45 € (ordermin) | 29.8% | 3.4 |
| 100 | 0.25 | ETHEUR | 25.00 € | **NO** (0.00987 < 0.01 ETH) | — | — | 0 |
| 500 | 0.25 | both | 125.00 € | yes | ~25 € | ~20% | ~5 |
| 1,000 | 0.25 | both | 250.00 € | yes | 50 € (dead-band) | 20% | 5 |
| 5,000 | 0.25 | both | 1,250.00 € | yes | 250 € (dead-band) | 20% | 5 |

**ETHEUR is untradeable below ~EUR 101 at 0.25 allocation (~EUR 127 at
0.20): a full position is smaller than Kraken's minimum order and is never
submitted.** At EUR 100 the deployable strategy space is therefore
"BTC only, roughly three position levels" — binary in/out plus one
intermediate step. Ordermin stops being the binding constraint at ~EUR 149
(XBT) and ~EUR 505 (ETH); beyond that, §4's config-permanent floor governs.

---

## 6. Findings that hold regardless of outcome

- **Real trend information exists in these series.** Both strategies cleared
  the shuffled-returns control with preregistered margins: Donchian OOS
  Sharpe 0.907/1.205 vs 0.358/0.538 shuffled; voltrend 1.339/1.187 vs
  0.046/0.628. The information is real; after costs and time-out-of-market
  it was insufficient to beat holding. (Caveat on record: each control is a
  single shuffle draw per pair, an n=1 estimate of the null.)
- **Selection inflation is large and was measured, not assumed.** The
  396-combination Donchian search manufactured ~0.4–0.5 of out-of-sample
  Sharpe from noise alone (shuffled-data selection still produced 0.358 /
  0.538). This measurement motivated the 12-combination grid cap
  (preregistration rule 2) that governed strategy #2.
- **Survivorship bias in free exchange archives is identifiable but not
  correctable** (Amendment 1 investigation): delisted pairs' price series
  through their declines are unrecoverable from these sources, and a
  cross-sectional strategy's edge claim lives entirely inside that missing
  region. A delisting list names what is absent; it cannot supply the
  missing returns. Cross-sectional momentum was abandoned on these grounds
  before any backtest, consuming no slot.

---

## 7. Defect register — UNFIXED at conclusion

Found in the 2026-08-11 fresh-eyes audit; each was demonstrated by executing
the shipped code, none has been repaired. Anyone reviving this repository
starts here. The walk-forward selection defect (§3) sits above all of these
in consequence.

| # | Sev | Location | Defect and production manifestation |
|---|---|---|---|
| F1 | **HIGH** | `scheduler/cycle.py` `_mirror_fills_locally` | Partial fills are mirrored at the venue's *blended average* price, not the leg's actual price. One order filling in two legs at two prices drifts local cash from venue cash (demonstrated: +45.71 EUR from a single two-leg fill) → every-cycle reconciliation reports a mismatch → **spurious HALT** with books that genuinely disagree. With the shipped 10% partial-fill rate, paper trading halts within weeks and destroys trust in the one alarm that must never cry wolf. |
| F2 | **MED** | `broker/paper.py` (`_pending_fills`) | Scheduled fills live in an in-memory dict. A restart between submit and settle strands the order OPEN forever — it can never fill, the position never arrives, and reconciliation **agrees** (both sides show the same open order). Silent under-exposure plus zombie orders that accumulate across restarts. |
| F3 | **MED** | `scheduler/cycle.py` + `risk/gate.py` | In-flight orders are invisible to position sizing and to the risk gate (`AccountState` has no open-order field; cash is not reserved at submit; no account-scoped per-pair position check). Demonstrated: with fill latency beyond the cycle interval, three stacked full-size orders produced a **74.8% position against a 25% cap** with no limit firing. Under shipped paper defaults the realistic bound is ~1.5× the cap via partial remainders, ~10% of trading days, self-correcting after one cycle. Materially more dangerous the day a real asynchronous broker replaces the paper one. |
| F4 | LOW | `run.py` / `scripts/check_heartbeat.py` / `tests/test_ops.py` | The heartbeat's `halted` field is unreachable in production: every halt path raises before the heartbeat is written, so the checker's exit-2 ("HALTED") branch is dead code and its test fabricates the file by hand. After a real halt the operator gets the wrong alarm class (STALE, exit 1). **Introduced during the very session that audited for mechanism-vs-outcome tests — recorded as evidence of how easily the pattern reproduces, even under explicit vigilance.** |
| F5 | LOW | `execution/translate.py` | The full-exit exemption bypasses the dead-band but not ordermin: a position remainder below 0.0001 BTC / 0.01 ETH can never be closed. Permanent dust exposure; `ends_flat` never true. |
| F6 | LOW now, **CRITICAL if a slot were ever spent** | `backtest/metrics.py`, `scripts/run_4a*.py` | Amendment 4's capital-matched benchmark exists only in the preregistration. `excess_return` is still computed against 100% buy-and-hold; a future gate run at allocation < 1.0 would mechanically evaluate criterion 1 against the superseded benchmark. |

---

## 8. The infrastructure, and how to reuse it for a different question

What exists and is tested (481 tests, network-free):

- **Data layer** — Kraken OHLCVT dump ingestion + ccxt REST top-up, mapped
  by column name; parquet store with atomic writes; validation (gaps,
  duplicates, monotonicity) with a no-forward-fill policy; per-source
  provenance column; liquidity-floor detection.
- **Backtest engine** — the execution-timing rule (signal N fills at open
  N+1) in exactly one place, guarded by an adversarial look-ahead test;
  mark-to-market accounting with proportional cost-basis retirement;
  reconciliation invariants; a prefix-vs-full causality gate every
  registered strategy must pass.
- **Walk-forward** — position continuity via parameter schedules over one
  continuous engine run (subject to §3's selection defect).
- **Execution stack** — single-sourced `FillModel`; order translation with
  the risk-budget mapping in one shared code path; a risk gate whose
  approvals are unforgeable (closure-held mint key + HMAC); durable SQLite
  state with crash-tested single-transaction fills; broker-agnostic
  reconciliation; a cycle loop with account-scoped risk checks; deploy
  artifacts and a runbook that documents only commands that exist.
- **Governance** — the preregistration pattern: binding rules committed
  before results, append-only amendments, slot accounting, a sealed holdout.

To reuse for a different question: the strategy registry + causality gate
accept any pure `fn(candles, params) -> Series` in [0,1]; the data schema
accepts any OHLCVT source; the engine, gate criteria, and preregistration
template are strategy-agnostic. **Before any new gate run**: repair §3
(warmup-aware in-sample evaluation, or history-fed IS windows, and an
argmax that cannot select unevaluated combos), implement F6's
capital-matched benchmark, and fix F1 — everything else can be triaged.

---

## 9. What would have to change to revisit

Any one of these reopens the question honestly; none existed at conclusion:

1. **Capital above ~EUR 500** — both pairs tradeable and exchange minimums
   no longer the binding granularity (though §4's dead-band floor remains
   until reconsidered).
2. **A reconsidered dead-band** — a deliberate re-specification (with
   re-validation) that restores fractional expressiveness at the deployed
   allocation; not a parameter tweak.
3. **A repaired selection layer** (§3) plus the capital-matched benchmark
   (F6) — preconditions for trusting any future gate result at all.
4. **A different data source** — derivatives funding/carry, intraday bars
   with maker execution, or survivorship-complete cross-sectional data:
   families this project's spot-daily-taker frame could not test.

The banked strategy slot (Amendment 5) remains available to whoever reopens
the project under those conditions. It was banked deliberately: the last
irreversible resource was judged worth more unspent than spent on a
one-in-four shot at an undeployable result.

---

*Concluded 2026-08-11. Data: Kraken OHLCVT dumps + REST top-up,*
*2013-09-10 → 2025-12-31 (XBT, 4,491 candles), 2015-08-07 → 2025-12-31*
*(ETH, 3,800 candles); liquidity floors 2013-12-03 / 2016-01-27. Holdout*
*2026-01-01+ sealed and empty. All results under `results/`, all rulings in*
*`PREREGISTRATION.md`, amendments 1–5.*
