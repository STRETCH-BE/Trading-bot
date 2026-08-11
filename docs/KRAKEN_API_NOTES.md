# Kraken API notes — research for Stage 6b

Recorded before `KrakenBroker` was written, so the design decisions can be
traced to evidence rather than recall.

**Verified against ccxt `4.5.71`** (the version pinned in this environment).
**If ccxt is upgraded, every claim marked *installed source* below must be
re-verified** — several of them describe behaviour that changed between ccxt
releases, and one of them (`clientOrderId` → `cl_ord_id`) changed recently
enough that the public issue tracker still describes the old behaviour.

## Verification method, and its limits

| Confidence | Method | Applies to |
|---|---|---|
| **Verified — installed source** | Read `.venv/lib/python3.11/site-packages/ccxt/kraken.py` directly | ccxt field mapping, nonce |
| **Reported — search results** | Web search summarising Kraken docs; the docs pages themselves could not be opened | `cl_ord_id` formats, rate-limit decay, uniqueness scope |
| **UNVERIFIED** | Could not be established | rate-limit counter ceilings |

`docs.kraken.com`, `api.kraken.com` and `support.kraken.com` are **blocked by
this environment's egress proxy** (`EGRESS_BLOCKED` on WebFetch, `000` on
curl). Nothing here was confirmed by opening a Kraken documentation page or by
calling the API. Anything marked *reported* rests on search-result summaries
and should be re-checked from a machine with normal network access.

---

## 1. Client order identifiers — the `cl_ord_id` / `userref` asymmetry

**Status: verified against installed source.**

ccxt 4.5.71 uses **different fields in different functions**. This matters
more than it looks: an id you can submit with is not necessarily an id you can
look up with.

| ccxt function | line | field it sets | type |
|---|---|---|---|
| `order_request` (used by `create_order`) | 2077–2080 | `request['cl_ord_id']` | **string** |
| `edit_order` | 2209–2212 | `request['cl_ord_id']` | string |
| `cancel_order` | 2500–2503 | `request['cl_ord_id']` | string |
| **`fetch_order`** | 2257–2265 | **`request['userref']`** | **32-bit int** |
| `parse_order` (reading back) | 2012–2013 | reads `cl_ord_id`, falls back to `userref` | — |

```python
# order_request — submission uses the string field
clientOrderId = self.safe_string(params, 'clientOrderId')
params = self.omit(params, ['clientOrderId'])
if clientOrderId is not None:
    request['cl_ord_id'] = clientOrderId

# fetch_order — lookup uses the integer field
clientOrderId = self.safe_value_2(params, 'userref', 'clientOrderId')
...
request['userref'] = clientOrderId
```

### Consequences for our design

1. **No 32-bit truncation is needed for submission**, and therefore **no
   collision probability to bound**. Submission carries our string id
   verbatim. An earlier plan assumed `userref` and budgeted for a hash
   truncation plus a collision check; that is not required.
2. **The idempotency lookup must NOT use `fetch_order(params={'clientOrderId':…})`** —
   it would query `userref`, which we never set, and find nothing. Instead:
   fetch open and closed orders and filter client-side on the `clientOrderId`
   that `parse_order` populates from `cl_ord_id`.
3. [ccxt issue #23370](https://github.com/ccxt/ccxt/issues/23370) states that
   ccxt maps `clientOrderId` → `userref` and ignores `cl_ord_id`. **That is
   out of date for 4.5.71.** It is the clearest example of why the installed
   source is the authority here, not the issue tracker.

## 2. `cl_ord_id` accepted formats — and why our id fits none of them

**Status: reported (search results). Could not open the docs page.**

Per [Kraken's Spot Client Order Identifiers guide](https://docs.kraken.com/api/docs/guides/spot-clordid/),
`cl_ord_id` accepts exactly three shapes:

| Format | Shape | Example |
|---|---|---|
| Long UUID | 32 hex + 4 dashes | `6d1b345e-2821-40e2-ad83-4ecb18a06876` |
| **Short UUID** | **32 hex, no dashes** | `da8e4ad59b78481c93e589746b0cf91f` |
| Free text | ASCII, **≤ 18 characters** | `arb-20240509-00010` |

**Our internal id is `tb-` + 24 hex = 27 characters.** It is:

- too long for free text (27 > 18), and
- not a short UUID (has a `tb-` prefix; 24 hex, not 32).

So it fits **none** of the three and would be rejected. A Kraken-specific
derivation is required: take the same canonical SHA-256 payload used by
`make_client_order_id` and emit the **first 32 hex characters, unprefixed**.
That is a valid short UUID, is still deterministic from the same inputs, and
carries 128 bits.

The exchange id and the internal id are then two representations of one
decision, and the mapping must be a tested pure function of the same payload —
not a truncation of the internal id string, which would couple the two
formats.

Corroborating third-party evidence that the 18-character limit is real and
bites in practice: [nautilus_trader issue #3651](https://github.com/nautechsystems/nautilus_trader/issues/3651),
"Sequential ClientOrderId exceeds cl_ord_id 18-character free-text limit".

## 3. Uniqueness is enforced only across OPEN orders

**Status: reported (search results).**

> "Kraken verifies `cl_ord_id` uniqueness across open orders for each client."

Scope matters. Kraken will reject a duplicate id **while the earlier order is
still open**, but will **not** reject reuse of an id belonging to a closed,
cancelled or filled order.

**Therefore the exchange cannot be relied on to catch historical reuse — our
own state must.** Before submitting, check the local `orders` table for the
client order id as well as querying the venue. Our ids are derived from
`(strategy, pair, signal_timestamp, intent)`, so a genuine repeat means either
a legitimate retry of the same decision (adopt it) or a bug replaying an old
signal timestamp (halt) — and only local state can tell those apart once the
original order has closed.

## 4. Rate limits — counter-based, tier-dependent decay

**Status: partly reported, ceilings UNVERIFIED.**

Kraken uses a **counter that increases per request and decays continuously**,
not a fixed requests-per-window quota. A generic exponential backoff is the
wrong shape; the backoff must model the counter.

| Quantity | Value | Confidence |
|---|---|---|
| AddOrder counter cost | **1 point** | reported |
| Decay, Intermediate tier | **2.34 points/second** | reported |
| Decay, Pro tier | **3.75 points/second** | reported |
| **Maximum counter ceiling (per tier)** | **UNKNOWN** | **could not be established** |

The ceiling is what determines burst capacity, and it is exactly the number
the blocked docs page holds. Implementing a token bucket without it means
choosing a ceiling constant that is a guess.

**Recommendation:** make the ceiling and decay configurable per tier, default
conservatively, and have the Stage 6b startup assertion log the observed
values so the guess is visible rather than buried. Sources:
[Spot rate limits](https://docs.kraken.com/api/docs/guides/spot-ratelimits/),
[Trading rate limits](https://support.kraken.com/articles/360045239571-trading-rate-limits).

## 5. Nonce — millisecond resolution makes the lockfile mandatory

**Status: verified against installed source.**

```python
def nonce(self):
    return self.milliseconds() - self.options['timeDifference']
```

Kraken's private API requires a **strictly increasing** nonce per API key.
ccxt's is derived from the wall clock at **millisecond** resolution, which
gives two independent consequences:

1. **Two processes sharing one key will collide.** They are not coordinated,
   so both can emit the same millisecond, and either can run ahead of the
   other — after which the lagging process's nonces are permanently rejected.
2. **A backward clock step breaks a single process too.** If NTP steps the
   clock back, `milliseconds()` regresses and the nonce stops increasing.
   This is the same clock-drift exposure as audit finding 5, arriving through
   a different door, and it is why Stage 7 requires verified time sync.

The startup lockfile guard is therefore **necessary, not defensive**. A nonce
error at runtime should be treated as evidence that a second process is live
and is a state-corruption risk — halt immediately rather than retry.

## 6. Pair naming — three names, mapped explicitly

**Status: partly verified. The altname/wsname mapping needs live confirmation.**

Three distinct spellings are in play for each pair:

| Context | XBT | ETH |
|---|---|---|
| Kraken dump filenames / our `kraken_name` | `XBTEUR` | `ETHEUR` |
| Kraken REST response keys (altname vs. primary) | `XXBTZEUR` *(expected)* | `XETHZEUR` *(expected)* |
| ccxt unified symbol / our `ccxt_symbol` | `BTC/EUR` | `ETH/EUR` |

`kraken_name` and `ccxt_symbol` are already stored as **separate explicit
fields** in `schema.PAIRS` — neither is derived from the other, per the design
rule. The **response-key spelling is not yet stored** and is marked *expected*
because it could not be confirmed without `load_markets()`.

**Action for Stage 6b:** add a third explicit field (`kraken_response_key`)
and confirm it from `load_markets()` at startup rather than deriving it. Kraken's
`X`/`Z` asset-class prefixes are not applied uniformly across all assets, so
deriving `XXBTZEUR` from `XBTEUR` by rule would be wrong for some pairs.

## 7. Startup assertion — how the provisional constants become trustworthy

`schema.PAIRS` currently carries `ordermin`, `costmin`, `lot_decimals` and
`price_decimals` that were **recalled, then partially corroborated by search**
(ordermin BTC 0.0001 / ETH 0.01 and the 1 EUR costmin were confirmed by
search; the decimals were not). None has been checked against the API.

Stage 6b must, on construction:

1. call `load_markets()`;
2. compare all four constants per pair against
   `market['limits']['amount']['min']`, `market['limits']['cost']['min']`, and
   `market['precision']`;
3. on any mismatch **log both values, alert, and refuse to start**.

That assertion is the mechanism by which these constants stop being
provisional. Until it has run against the live API at least once, every number
in `schema.PAIRS` should be treated as unconfirmed.

---

## Re-verification checklist (run after any ccxt upgrade)

- [ ] `grep -n "cl_ord_id\|userref" .venv/lib/python3.11/site-packages/ccxt/kraken.py`
      — confirm `order_request` still sets `cl_ord_id` and `fetch_order` still uses `userref`
- [ ] `grep -n "def nonce" -A3 …/ccxt/kraken.py` — confirm resolution
- [ ] Re-open the `cl_ord_id` format guide and confirm the 18-char / 32-hex limits
- [ ] Establish the rate-limit counter ceilings (still unknown here)
- [ ] Run the Stage 6b startup assertion and record the live `schema.PAIRS` values
