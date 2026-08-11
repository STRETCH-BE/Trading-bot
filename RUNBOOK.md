# Trading-bot RUNBOOK

**Mode: PAPER. There is no live trading and no way to enable it.**
`--live` requires both the flag and `LIVE_TRADING=yes`, and then refuses anyway
because no live broker exists (Stage 6b is not built). No order in this system
reaches Kraken.

Every command below was verified against the code at the commit that ships this
file. If a command here does not work, that is a bug in this document — see
[What is NOT available](#what-is-not-available-yet) before assuming you are
holding it wrong.

---

## Paths on the VM

| What | Where |
|---|---|
| Code | `/opt/trading-bot` |
| Deployed commit | `/opt/trading-bot/DEPLOYED_COMMIT` |
| Virtualenv | `/opt/trading-bot/.venv` |
| State (SQLite) | `/var/lib/trading-bot/state/` |
| Market data (parquet) | `/var/lib/trading-bot/parquet/` |
| Backups | `/var/lib/trading-bot/backups/` |
| **HALT file** | `/var/lib/trading-bot/HALT` |
| Heartbeat | `/var/lib/trading-bot/state/heartbeat.json` |
| JSON log | `/var/log/trading-bot/trading-bot.jsonl` |
| Secrets | `/etc/trading-bot/env` (root:root, 600, currently empty) |
| Service user | `tradingbot` |

The bot is invoked with long paths. This runbook uses `$BOT` as shorthand:

```sh
BOT="sudo -u tradingbot /opt/trading-bot/.venv/bin/python -m trading_bot.run \
  --config /opt/trading-bot/config.yaml \
  --data-dir /var/lib/trading-bot/parquet \
  --state-dir /var/lib/trading-bot/state \
  --halt-file /var/lib/trading-bot/HALT"
```

---

## STOP IT NOW

Two methods. They do different things — pick deliberately.

### 1. `touch HALT` — stop trading, leave the process up

```sh
sudo -u tradingbot touch /var/lib/trading-bot/HALT
```

- The **next cycle** aborts before the strategy is even consulted, and the
  process exits with status 2.
- The order-approval gate also refuses everything while the file exists, so
  even a cycle already in flight cannot place an order.
- systemd treats exit 2 as a clean stop (`SuccessExitStatus=2`), so the service
  goes **inactive** rather than restart-looping.
- The halt **survives restart**: the bot refuses to start while the file is
  there. Only a human removing the file resumes it.

The default cycle interval is one day. `touch HALT` guarantees no further
order, but the process may not exit until the next cycle wakes up. **If you
need the process gone right now, also `systemctl stop`** (below). Doing both is
correct and normal: HALT is the durable decision, stop is the immediate action.

### 2. `systemctl stop` — stop the process, do not record a decision

```sh
sudo systemctl stop trading-bot
```

- Sends SIGTERM. The bot finishes the cycle in flight and starts no new one
  (`TimeoutStopSec=300`).
- **Does NOT set the halt.** `systemctl start`, a reboot, or anything else that
  starts the unit will resume trading immediately.

**If you are stopping because something is wrong, use both — HALT first:**

```sh
sudo -u tradingbot touch /var/lib/trading-bot/HALT
sudo systemctl stop trading-bot
```

### What a halt does NOT do

**It does not sell anything.** There is no flatten. An open position stays open
and fully exposed to the market until a human closes it, including when the
halt was triggered automatically by the daily-loss or max-drawdown limit. The
risk limits stop the bot from *trading*; they do not get you out of the market.

In paper mode that costs nothing. Do not carry the assumption into live
trading — see [What is NOT available](#what-is-not-available-yet).

### Resuming

```sh
# read WHY it halted before you delete the evidence
sudo cat /var/lib/trading-bot/HALT

$BOT --reconcile-only            # confirm the books agree
sudo rm /var/lib/trading-bot/HALT
sudo systemctl reset-failed trading-bot   # only if it hit the restart limit
sudo systemctl start trading-bot
```

---

## Diagnose

### Is it running, and is it working?

Two different questions. `is-active` answers the first; the heartbeat answers
the second — a process can be alive and wedged in a socket read forever.

```sh
systemctl status trading-bot
sudo -u tradingbot /opt/trading-bot/.venv/bin/python \
    /opt/trading-bot/scripts/check_heartbeat.py \
    --heartbeat-file /var/lib/trading-bot/state/heartbeat.json
```

Heartbeat exit codes: `0` fresh · `1` stale, missing, or unreadable · `2` the
last cycle reported a halt.

It runs hourly on its own timer (`trading-bot-heartbeat.timer`), outside the bot
process. To see its history:

```sh
journalctl -u trading-bot-heartbeat --since '2 days ago'
```

### Do the books agree?

Reconciles the bot's beliefs against the broker. Read-only, places no orders,
and **works while halted** — it is the first thing to run after any incident.

```sh
$BOT --reconcile-only
```

Exit `0` they agree, `1` they do not. If they disagree, halt and stop before
doing anything else; do not let another cycle trade on a book you do not trust.

### What would it do right now?

Runs a full cycle — settle, reconcile, data, signal, sizing, risk approval —
and logs the order it *would* place instead of submitting it. No code path
submits an order under `--dry-run`.

```sh
$BOT --once --dry-run
```

Look for `WOULD SUBMIT` in the output.

### Logs

```sh
journalctl -u trading-bot -f                 # live, human-readable
journalctl -u trading-bot --since '1 hour ago' -p warning
sudo tail -f /var/log/trading-bot/trading-bot.jsonl        # structured
sudo jq -r 'select(.level!="INFO") | "\(.ts) \(.level) \(.message)"' \
    /var/log/trading-bot/trading-bot.jsonl                 # errors only
```

The JSON log rotates at 50 MB, 10 files kept. journald applies its own limits
independently.

### Why did it do that?

Every evaluation is recorded, including the decisions to do nothing.

```sh
sudo -u tradingbot sqlite3 -header -column \
    /var/lib/trading-bot/state/local.db \
    "SELECT timestamp, pair, current_position, target_position, action_taken,
            reasoning FROM decisions ORDER BY id DESC LIMIT 20;"
```

`current_position` and `target_position` are both fractions of **equity**. A
target of `0.25` under the default config means the strategy asked for a full
allocation — the strategy's `1.0` maps through `strategy_max_allocation` to 25%
of equity. The raw signal is in the `reasoning` column, labelled.

---

## Exit codes

| Code | Meaning | systemd | What to do |
|---|---|---|---|
| 0 | Clean stop (`--once`, SIGTERM) | no restart | nothing |
| 1 | `--reconcile-only` found a mismatch | n/a | investigate before restarting |
| 2 | **HALTED** — kill switch or halt file | **no restart** (`SuccessExitStatus=2`) | read the HALT file |
| 3 | Unhandled exception; the HALT file was written | one restart, then the startup check stops it | read the traceback in the journal |

---

## Backups

Nightly at 03:17 UTC via `trading-bot-backup.timer`. Uses the SQLite
online-backup API, not `cp` — the bot runs in WAL mode and never stops, so a
file copy would miss recent commits still in the `-wal`. Each snapshot is
verified with `PRAGMA integrity_check` before it counts. 30-day retention.

```sh
systemctl list-timers 'trading-bot-*'
journalctl -u trading-bot-backup --since '3 days ago'
ls -lh /var/lib/trading-bot/backups/
```

Run one by hand:

```sh
sudo systemctl start trading-bot-backup.service
```

Restore (the bot must be stopped and halted first):

```sh
sudo -u tradingbot touch /var/lib/trading-bot/HALT
sudo systemctl stop trading-bot
sudo -u tradingbot cp /var/lib/trading-bot/state/local.db \
                      /var/lib/trading-bot/state/local.db.before-restore
sudo -u tradingbot cp /var/lib/trading-bot/backups/local_<STAMP>.db \
                      /var/lib/trading-bot/state/local.db
$BOT --reconcile-only        # MUST agree before you remove the HALT file
```

A restored local database will disagree with the paper venue if the venue moved
on after the snapshot. Reconcile before resuming, every time.

---

## Deploying a new version

```sh
cd /path/to/Trading-bot        # a clean checkout, HEAD pushed
sudo ./deploy/deploy.sh
```

The script refuses to run if the working tree is dirty, if HEAD is not on the
remote branch, or if the system clock is not NTP-synchronised. It is idempotent.
It does **not** start the bot, and does **not** copy market data (`data/` is
gitignored) — sync that separately.

After deploying, before starting:

```sh
$BOT --reconcile-only
$BOT --once --dry-run
sudo systemctl start trading-bot
```

---

## What is NOT available yet

Read this section now, not at 3am. Each item is a thing you might reasonably
expect to exist and which does not.

| Missing | What that means for you | Arrives in |
|---|---|---|
| **`/halt`, `/flatten`, `/status` commands** | There is no Telegram bot, no chat interface, and no remote control of any kind. Every action in this runbook requires an SSH session. | Stage 6d |
| **Any live trading** | `KrakenBroker` does not exist. `--live` is refused twice over. Nothing here has ever sent an order to an exchange. All fills are simulated by `PaperBroker`. | Stage 6b |
| **Automatic flatten** | Nothing ever sells a position to protect you. A daily-loss or drawdown breach halts *trading* and leaves the position open. If you want out of a real market, you place that order yourself. | not planned; needs an explicit decision |
| **Alerting / paging** | The heartbeat timer writes to the journal and exits non-zero. Nothing emails, pages, or messages anyone. If nobody reads the journal, nobody finds out. | Stage 6d |
| **Automatic data updates** | The bot reads whatever parquet is in `--data-dir`. It does not fetch new candles. Stale data trips the freshness check and the cycle refuses to trade — which looks like a broken bot. Run `tbot-data update` yourself. | not scheduled; no timer ships for it |
| **A live-market kill** | `touch HALT` stops *this bot* from placing orders. It has no effect on orders already resting at an exchange (there are none today, because there is no live broker). | with Stage 6b |
| **Config hot-reload** | `config.yaml` is read once at startup. Editing it does nothing until you restart the service. | not planned |
| **Log shipping / retention beyond the box** | Logs live on this VM only. If the VM is lost, so are they. | not planned |

### Consequence worth stating plainly

Between the missing alerting and the missing auto-update, the realistic failure
mode is **silent**: data goes stale, the freshness check refuses every cycle,
the heartbeat keeps ticking because cycles are completing normally, and the bot
sits there doing nothing for days. Nothing tells you.

Until Stage 6d, check it deliberately:

```sh
$BOT --once --dry-run     # shows the data timestamp and the decision
```

---

## Things that look wrong but are not

- **`/etc/trading-bot/env` is root-owned 600, but the bot runs as `tradingbot`.**
  Correct. systemd is PID 1 and reads `EnvironmentFile=` as root *before*
  dropping to `User=tradingbot`. The bot never needs to read the file. Making it
  readable by `tradingbot` would weaken it for no benefit.
- **`/etc/trading-bot/env` is empty.** Correct. Paper mode needs no credentials
  and there is no live broker to use them. The file exists so that adding keys
  later is a file edit, not a unit-file change made under pressure.
- **The startup banner says a 25% budget can drift to 30%.** Correct, and
  deliberate. `strategy_max_allocation` caps the position at *trade* time; price
  movement between rebalances carries it further, and the rebalance dead-band
  suppresses the trim until it is 5 percentage points out. No order creates that
  exposure, so the risk gate does not reject it — but any *buy* is still blocked
  while the position is over the cap.
- **The bot exits 2 and systemd reports the unit as inactive, not failed.**
  Correct. A halt is a decision, not a crash.
