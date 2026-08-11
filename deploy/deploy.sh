#!/usr/bin/env bash
#
# Deploy Trading-bot to an Ubuntu VM. PAPER MODE ONLY.
#
# Run this ON THE VM, as a user with sudo. It is idempotent: running it twice
# changes nothing the second time. It refuses to deploy anything that is not
# reproducible from the git remote — a dirty tree or an unpushed HEAD means the
# code on the VM could never be recovered from the repository.
#
#   sudo ./deploy/deploy.sh
#
# What it will NOT do: enable live trading. LIVE_TRADING is never written, and
# no --live flag is ever passed. Stage 6b (KrakenBroker) does not exist, so
# there is nothing live to enable.

set -euo pipefail

APP_USER=tradingbot
APP_GROUP=tradingbot
APP_DIR=/opt/trading-bot
STATE_DIR=/var/lib/trading-bot
LOG_DIR=/var/log/trading-bot
SECRET_DIR=/etc/trading-bot
SECRET_FILE="${SECRET_DIR}/env"
PYTHON_VERSION=3.11

die() { printf '\nDEPLOY ABORTED: %s\n' "$*" >&2; exit 1; }
step() { printf '\n=== %s\n' "$*"; }

[[ ${EUID} -eq 0 ]] || die "run with sudo (needs to create a system user and units)"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

for tool in git rsync install systemctl; do
    command -v "${tool}" >/dev/null 2>&1 \
        || die "${tool} is not installed. sudo apt-get install -y ${tool}"
done

# --- 1. refuse to deploy anything unreproducible -----------------------------

step "Verifying the tree is deployable"

git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "${REPO_ROOT} is not a git repository"

if [[ -n "$(git status --porcelain)" ]]; then
    git status --short >&2
    die "working tree is dirty. Commit or stash first — a deploy you cannot
     reproduce from the remote is a deploy you cannot roll back."
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
HEAD_SHA="$(git rev-parse HEAD)"

git fetch --quiet origin "${BRANCH}" 2>/dev/null \
    || die "cannot reach origin to verify HEAD is pushed"

if ! git merge-base --is-ancestor "${HEAD_SHA}" "origin/${BRANCH}" 2>/dev/null; then
    die "HEAD (${HEAD_SHA:0:8}) is not on origin/${BRANCH}. Push first, so the
     deployed commit can be fetched again later."
fi

echo "  branch ${BRANCH} @ ${HEAD_SHA:0:8} (pushed)"

# --- 2. the clock ------------------------------------------------------------
#
# Candle timestamps, cycle scheduling, the freshness check and (once Stage 6b
# lands) the API nonce all depend on the clock being right. A skewed clock makes
# the bot reject good data as stale, or accept stale data as good.

step "Verifying time synchronisation"

if ! command -v timedatectl >/dev/null 2>&1; then
    die "timedatectl not found — cannot verify the clock. Install systemd-timesyncd
     or chrony and re-run."
fi

SYNCED="$(timedatectl show -p NTPSynchronized --value 2>/dev/null || echo no)"
if [[ "${SYNCED}" != "yes" ]]; then
    timedatectl status >&2 || true
    die "the system clock is NOT synchronised (NTPSynchronized=${SYNCED}).
     Fix it before deploying:
       sudo timedatectl set-ntp true
       systemctl status systemd-timesyncd    # or: systemctl status chronyd
     Then re-run this script."
fi
echo "  clock synchronised ($(timedatectl show -p Timezone --value))"

# --- 3. user, directories, permissions ---------------------------------------

step "Creating the service user and directories"

if ! id -u "${APP_USER}" >/dev/null 2>&1; then
    useradd --system --home-dir "${APP_DIR}" --shell /usr/sbin/nologin "${APP_USER}"
    echo "  created system user ${APP_USER}"
else
    echo "  user ${APP_USER} already exists"
fi

install -d -o "${APP_USER}" -g "${APP_GROUP}" -m 0750 "${APP_DIR}"
install -d -o "${APP_USER}" -g "${APP_GROUP}" -m 0750 "${STATE_DIR}"
install -d -o "${APP_USER}" -g "${APP_GROUP}" -m 0750 "${STATE_DIR}/state"
install -d -o "${APP_USER}" -g "${APP_GROUP}" -m 0750 "${STATE_DIR}/parquet"
install -d -o "${APP_USER}" -g "${APP_GROUP}" -m 0750 "${STATE_DIR}/backups"
install -d -o "${APP_USER}" -g "${APP_GROUP}" -m 0750 "${LOG_DIR}"

# Secrets live here and ONLY here: root-owned, 600, outside the repo and outside
# the working directory. systemd reads it as root before dropping privileges,
# so the bot user does not need — and must not have — access to the file.
install -d -o root -g root -m 0700 "${SECRET_DIR}"
if [[ ! -f "${SECRET_FILE}" ]]; then
    cat > "${SECRET_FILE}" <<'SECRETS'
# Trading-bot secrets. Root-owned, mode 600. Read by systemd (as root) via
# EnvironmentFile= before it drops to the tradingbot user.
#
# EMPTY ON PURPOSE. Paper mode needs no credentials, and Stage 6b (KrakenBroker)
# does not exist yet, so there is nothing that could use a key.
#
# Do NOT add LIVE_TRADING here. run.py requires BOTH --live and LIVE_TRADING=yes,
# and there is no live broker to construct; setting it achieves nothing except
# removing one of the two guards that stop an accident later.
SECRETS
    echo "  created ${SECRET_FILE} (empty; paper mode needs no keys)"
else
    echo "  ${SECRET_FILE} already exists — left untouched"
fi
chown root:root "${SECRET_FILE}"
chmod 600 "${SECRET_FILE}"

# --- 4. code -----------------------------------------------------------------

step "Installing the code at ${APP_DIR}"

# Copy the tree, minus the things that must not leave the build host. `data/`
# is market data (gitignored, large); `.venv` is rebuilt on the target.
rsync -a --delete \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    --exclude 'data' \
    --exclude 'results' \
    "${REPO_ROOT}/" "${APP_DIR}/"

printf '%s\n' "${HEAD_SHA}" > "${APP_DIR}/DEPLOYED_COMMIT"
chown -R "${APP_USER}:${APP_GROUP}" "${APP_DIR}"

step "Building the virtualenv"

if ! command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then
    die "python${PYTHON_VERSION} not found. Install it:
       sudo apt-get update && sudo apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv"
fi

sudo -u "${APP_USER}" "python${PYTHON_VERSION}" -m venv "${APP_DIR}/.venv"
sudo -u "${APP_USER}" "${APP_DIR}/.venv/bin/pip" install --quiet --upgrade pip
sudo -u "${APP_USER}" "${APP_DIR}/.venv/bin/pip" install --quiet -e "${APP_DIR}"

# Byte-compile now, while /opt is still writable. The unit runs with
# ProtectSystem=strict, which makes /opt read-only, so the first import under
# systemd could not write __pycache__. Python tolerates that silently, but
# paying the compile cost on every start for no reason is avoidable.
sudo -u "${APP_USER}" "${APP_DIR}/.venv/bin/python" -m compileall -q "${APP_DIR}/src" \
    >/dev/null 2>&1 || true
echo "  virtualenv ready"

# --- 5. verify the installed code actually runs ------------------------------
#
# Before pointing systemd at it. A unit that fails on ExecStart at 03:00 is a
# worse discovery than a deploy that fails now.

step "Smoke-testing the installed entry point"

sudo -u "${APP_USER}" "${APP_DIR}/.venv/bin/python" -m trading_bot.run --help >/dev/null \
    || die "the installed entry point does not run"

sudo -u "${APP_USER}" "${APP_DIR}/.venv/bin/python" -c \
    'import trading_bot, trading_bot.run, trading_bot.logging_setup' \
    || die "the installed package does not import"
echo "  entry point runs"

# --- 6. units ----------------------------------------------------------------

step "Installing systemd units"

for unit in trading-bot.service \
            trading-bot-heartbeat.service trading-bot-heartbeat.timer \
            trading-bot-backup.service trading-bot-backup.timer; do
    install -o root -g root -m 0644 "${REPO_ROOT}/deploy/${unit}" "/etc/systemd/system/${unit}"
    echo "  installed ${unit}"
done

systemd-analyze verify /etc/systemd/system/trading-bot.service \
    || die "systemd rejected the unit file"

systemctl daemon-reload
systemctl enable --now trading-bot-heartbeat.timer
systemctl enable --now trading-bot-backup.timer
systemctl enable trading-bot.service

# --- 7. done -----------------------------------------------------------------

cat <<EOF

=== Deployed ${HEAD_SHA:0:8} to ${APP_DIR} (PAPER MODE)

The bot service is ENABLED but NOT STARTED. Market data is not deployed by this
script — data/ is gitignored, so it has to be put on the box separately:

    sudo -u ${APP_USER} rsync -a <your-machine>:Trading-bot/data/parquet/ ${STATE_DIR}/parquet/

Then, in order:

    sudo -u ${APP_USER} ${APP_DIR}/.venv/bin/python -m trading_bot.run \\
        --reconcile-only --config ${APP_DIR}/config.yaml \\
        --data-dir ${STATE_DIR}/parquet --state-dir ${STATE_DIR}/state \\
        --halt-file ${STATE_DIR}/HALT

    sudo -u ${APP_USER} ${APP_DIR}/.venv/bin/python -m trading_bot.run \\
        --once --dry-run --config ${APP_DIR}/config.yaml \\
        --data-dir ${STATE_DIR}/parquet --state-dir ${STATE_DIR}/state \\
        --halt-file ${STATE_DIR}/HALT

    sudo systemctl start trading-bot

Read RUNBOOK.md before the first start. It is at ${APP_DIR}/RUNBOOK.md.
EOF
