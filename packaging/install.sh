#!/usr/bin/env sh
# Install the SECTOR daemon as a systemd service.
#
#   sudo ./install.sh [--image /path/to/volume.sector]
#
# Run from an unpacked release archive, which contains the `sector` binary, this
# script, and the systemd/ directory beside it.
#
# POSIX sh rather than bash: Raspberry Pi OS Lite has bash, but Alpine and some
# Yocto images ship only busybox ash, and there is nothing here that needs more.
set -eu

BIN_DIR=/usr/local/bin
CONF_DIR=/etc/sector
DATA_DIR=/var/lib/sector
UNIT_DIR=/etc/systemd/system
HERE=$(cd "$(dirname "$0")" && pwd)

IMAGE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --image) IMAGE="${2:-}"; shift 2 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ "$(id -u)" -ne 0 ]; then
    echo "error: must run as root (installs to $BIN_DIR and $UNIT_DIR)" >&2
    exit 1
fi

if [ ! -x "$HERE/sector" ]; then
    echo "error: no sector binary beside this script" >&2
    echo "       run install.sh from an unpacked release archive" >&2
    exit 1
fi

# --- The ABI check, before anything is installed ---
#
# This is the failure the release matrix exists to prevent: a binary built for a
# newer instruction set installs cleanly on an older Pi and then dies with SIGILL
# when a query reaches an unsupported instruction — possibly not until the first
# real request. `sector doctor` compares the running binary's ISA baseline against
# the board and exits 2 when they cannot work together.
#
# Checked here rather than after installation so a wrong artifact is refused
# before it becomes the thing systemd tries to start.
echo "checking this binary against this board"
if "$HERE/sector" doctor; then
    :
else
    status=$?
    if [ "$status" -eq 2 ]; then
        echo >&2
        echo "error: this binary cannot run correctly on this board." >&2
        echo "       install the artifact named above. nothing was installed." >&2
        exit 1
    fi
    # Exit 1 means it runs but a better-matched artifact exists. Worth saying
    # once; not worth refusing over.
    echo
    echo "note: continuing with a working but suboptimal binary (see above)"
fi

echo
echo "installing"
install -d -m 0755 "$BIN_DIR" "$CONF_DIR" "$DATA_DIR"
install -m 0755 "$HERE/sector" "$BIN_DIR/sector"
echo "  $BIN_DIR/sector"

# The config is preserved on reinstall: it holds the operator's paths and worker
# count, and overwriting it would silently revert a tuned deployment.
if [ -f "$CONF_DIR/sector.conf" ]; then
    echo "  $CONF_DIR/sector.conf (kept, already present)"
else
    install -m 0644 "$HERE/systemd/sector.conf" "$CONF_DIR/sector.conf"
    echo "  $CONF_DIR/sector.conf"
fi

install -m 0644 "$HERE/systemd/sector.service" "$UNIT_DIR/sector.service"
install -m 0644 "$HERE/systemd/sector.socket" "$UNIT_DIR/sector.socket"
echo "  $UNIT_DIR/sector.service"
echo "  $UNIT_DIR/sector.socket"

# The group the socket is owned by. DynamicUser gives the daemon its own UID, so
# this group exists only to name who may connect.
if ! getent group sector >/dev/null 2>&1; then
    groupadd --system sector 2>/dev/null || addgroup -S sector 2>/dev/null || true
    echo "  group 'sector' (add your client's user to it to allow queries)"
fi

if [ -n "$IMAGE" ]; then
    if [ ! -f "$IMAGE" ]; then
        echo "error: $IMAGE does not exist" >&2
        exit 1
    fi
    # Verified before installation, not after: a damaged volume should be caught
    # here rather than by the first query. `verify` exits 1 on damage and 2 when
    # it could not check, and both mean do not proceed.
    echo
    echo "verifying $IMAGE"
    if ! "$BIN_DIR/sector" verify --image "$IMAGE"; then
        echo >&2
        echo "error: the volume did not verify. nothing was copied to $DATA_DIR." >&2
        exit 1
    fi
    install -m 0644 "$IMAGE" "$DATA_DIR/volume.sector"
    echo "  $DATA_DIR/volume.sector"
fi

systemctl daemon-reload

echo
echo "installed. next:"
if [ -z "$IMAGE" ]; then
    echo "  1. build a volume on a host:  sector build --input corpus.fvecs --out volume.sector"
    echo "     (add --reserve N to leave room for 'sector append' later)"
    echo "  2. copy it to $DATA_DIR/volume.sector"
    echo "  3. systemctl enable --now sector"
else
    echo "  systemctl enable --now sector"
fi
echo
echo "then:"
echo "  systemctl status sector"
echo "  curl --unix-socket /run/sector/sector.sock http://localhost/info"
echo "  sector selftest        # proves the binary works on this board"
echo
echo "the startup banner reports resident bytes per worker and in total; size"
echo "MemoryMax in the unit against it rather than guessing."