#!/bin/bash
# test_multi_user_setup.sh
# Verifies multi-user setup is correct on a Raspberry Pi
# Run as: bash test_multi_user_setup.sh

PASS=0
FAIL=0
WARN=0

pass() { echo "  PASS: $1"; ((PASS++)); }
fail() { echo "  FAIL: $1"; ((FAIL++)); }
warn() { echo "  WARN: $1"; ((WARN++)); }

echo "========================================="
echo "Multi-User Setup Test — $(hostname)"
echo "========================================="
echo ""

# --- Shared directories ---
echo "[1] Shared directories"

if [ -d "/opt/oak-shared/venv" ]; then
    pass "/opt/oak-shared/venv/ exists"
else
    fail "/opt/oak-shared/venv/ not found"
fi

if [ -d "/opt/oak-shared/oak-examples" ]; then
    pass "/opt/oak-shared/oak-examples/ exists"

    # Check group ownership
    GROUP=$(stat -c '%G' /opt/oak-shared/oak-examples)
    if [ "$GROUP" = "users" ]; then
        pass "oak-examples group is 'users'"
    else
        fail "oak-examples group is '$GROUP' (expected 'users')"
    fi

    # Check group write permission
    PERMS=$(stat -c '%a' /opt/oak-shared/oak-examples)
    if [[ "$PERMS" == *"7"* ]]; then
        pass "oak-examples has group write permission ($PERMS)"
    else
        fail "oak-examples missing group write ($PERMS)"
    fi

    # Check setgid on directories
    SETGID=$(find /opt/oak-shared/oak-examples -maxdepth 0 -perm -g+s 2>/dev/null)
    if [ -n "$SETGID" ]; then
        pass "oak-examples has setgid bit set"
    else
        warn "oak-examples missing setgid — new files may not inherit group"
    fi
else
    fail "/opt/oak-shared/oak-examples/ not found"
fi

echo ""

# --- User accounts and groups ---
echo "[2] User accounts and groups"

USER_COUNT=$(ls /home/ | wc -l)
echo "  Found $USER_COUNT user home directories"

USERS_GROUP=$(getent group users | cut -d: -f4)
if [ -n "$USERS_GROUP" ]; then
    MEMBER_COUNT=$(echo "$USERS_GROUP" | tr ',' '\n' | wc -l)
    pass "'users' group has $MEMBER_COUNT members"
else
    fail "'users' group has no members"
fi

VIDEO_GROUP=$(getent group video | cut -d: -f4)
if [ -n "$VIDEO_GROUP" ]; then
    pass "'video' group has members (camera access)"
else
    warn "'video' group has no members"
fi

echo ""

# --- Symlinks ---
echo "[3] Symlinks in home directories"

SYMLINK_COUNT=0
MISSING_COUNT=0
for userdir in /home/*/; do
    username=$(basename "$userdir")
    if sudo test -L "/home/$username/oak-examples"; then
        ((SYMLINK_COUNT++))
    else
        warn "Missing symlink: /home/$username/oak-examples"
        ((MISSING_COUNT++))
    fi
done

if [ $MISSING_COUNT -eq 0 ]; then
    pass "All $SYMLINK_COUNT users have oak-examples symlink"
else
    fail "$MISSING_COUNT users missing oak-examples symlink ($SYMLINK_COUNT have it)"
fi

echo ""

# --- Sudoers ---
echo "[4] Sudoers"

SUDO_GROUP=$(getent group sudo | cut -d: -f4)
if [ -n "$SUDO_GROUP" ]; then
    SUDO_COUNT=$(echo "$SUDO_GROUP" | tr ',' '\n' | wc -l)
    echo "  sudo group members: $SUDO_GROUP ($SUDO_COUNT users)"
else
    echo "  sudo group members: (none besides root)"
fi

if [ -f /etc/sudoers.d/smart-objects-students ]; then
    pass "Custom sudoers file exists at /etc/sudoers.d/smart-objects-students"
else
    warn "No custom sudoers file at /etc/sudoers.d/smart-objects-students"
fi

echo ""

# --- Python environment ---
echo "[5] Python environment"

if [ -f "/opt/oak-shared/venv/bin/python3" ]; then
    pass "Shared venv python3 exists"

    DEPTHAI=$(/opt/oak-shared/venv/bin/pip list 2>/dev/null | grep "^depthai " | awk '{print $2}')
    if [ -n "$DEPTHAI" ]; then
        pass "depthai $DEPTHAI installed"
    else
        fail "depthai not found in shared venv"
    fi

    DEPTHAI_NODES=$(/opt/oak-shared/venv/bin/pip list 2>/dev/null | grep "depthai-nodes" | awk '{print $2}')
    if [ -n "$DEPTHAI_NODES" ]; then
        pass "depthai-nodes $DEPTHAI_NODES installed"
    else
        fail "depthai-nodes not found in shared venv"
    fi
else
    fail "Shared venv python3 not found"
fi

echo ""

# --- Summary ---
echo "========================================="
echo "Results: $PASS passed, $FAIL failed, $WARN warnings"
echo "========================================="

if [ $FAIL -gt 0 ]; then
    echo "Some checks failed. See MULTI_USER_SETUP.md for fix instructions."
    exit 1
else
    echo "Setup looks good!"
    exit 0
fi
