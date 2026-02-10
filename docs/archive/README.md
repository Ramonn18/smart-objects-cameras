# Archive - Setup Documentation

This folder contains setup documentation that has **already been completed** on all three Raspberry Pis (Orbit, Gravity, Horizon).

Students don't need to run these steps - they're archived here for instructor reference only.

---

## What's Already Configured

### 1. Shared Model Cache
- **Location**: `/opt/depthai-cache`
- **Purpose**: DepthAI models download once and are accessible to all users
- **Benefit**: Prevents permission errors when multiple students test their own script copies

### 2. Shared Oak Examples
- **Location**: `/opt/oak-shared/oak-examples/`
- **Purpose**: Luxonis example code (neural networks, depth, tutorials, etc.)
- **Access**: Symlinked to `~/oak-examples/` in each user's home directory

### 3. User Accounts & Groups
- All student accounts created and configured
- Added to necessary groups (video, gpio, i2c, spi, users)
- SSH keys configured for passwordless access

---

## Files in This Archive

### multi-user-setup.md
**Original**: `MULTI_USER_SETUP.md` (root of repository)

Complete instructions for:
- Sharing oak-examples with all users
- Adding users to sudoers
- Creating user accounts
- Setting up shared directory structure
- Testing multi-user setup

**Status**: ✅ Completed on all three Pis

---

### setup_shared_model_cache.sh
**Purpose**: Bash script that creates `/opt/depthai-cache` directory and configures `DEPTHAI_ZOO_CACHE` environment variable system-wide.

**What it does**:
1. Creates `/opt/depthai-cache` with 777 permissions
2. Creates `/etc/profile.d/depthai.sh` to set `DEPTHAI_ZOO_CACHE` for all users
3. Reloads environment

**Status**: ✅ Completed on all three Pis

---

### test_multi_user_setup.sh
**Purpose**: Verification script that checks if multi-user setup is correct.

**Tests performed**:
- Shared directories exist
- Permissions are correct
- Environment variables are set
- Symlinks work
- Model cache is accessible

**Status**: ✅ All tests passing on all three Pis

---

## When to Use This Archive

- **Instructors setting up new Pis**: Follow `multi-user-setup.md`
- **Troubleshooting permission issues**: Reference `multi-user-setup.md` to verify configuration
- **Understanding the setup**: Review these files to see how shared resources were configured

---

## Students: You Don't Need This!

If you're a student, you don't need to run any of these scripts. Everything is already set up and working.

Instead, see the main student documentation:
- [README.md](../../README.md) - Main guide for using the cameras
- [docs/multi-user-access.md](../multi-user-access.md) - Best practices for collaborative work

---

## Related Documentation

- [../multi-user-access.md](../multi-user-access.md) - Student-facing collaborative workflow guide
- [../../WORKING_VERSIONS.md](../../WORKING_VERSIONS.md) - Package versions and compatibility notes
- [../../INITIAL_SETUP.md](../../INITIAL_SETUP.md) - Complete Pi setup from scratch (instructors)
