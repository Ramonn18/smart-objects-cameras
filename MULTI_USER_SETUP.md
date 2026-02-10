# Multi-User Setup for Raspberry Pi Smart Objects

Instructions for making shared resources (oak-examples, sudoers) available to multiple users on each Pi. Tested on orbit, to be repeated on horizon and gravity.

## Current State (orbit)

- **14 user accounts**: carrie + 13 students
- **`users` group**: All students are already members
- **`video` + `gpio` groups**: All students already added (needed for camera + GPIO access)
- **Shared venv**: `/opt/oak-shared/venv/` — already shared
- **Shared depthai**: `/opt/oak-shared/depthai/` — already `root:users` with `775`
- **oak-examples**: Currently at `/home/carrie/oak-examples/` — owned by `carrie:carrie`, needs sharing

## Step 1: Share oak-examples with All Users

Move oak-examples to the shared location (`/opt/oak-shared/`) and set group ownership to `users`, matching the existing `depthai` directory pattern.

```bash
# Run as carrie (who has sudo) on the target Pi

# Move oak-examples to shared location
sudo mv /home/carrie/oak-examples /opt/oak-shared/oak-examples

# Set ownership to root:users (matches /opt/oak-shared/depthai pattern)
sudo chown -R root:users /opt/oak-shared/oak-examples

# Set permissions: owner+group can read/write, others can read
sudo chmod -R 775 /opt/oak-shared/oak-examples

# Ensure new files/dirs inherit the group (setgid)
sudo find /opt/oak-shared/oak-examples -type d -exec chmod g+s {} \;
```

### Create Convenience Symlinks for All Users

```bash
# Create symlink in each user's home directory
for user in $(ls /home/); do
    sudo ln -sf /opt/oak-shared/oak-examples /home/$user/oak-examples
    echo "Created symlink for $user"
done
```

After this, every user can access examples via `~/oak-examples/`.

### Verify

```bash
# Check ownership and permissions
ls -la /opt/oak-shared/oak-examples/

# Test as another user (e.g., seren)
sudo -u seren ls /opt/oak-shared/oak-examples/
sudo -u seren cat /opt/oak-shared/oak-examples/README.md

# Test symlink works
sudo -u seren ls ~/oak-examples/
```

## Step 2: Add Users to Sudoers

Add specific students as sudoers so they can manage services, install packages, etc.

### Option A: Add to sudo group (full sudo access)

```bash
# Replace USERNAME with actual student usernames
sudo usermod -aG sudo USERNAME1
sudo usermod -aG sudo USERNAME2
sudo usermod -aG sudo USERNAME3
sudo usermod -aG sudo USERNAME4
```

### Option B: Limited sudo via sudoers.d (safer for classroom)

Create a file that grants sudo for specific commands only:

```bash
# Create a sudoers drop-in file
sudo visudo -f /etc/sudoers.d/smart-objects-students

# Add these lines (replace usernames):
# USERNAME1 ALL=(ALL) NOPASSWD: /bin/systemctl, /usr/bin/apt, /usr/bin/pip3
# USERNAME2 ALL=(ALL) NOPASSWD: /bin/systemctl, /usr/bin/apt, /usr/bin/pip3
# USERNAME3 ALL=(ALL) NOPASSWD: /bin/systemctl, /usr/bin/apt, /usr/bin/pip3
# USERNAME4 ALL=(ALL) NOPASSWD: /bin/systemctl, /usr/bin/apt, /usr/bin/pip3
```

### Option C: Full sudo with password (recommended for trusted students)

```bash
sudo visudo -f /etc/sudoers.d/smart-objects-students
```

Add:
```
USERNAME1 ALL=(ALL:ALL) ALL
USERNAME2 ALL=(ALL:ALL) ALL
USERNAME3 ALL=(ALL:ALL) ALL
USERNAME4 ALL=(ALL:ALL) ALL
```

### Verify Sudoers

```bash
# Test that the user has sudo access
sudo -u USERNAME1 sudo -l
```

## Step 3: Repeat on horizon and gravity

Run the same steps on each Pi. The setup assumes:

1. User accounts already exist (created during initial setup)
2. Users are already in `users`, `video`, `gpio` groups
3. `/opt/oak-shared/venv/` exists with the shared Python environment
4. `oak-examples` is cloned somewhere on the Pi

### If oak-examples hasn't been cloned yet

```bash
# Clone directly to shared location
sudo git clone https://github.com/luxonis/oak-examples.git /opt/oak-shared/oak-examples
sudo chown -R root:users /opt/oak-shared/oak-examples
sudo chmod -R 775 /opt/oak-shared/oak-examples
sudo find /opt/oak-shared/oak-examples -type d -exec chmod g+s {} \;

# Create symlinks
for user in $(ls /home/); do
    sudo ln -sf /opt/oak-shared/oak-examples /home/$user/oak-examples
done
```

### If user accounts need to be created

```bash
# Create user
sudo adduser STUDENTNAME

# Add to required groups
sudo usermod -aG users,video,gpio,i2c,spi STUDENTNAME

# Create symlinks to shared resources
sudo ln -sf /opt/oak-shared/oak-examples /home/STUDENTNAME/oak-examples
```

## Quick Reference: All Three Pis

| Hostname | Role | RAM | Access |
|----------|------|-----|--------|
| orbit | Desktop (VNC) | 16GB | `ssh orbit.local` |
| horizon | TBD | TBD | `ssh horizon.local` |
| gravity | TBD | TBD | `ssh gravity.local` |

## Shared Directory Structure (after setup)

```
/opt/oak-shared/
├── venv/                    # Shared Python venv (all users)
├── depthai/                 # Shared depthai source (root:users, 775)
├── depthai_models/          # Shared model YAML configs (root:users, 775, setgid)
│   ├── yunet.RVC2.yaml              # Face detection (640x480, for fatigue)
│   ├── yunet_gaze.RVC2.yaml         # Face detection (320x240, for gaze)
│   ├── mediapipe_face_landmarker.RVC2.yaml  # Face landmarks (for fatigue)
│   ├── head_pose_estimation.RVC2.yaml       # Head pose (for gaze)
│   └── gaze_estimation_adas.RVC2.yaml       # Gaze estimation (for gaze)
└── oak-examples/            # Shared Luxonis examples (root:users, 775)
    ├── neural-networks/     # NN examples including fatigue detection
    ├── apps/
    ├── camera-controls/
    ├── depth-measurement/
    ├── tutorials/
    └── ...

/home/USERNAME/
├── oak-examples -> /opt/oak-shared/oak-examples  # Symlink
└── oak-projects/            # Personal project directory
    ├── person_detector.py
    ├── fatigue_detector.py
    ├── gaze_detector.py
    ├── discord_dm_notifier.py
    ├── depthai_models -> /opt/oak-shared/depthai_models  # Symlink
    ├── utils/               # Shared utility modules
    │   ├── face_landmarks.py         # EAR + head pose (fatigue)
    │   ├── process_keypoints.py      # Eye/face crop configs (gaze)
    │   ├── node_creators.py          # Pipeline crop node helper (gaze)
    │   ├── config_sender_script.py   # On-device script (gaze)
    │   └── host_concatenate_head_pose.py  # Head pose concat (gaze)
    ├── .env                 # Personal Discord tokens
    └── ...
```

## Test Script

Run `test_multi_user_setup.sh` on each Pi to verify the setup is correct:

```bash
bash test_multi_user_setup.sh
```
