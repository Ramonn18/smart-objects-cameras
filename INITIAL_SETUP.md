# OAK-D + Raspberry Pi 5 - Initial Setup Guide

**For Instructors Only**

This guide covers the initial setup process for configuring the Raspberry Pi 5 systems from scratch. Students will receive pre-configured Pis and should refer to [README.md](README.md) instead.

---

## Overview

This document walks through setting up two Raspberry Pi 5 configurations:

| Device  | RAM  | Configuration | Hostname      | Access Method                |
| ------- | ---- | ------------- | ------------- | ---------------------------- |
| Pi 5 #1 | 16GB | Desktop + VNC | smartobjects1 | SSH (key-based) + VNC Viewer |
| Pi 5 #2 | 8GB  | Headless      | smartobjects2 | SSH (key-based) / VS Code    |

---

## Prerequisites

### Hardware

- Raspberry Pi 5 (8GB or 16GB)
- MicroSD card (32GB+ recommended)
- USB-C power supply (27W official Pi 5 supply recommended)
- OAK-D camera (USB)
- Ethernet cable (recommended) or WiFi access
- **Recommended:** Powered USB hub (for camera stability)

### Software (on your computer)

- [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
- [RealVNC Viewer](https://www.realvnc.com/en/connect/download/viewer/) (for VNC setup)
- SSH client (built into Mac/Linux, use PuTTY or Windows Terminal on Windows)

---

## Part 1: Flash the Operating System

### Prepare Your SSH Key (First-Time Setup)

**Before imaging**, you'll need an SSH key pair on your computer for passwordless authentication.

**Important Notes:**
- ✅ **You can use the SAME key for BOTH Raspberry Pis** (smartobjects1 and smartobjects2)
- ✅ We'll use a project-specific key name to avoid conflicts with existing keys

**On Mac/Linux:**
```bash
# Check if you already have an SSH key
ls ~/.ssh/id_ed25519_smartobjects.pub

# If not, generate one with a project-specific name
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_smartobjects -C "your_email@example.com"
# Press Enter when asked for passphrase (or set one for extra security)

# Display your public key (you'll need this in the next step)
cat ~/.ssh/id_ed25519_smartobjects.pub
```

**On Windows (PowerShell):**
```powershell
# Check for existing key
ls $env:USERPROFILE\.ssh\id_ed25519_smartobjects.pub

# Generate if needed
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\id_ed25519_smartobjects -C "your_email@example.com"
# Press Enter when asked for passphrase (or set one for extra security)

# Display your public key
type $env:USERPROFILE\.ssh\id_ed25519_smartobjects.pub
```

**Copy the entire output** starting with `ssh-ed25519 ...` — you'll paste this into Raspberry Pi Imager for **BOTH** Pis.

**What is this key doing?**
- **Private key** (`id_ed25519_smartobjects`) - Stays on YOUR computer (never share!)
- **Public key** (`id_ed25519_smartobjects.pub`) - Goes on the Pis (safe to share)
- Same key pair works for both smartobjects1 and smartobjects2

---

### For the 16GB Pi (Desktop + VNC)

1. Open Raspberry Pi Imager
2. Click **Choose Device** → Raspberry Pi 5
3. Click **Choose OS** → Raspberry Pi OS (64-bit) _(the full desktop version)_
4. Click **Choose Storage** → Select your SD card
5. Click **Next**, then **Edit Settings**

#### General Tab:

```
☑ Set hostname: smartobjects1
☑ Set username and password
    Username: [your username, e.g., carrie]
    Password: [choose a password - optional if using SSH keys]
☑ Configure wireless LAN
    SSID: [your home network]
    Password: [your WiFi password]
    Country: [your country code, e.g., US]
☑ Set locale settings
    Time zone: [your timezone, e.g., America/New_York]
    Keyboard layout: us
```

**💡 Tip:** You can add multiple WiFi networks later (see [README.md - WiFi Management](README.md#wifi-network-management)).

#### Services Tab:

```
☑ Enable SSH
  ● Allow public-key authentication only (RECOMMENDED for headless Pi)

  Paste your SSH public key (from preparation step above):
  [Paste the entire ssh-ed25519 ... key here]
```

**For the desktop Pi (smartobjects1)**, you can use either:
- **Public-key authentication only** (more secure)
- **Use password authentication** (easier for beginners with VNC access)

6. Click **Save**, then **Yes** to apply settings
7. Click **Yes** to confirm and write the image

---

### For the 8GB Pi (Headless)

**IMPORTANT:** For the headless Pi, you **MUST** use SSH key authentication since there's no display/VNC to fall back on.

Repeat the above steps with these changes:

- **Choose OS** → Raspberry Pi OS Lite (64-bit) _(no desktop)_
- **Set hostname:** `smartobjects2`
- **Services Tab:** Select **"Allow public-key authentication only"** and paste **the SAME public key** you used for smartobjects1

**💡 Pro tip:** You're using the same key for both Pis - this means one key unlocks both systems!

---

### Configure SSH to Use Your Custom Key (Recommended)

Since we used a custom key name (`id_ed25519_smartobjects`), we need to tell SSH to use it automatically.

**On Mac/Linux:**
```bash
# Edit (or create) your SSH config
nano ~/.ssh/config

# Add these entries:
Host smartobjects1 smartobjects1.local
    HostName smartobjects1.local
    User [your-username]
    IdentityFile ~/.ssh/id_ed25519_smartobjects

Host smartobjects2 smartobjects2.local
    HostName smartobjects2.local
    User [your-username]
    IdentityFile ~/.ssh/id_ed25519_smartobjects

# Save: Ctrl+O, Enter, Ctrl+X
# Secure the file
chmod 600 ~/.ssh/config
```

**On Windows (PowerShell):**
```powershell
# Edit SSH config
notepad $env:USERPROFILE\.ssh\config

# Add the same entries as above, then save
```

**Replace `[your-username]`** with the username you set during OS imaging (e.g., `carrie`).

**Now you can simply connect with:**
```bash
ssh smartobjects1.local  # No need to specify key!
ssh smartobjects2.local
```

---

## Part 2: Initial Boot & Network Setup

1. Insert the SD card into the Pi
2. Connect Ethernet (recommended) or ensure WiFi credentials are correct
3. Connect power — wait 2-3 minutes for first boot
4. Find your Pi's IP address:
   - Check your router's admin page, or
   - Use `ping smartobjects1.local` (or `smartobjects2.local`)

### First SSH Connection

```bash
# For Desktop Pi (16GB)
ssh smartobjects1.local

# For Headless Pi (8GB)
ssh smartobjects2.local
```

**If you configured SSH keys:** You'll connect automatically without entering a password!

**If you're using password authentication:** Enter your password when prompted.

On first connection, you'll see a fingerprint verification prompt:
```
The authenticity of host 'smartobjects1.local' can't be established.
ED25519 key fingerprint is SHA256:...
Are you sure you want to continue connecting (yes/no)?
```
Type `yes` and press Enter.

---

## Part 3: System Updates (Both Pis)

Run these commands on both Pis:

```bash
# Update package lists and upgrade
sudo apt update && sudo apt full-upgrade -y

# Install essential dependencies
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    libopencv-dev \
    git \
    htop

# Reboot to apply any kernel updates
sudo reboot
```

---

## Part 4: VNC Setup (16GB Pi Only)

After reboot, SSH back into the desktop Pi:

```bash
ssh smartobjects1.local
```

### Enable VNC

```bash
sudo raspi-config
```

Navigate to:

1. **Interface Options** → **VNC** → **Yes** (Enable)
2. **Display Options** → **VNC Resolution** → **1920x1080** (or your preference)
3. Select **Finish** and reboot when prompted

### Connect via VNC

1. Open RealVNC Viewer on your computer
2. Enter: `smartobjects1.local` (or the IP address)
3. Enter your username and password
4. You should see the Pi desktop

### Optional: Performance Tweaks for VNC

```bash
# Disable screen blanking (prevents black screen over VNC)
sudo raspi-config
# Display Options → Screen Blanker → No

# For smoother VNC, reduce desktop effects
# Right-click desktop → Desktop Preferences → Defaults tab →
# Set to "No image" for less overhead
```

---

## Part 5: DepthAI Installation (Both Pis)

### Understanding User Accounts

Before we proceed, note that:
- **Your username** - The account you created during OS installation (e.g., `carrie`)
- **`root`** - The superuser/administrator (accessed via `sudo`)
- **`pi`** - Old default username (no longer used in modern Raspberry Pi OS)

In this guide, we'll use `[username]` to represent your actual username.

---

### Installation Method: Shared Virtual Environment (Recommended)

For classroom use with multiple students, we'll create **one shared virtual environment** that all users can access. This saves disk space and ensures everyone uses the same package versions.

#### Step 1: Create Shared Virtual Environment

```bash
# Create a system-wide directory for the shared environment
sudo mkdir -p /opt/oak-shared
sudo chown $USER:$USER /opt/oak-shared
cd /opt/oak-shared

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### Step 2: Install DepthAI 3.x and Dependencies

```bash
# Install DepthAI 3.x and depthai-nodes with piwheels for faster ARM builds
pip install --extra-index-url https://www.piwheels.org/simple/ \
    depthai \
    depthai-nodes \
    opencv-python \
    numpy \
    requests \
    aiohttp \
    python-dotenv

# Verify installation
python3 -c "import depthai as dai; print(f'DepthAI {dai.__version__} installed successfully!')"
python3 -c "from depthai_nodes.node import ParsingNeuralNetwork; print('depthai-nodes installed successfully!')"
```

**Note:** We're using `depthai` version 3.x with `depthai-nodes` for high-level neural network parsing. Models are downloaded automatically from the Luxonis Hub (models.luxonis.com).

#### Step 3: Make It Accessible to All Users

```bash
# Make the venv readable by all users
sudo chmod -R 755 /opt/oak-shared

# Deactivate for now
deactivate
```

#### Step 4: Create Project Directory

Each user will have their own project directory, but share the virtual environment:

```bash
# Create your personal project directory
mkdir -p ~/oak-projects
cd ~/oak-projects

# Test activating the shared venv
source /opt/oak-shared/venv/bin/activate

# Verify it works
python3 -c "import depthai as dai; print(f'DepthAI {dai.__version__} ready!')"
```

**For convenience**, create an alias to activate the venv:

```bash
# Add to your shell profile
echo "alias activate-oak='source /opt/oak-shared/venv/bin/activate'" >> ~/.bashrc
source ~/.bashrc

# Now you can just type:
activate-oak
```

---

### Set Up udev Rules (USB Permissions)

```bash
# Create udev rule for OAK cameras
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
    sudo tee /etc/udev/rules.d/80-movidius.rules

# Reload rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Test Camera Connection

Connect your OAK-D camera via USB, then:

```bash
# Activate shared virtual environment
activate-oak

# Quick test - check if camera is detected
python3 -c "import depthai as dai; devices = dai.Device.getAllAvailableDevices(); print(f'Found {len(devices)} camera(s)')"

# More detailed test with device info
python3 -c "import depthai as dai; device = dai.Device(); print(f'Camera connected: {device.getDeviceId()}')"
```

You should see output showing your camera is detected. Expected output:
```
Found 1 camera(s)
Camera connected: 14442C108124F4D000
```

---

### Camera Firmware Notes

**USB OAK-D cameras** (OAK-D, OAK-D-Lite, OAK-1) boot from the host computer and don't require firmware updates. They automatically use the bootloader embedded in the depthai library.

**PoE cameras** (OAK-D-POE) have onboard flash and can be updated if needed.

You can check the bootloader version:

```bash
python3 -c "import depthai as dai; device = dai.Device(); print(f'Bootloader: {device.getBootloaderVersion()}')"
```

- **USB cameras**: Will typically show `None` - this is normal and expected
- **PoE cameras**: Will show actual bootloader version

**For most USB setups, you can skip firmware updates entirely.** The depthai library manages bootloader compatibility automatically.

---

## Part 6: Person Detection Script

See [README.md - Person Detection](README.md#running-the-person-detector) for the detection script code and usage instructions.

The script should be placed at `~/oak-projects/person_detector.py` on the Pi.

---

## Part 7: Auto-Start on Boot (Optional)

To run the person detector automatically when the Pi boots:

### Create systemd Service

```bash
sudo nano /etc/systemd/system/person-detector.service
```

Paste (replace `carrie` with your actual username):

```ini
[Unit]
Description=OAK-D Person Detector
After=network.target

[Service]
Type=simple
User=carrie
WorkingDirectory=/home/carrie/oak-projects
ExecStart=/opt/oak-shared/venv/bin/python3 /home/carrie/oak-projects/person_detector.py --log
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Note:** Replace `carrie` with your actual username throughout the file.

### Enable the Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start
sudo systemctl enable person-detector

# Start now
sudo systemctl start person-detector

# Check status
sudo systemctl status person-detector

# View logs
journalctl -u person-detector -f
```

### Disable Auto-Start (if needed)

```bash
sudo systemctl disable person-detector
sudo systemctl stop person-detector
```

---

## WiFi Network Configuration

For instructions on configuring multiple WiFi networks (home + classroom), see [README.md - WiFi Network Management](README.md#wifi-network-management).

---

## Next Steps

Once the Pis are configured, students can follow [README.md](README.md) for:
- Connecting to the Pis (SSH, VNC, VS Code)
- Running the person detector
- Multi-user access
- Discord notifications
- Troubleshooting

---

## Reference

- Main student guide: [README.md](README.md)
- Quick command reference: [CHEATSHEET.md](CHEATSHEET.md)
- For AI assistant context: [CLAUDE.md](CLAUDE.md)
