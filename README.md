# OAK-D + Raspberry Pi 5 — Smart Objects Template

This is a **template project** for building Discord bots that communicate with Luxonis OAK-D cameras. Use this as a starting point to create your own smart object systems with computer vision and interactive communication.

**What's Included:**
- 👁️ Real-time person detection using YOLO
- 🤖 Discord bot with commands (!status, !detect, !screenshot)
- 📢 Automatic webhook notifications
- 🎯 Temporal smoothing for stable detection
- 🖼️ Live camera screenshots on demand

**Your instructor has pre-configured the Raspberry Pis** — you can connect and start experimenting immediately! This guide shows you how to use the template, then extend it for your own creative projects.

| Device  | RAM  | Configuration | Hostname      | Access Method                |
| ------- | ---- | ------------- | ------------- | ---------------------------- |
| Pi 5 #1 | 16GB | Desktop + VNC | smartobjects1 | SSH (key-based) + VNC Viewer |
| Pi 5 #2 | 8GB  | Headless      | smartobjects2 | SSH (key-based) / VS Code    |

**Note for instructors:** If you need to set up new Pis from scratch, see [INITIAL_SETUP.md](INITIAL_SETUP.md).

---

## 📚 Table of Contents

### Getting Started

- [Part 1: Connecting to the Pis](#part-1-connecting-to-the-pis)
- [Part 2: Person Detection Script](#part-2-person-detection-script)
- [Part 3: Auto-Start on Boot (Optional)](#part-3-auto-start-on-boot-optional)

### Configuration & Usage

- [Part 4: WiFi Network Management](#part-4-wifi-network-management)
- [Part 5: Multi-User Access](#part-5-multi-user-access)
- [Part 6: Quick Reference](#part-6-quick-reference)

### Troubleshooting

- [Common Issues](#troubleshooting)

### Appendices

- [Appendix A: VS Code Remote Development](#appendix-a-vs-code-remote-development)
- [Appendix B: Discord Notifications](#appendix-b-discord-notifications)

### Additional Resources

- [Next Steps](#next-steps)
- [INITIAL_SETUP.md](INITIAL_SETUP.md) - For instructors: setting up new Pis from scratch
- [CHEATSHEET.md](CHEATSHEET.md) - Quick command reference (print this!)
- [CLAUDE.md](CLAUDE.md) - For AI assistant context only

---

## Part 1: Connecting to the Pis

The Pis are already configured and ready to use! Here's how to connect.

### Prerequisites

**On your computer:**

- SSH client (built into Mac/Linux, use PowerShell on Windows)
- [RealVNC Viewer](https://www.realvnc.com/en/connect/download/viewer/) (optional, for desktop Pi GUI access)
- [VS Code](https://code.visualstudio.com/) with Remote-SSH extension (recommended - see [Appendix A](#appendix-a-vs-code-remote-development))

**Network:**

- Your computer and the Pis must be on the same network
- The Pis should power on and connect to WiFi automatically

### SSH Connection (Terminal Access)

```bash
# For Desktop Pi (16GB)
ssh smartobjects1.local

# For Headless Pi (8GB)
ssh smartobjects2.local
```

**Your instructor has configured SSH key-based authentication**, so you should connect automatically without entering a password!

**First time connecting?** You'll see a fingerprint verification prompt:

```
The authenticity of host 'smartobjects1.local' can't be established.
ED25519 key fingerprint is SHA256:...
Are you sure you want to continue connecting (yes/no)?
```

Type `yes` and press Enter.

**Troubleshooting:** If you get "Host not found" or connection fails:
- Make sure the Pi is powered on and connected to the network
- Try `ping smartobjects1.local` to check if it's reachable
- Ask your instructor for the Pi's IP address and use that instead: `ssh smartobjects1` (replace with actual IP)

### VNC Connection (Desktop Pi Only - Optional)

For graphical desktop access to the 16GB Pi:

1. Open **RealVNC Viewer** on your computer
2. Enter: `smartobjects1.local` (or the IP address)
3. Enter the username and password (ask your instructor)
4. You should see the Pi desktop

**Note:** VNC is only available on smartobjects1 (the desktop Pi). The headless Pi (smartobjects2) has no desktop environment.

### VS Code Remote SSH (Recommended)

The best way to code on the Pi is using VS Code Remote-SSH extension. See **[Appendix A: VS Code Remote Development](#appendix-a-vs-code-remote-development)** for complete setup instructions.

**Quick start:**
1. Install VS Code and the "Remote - SSH" extension
2. **macOS users:** Grant VS Code "Local Network" permission (System Settings → Privacy & Security → Local Network)
3. Connect: `Ctrl+Shift+P` → "Remote-SSH: Connect to Host" → `smartobjects1.local`

---

## Part 2: Person Detection Script

The person detector script is already installed on the Pis. Here's how to use it.

### Understanding the Project Structure

Once you're connected via SSH (or VS Code), navigate to the project directory:

```bash
cd ~/oak-projects
ls -la
```

You should see:
```
/opt/oak-shared/
└── venv/                    # Shared Python virtual environment (all users)

~/oak-projects/              # Your personal project directory
├── person_detector.py       # Main detection script
├── discord_notifier.py      # Discord webhook module (optional)
├── discord_bot.py           # Discord bot for two-way interaction (optional)
├── camera_status.json       # Status file for bot integration (auto-generated)
├── .env                     # Environment variables (webhook URL, bot token)
└── person_detection_*.log   # Detection logs (if --log used)
```

### Activating the Virtual Environment

**Before running any Python scripts**, activate the shared virtual environment:

```bash
activate-oak
```

Your prompt should change to show `(venv)`:
```
(venv) username@smartobjects1:~/oak-projects $
```

### Running the Person Detector

```bash
# Basic detection (console output only)
python3 person_detector.py

# With video display (VNC Pi only - requires desktop)
python3 person_detector.py --display

# With file logging
python3 person_detector.py --log

# Adjust sensitivity (0.0 - 1.0, default 0.5)
python3 person_detector.py --threshold 0.7

# Combine options
python3 person_detector.py --log --threshold 0.6
```

**Stop the script with:** `Ctrl+C`

### Understanding the Detection Script

The script uses the OAK-D camera to detect people in real-time using a neural network that runs directly on the camera's processor.

**Key features:**
- Detects people using YOLO v6 (from Luxonis Hub)
- Runs at ~15 FPS on the OAK-D's onboard processor
- Temporal smoothing (1.5s debounce) to prevent detection flickering
- Only logs when detection status changes (person detected ↔ no person)
- Optional video display with bounding boxes
- Optional logging to timestamped files
- Adjustable confidence threshold
- Discord notifications (webhooks and bot support)

**To view the full script:**
```bash
cat ~/oak-projects/person_detector.py
# Or open in VS Code for syntax highlighting
```

**To modify the script:**
- Make your own copy first: `cp person_detector.py person_detector_yourname.py`
- Edit with VS Code (recommended) or nano: `nano person_detector_yourname.py`
- Run your version: `python3 person_detector_yourname.py`

### Viewing Detection Logs

If you ran with `--log`, check the logs:

```bash
# List all log files
ls -lh ~/oak-projects/*.log

# View the most recent log
tail -f ~/oak-projects/person_detection_*.log

# View specific log with line numbers
cat -n ~/oak-projects/person_detection_20260201_143052.log
```

---

## Part 3: Auto-Start on Boot (Optional)

**Check if it's already running:**

Your instructor may have already configured the person detector to auto-start. Check the status:

```bash
sudo systemctl status person-detector
```

If you see **"Active: active (running)"**, the detector is already running in the background!

**To view live detection logs:**

```bash
journalctl -u person-detector -f
```

Press `Ctrl+C` to stop viewing logs (the detector keeps running).

### Controlling the Auto-Start Service

```bash
# Start the service
sudo systemctl start person-detector

# Stop the service
sudo systemctl stop person-detector

# Restart the service
sudo systemctl restart person-detector

# Check status
sudo systemctl status person-detector

# Enable auto-start on boot
sudo systemctl enable person-detector

# Disable auto-start on boot
sudo systemctl disable person-detector
```

### Configuring Auto-Start (Advanced)

If you want to modify what runs on boot, edit the service file:

```bash
sudo nano /etc/systemd/system/person-detector.service
```

After making changes, reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart person-detector
```

See [INITIAL_SETUP.md](INITIAL_SETUP.md) for full service configuration details.

---

## Part 4: WiFi Network Management

### Switching Between Networks (Home ↔ Classroom)

Modern Raspberry Pi OS uses **NetworkManager** for WiFi configuration. Here are the best methods:

#### Method 1: Using nmtui (Recommended - Menu Interface)

**Perfect for adding classroom WiFi before you get to class!**

`nmtui` is a text-based menu interface that lets you add networks you're not currently connected to:

```bash
# SSH into the Pi
ssh smartobjects1.local

# Open NetworkManager Text UI
sudo nmtui
```

**Steps in nmtui:**
1. Select **"Activate a connection"** to see current networks, OR
2. Select **"Edit a connection"** to add new networks
3. To add a new WiFi network:
   - Choose **"Add"**
   - Select **"Wi-Fi"**
   - Enter SSID: `ClassroomWiFi`
   - Enter Password: `classpassword`
   - Tab to **"OK"** and press Enter
   - Tab to **"Back"** and press Enter
4. Select **"Quit"**

**Tip:** You can add as many networks as you want! The Pi will automatically connect to any available network you've saved.

#### Method 2: Using nmcli (Command Line)

Quick commands for adding and switching networks:

```bash
# List available WiFi networks
nmcli device wifi list

# Connect to a network (saves automatically - but requires network to be in range!)
sudo nmcli device wifi connect "ClassroomWiFi" password "classpassword"

# List saved connections
nmcli connection show

# Switch to a previously saved network
nmcli connection up "ClassroomWiFi"

# Delete a saved network
sudo nmcli connection delete "OldNetwork"
```

**Note:** `nmcli device wifi connect` only works if the network is currently in range. To add a network you're not near, use `nmtui` instead!

#### Method 3: Using raspi-config (Graphical Alternative)

If you prefer a menu interface:

```bash
sudo raspi-config
# Navigate: System Options → Wireless LAN
# Enter new SSID and password
# Reboot
```

#### Method 4: Check Current WiFi Settings

```bash
# See which network you're connected to
nmcli device wifi

# See detailed connection info
nmcli connection show --active

# See WiFi password for saved network
sudo nmcli connection show "YourNetworkName" | grep psk
```

#### Emergency: Editing via SD Card (If Locked Out)

If you can't connect to the Pi at all:

1. Power off the Pi and remove SD card
2. Insert SD card into your computer
3. In the `boot` partition, create a file called `wpa_supplicant.conf`:

   ```
   country=US
   ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
   update_config=1

   network={
       ssid="ClassroomWiFi"
       psk="classpassword"
   }
   ```

4. Save, eject SD card, and boot the Pi
5. Once connected, the Pi will import this to NetworkManager

---

## Part 5: Multi-User Access

Multiple students can access the same Pi simultaneously for collaborative work.

### Best Practices for Shared Access

1. **Use SSH keys for each person:**

   ```bash
   # Each student adds their SSH key to the Pi
   # If you set up SSH config, this works automatically:
   ssh-copy-id smartobjects1.local

   # Or specify your key explicitly:
   ssh-copy-id -i ~/.ssh/id_ed25519_smartobjects.pub smartobjects1.local
   ```

2. **Create personal script copies:**

   ```bash
   # Avoid editing the same file simultaneously
   cd ~/oak-projects
   cp person_detector.py person_detector_alice.py
   cp person_detector.py person_detector_bob.py
   ```

3. **Use VS Code Remote SSH:**
   - Each student can connect with their own VS Code instance
   - VS Code will warn if someone else is editing the same file
   - Coordinate with teammates before editing shared files

4. **Communication is key:**
   - Let teammates know before you restart services
   - Don't stop/restart the camera while someone else is testing
   - Use a shared chat or Discord channel to coordinate

### Adding Additional Users (Optional)

If you want separate user accounts for each student:

```bash
# SSH into the Pi
ssh smartobjects1.local

# Create new user
sudo adduser alice

# Add to required groups for camera access
sudo usermod -aG video,gpio,i2c,spi alice

# Copy project files to their home directory
sudo cp -r /home/pi/oak-projects /home/alice/
sudo chown -R alice:alice /home/alice/oak-projects

# The new user can now log in
ssh alice@smartobjects1.local
```

### Adding Multiple SSH Keys to Shared Account

If multiple people use the same Pi account:

```bash
# Person 1 adds their key (if SSH config is set up)
ssh-copy-id smartobjects1.local
# Or: ssh-copy-id -i ~/.ssh/id_ed25519_smartobjects.pub smartobjects1.local

# Person 2 adds their key (doesn't overwrite Person 1's key)
ssh-copy-id smartobjects1.local
# Or: ssh-copy-id -i ~/.ssh/id_ed25519_smartobjects.pub smartobjects1.local

# Or manually add keys
ssh smartobjects1.local
nano ~/.ssh/authorized_keys
# Paste each person's public key on a new line
```

---

## Part 6: Quick Reference

### Useful Commands

```bash
# Activate shared virtual environment
activate-oak

# Check camera connection (depthai 3.x)
python3 -c "import depthai as dai; devices = dai.Device.getAllAvailableDevices(); print(f'Found {len(devices)} camera(s)')"

# Quick camera test with device info
python3 -c "import depthai as dai; device = dai.Device(); print(f'Device: {device.getMxId()}')"

# Monitor system resources
htop

# Check service status
sudo systemctl status person-detector

# View detection logs
ls -la ~/oak-projects/*.log
tail -f ~/oak-projects/person_detection_*.log
```

### Network Access

| Pi   | Hostname      | SSH Command               | VNC                   |
| ---- | ------------- | ------------------------- | --------------------- |
| 16GB | smartobjects1 | `ssh smartobjects1.local` | `smartobjects1.local` |
| 8GB  | smartobjects2 | `ssh smartobjects2.local` | N/A                   |

**Note:** SSH uses key-based authentication (no password needed if configured).

### File Locations

```
/opt/oak-shared/
└── venv/                    # Shared Python virtual environment (all users)

~/oak-projects/              # Your personal project directory
├── person_detector.py       # Main detection script
├── discord_notifier.py      # Discord webhook module
├── discord_bot.py           # Discord bot for two-way interaction
├── camera_status.json       # Status file for bot integration (auto-generated)
├── .env                     # Environment variables (webhook URL, bot token)
└── person_detection_*.log   # Detection logs (if --log used)
```

---

## Troubleshooting

### Camera Not Found

```bash
# Check USB connection
lsusb | grep Myriad

# Should show something like:
# Bus 001 Device 002: ID 03e7:2485 Intel Movidius MyriadX

# If not found, try:
# 1. Unplug and replug the camera
# 2. Try a different USB port (use USB 3.0 blue ports)
# 3. Use a powered USB hub
# 4. Check udev rules are set up correctly
```

### Update Camera Firmware

If you're experiencing camera issues or want the latest features, update the firmware:

```bash
# Activate venv
activate-oak

# Check current version
python3 -c "import depthai as dai; device = dai.Device(); print(f'Bootloader: {device.getBootloaderVersion()}')"
```

**Note for USB OAK-D cameras:** USB cameras boot from the host and typically show `None` for bootloader version - this is normal and expected. The depthai library manages bootloader compatibility automatically. Firmware updates are typically only needed for PoE cameras. See [INITIAL_SETUP.md](INITIAL_SETUP.md) for detailed firmware information.

### Permission Denied

```bash
# Reapply udev rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Or add user to video group
sudo usermod -aG video $USER
# Then logout and back in
```

### VNC Black Screen

```bash
# Set a resolution in config
sudo raspi-config
# Display Options → VNC Resolution → 1920x1080

# Or edit config.txt directly
sudo nano /boot/firmware/config.txt
# Add:
# hdmi_force_hotplug=1
# hdmi_group=2
# hdmi_mode=82

sudo reboot
```

### Out of Memory (unlikely with your RAM)

```bash
# Check memory usage
free -h

# If needed, increase swap (not usually necessary with 8-16GB)
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Script Crashes on Startup

```bash
# Check service logs
journalctl -u person-detector -n 50

# Test manually first
activate-oak
python3 ~/oak-projects/person_detector.py
```

---

## Appendix A: VS Code Remote Development

VS Code Remote SSH is the **recommended** way to develop on the Raspberry Pi. This gives you a professional development environment on your laptop while the code executes on the Pi.

### What You'll Get

- Full VS Code editor running on your laptop
- Files stored and code executed on the Pi
- Integrated terminal (no separate SSH window needed)
- IntelliSense, syntax highlighting, debugging
- Works great over slow WiFi

---

### Step 1: Install VS Code

Download and install VS Code on your computer:

👉 **https://code.visualstudio.com/download**

Available for Windows, macOS, and Linux.

---

### Step 2: Install the Remote SSH Extension

1. Open VS Code
2. Click the **Extensions** icon in the left sidebar (or press `Ctrl+Shift+X` / `Cmd+Shift+X`)
3. Search for: `Remote - SSH`
4. Click **Install** on "Remote - SSH" by Microsoft

---

### Step 3: Grant Network Permission (macOS Only)

**IMPORTANT for Mac users:** VS Code needs permission to access your local network.

1. Open **System Settings** (Apple menu → System Settings)
2. Go to **Privacy & Security**
3. Scroll down and click **Local Network**
4. Find **Visual Studio Code** in the list
5. **Toggle it ON** ✅

If you don't see Visual Studio Code:
- Try connecting once (it will fail)
- The app will appear in the list
- Enable it and try again

**This is required!** Without this permission, you'll get "No route to host" errors even though terminal SSH works fine.

---

### Step 4: Connect to the Raspberry Pi

#### Using the Command Palette

1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
2. Type: `Remote-SSH: Connect to Host`
3. Click **+ Add New SSH Host**
4. Enter the connection string:
   ```
   ssh smartobjects1.local
   ```
   Or for the headless Pi:
   ```
   ssh smartobjects2.local
   ```
5. Select your SSH config file (usually the first option)
6. Click **Connect** in the popup

#### First Connection

On first connect:

1. Select **Linux** when asked about the platform
2. Enter your password when prompted (if not using SSH keys)
3. Wait while VS Code installs its server component on the Pi (1-2 minutes)

You'll know you're connected when the bottom-left corner shows:

```
>< SSH: smartobjects1.local
```

---

### Step 5: Open the Project Folder

1. Click **File** → **Open Folder** (or `Ctrl+K Ctrl+O`)
2. Navigate to `/home/[username]/oak-projects`
3. Click **OK**
4. Trust the folder when prompted

You should now see the project files in the Explorer sidebar.

---

### Step 6: Set Up the Python Environment

#### Install the Python Extension

1. Go to Extensions (`Ctrl+Shift+X`)
2. Search for: `Python`
3. Install "Python" by Microsoft

#### Select the Virtual Environment

1. Open any `.py` file
2. Look at the bottom status bar — click where it shows the Python version
3. Select: `~/oak-projects/venv/bin/python`

Or use Command Palette:

1. `Ctrl+Shift+P` → `Python: Select Interpreter`
2. Choose: `~/oak-projects/venv/bin/python`

---

### Step 7: Using the Integrated Terminal

Open a terminal inside VS Code:

- Press `` Ctrl+` `` (backtick), or
- **Terminal** → **New Terminal**

The terminal is already connected to the Pi! Activate the virtual environment:

```bash
activate-oak
```

Now you can run scripts:

```bash
python3 person_detector.py
```

---

### Step 8: Running and Debugging Code

#### Quick Run

- Open a Python file
- Click the **▶ Play** button in the top-right corner
- Or press `F5` to run with debugging

#### Run in Terminal

- Right-click in the editor
- Select **Run Python File in Terminal**

#### Debugging

1. Click the **Run and Debug** icon in the sidebar (or `Ctrl+Shift+D`)
2. Click **create a launch.json file**
3. Select **Python File**
4. Set breakpoints by clicking left of line numbers
5. Press `F5` to start debugging

---

### VS Code Tips and Tricks

#### SSH Key Authentication

If you configured SSH keys during OS installation, VS Code will connect automatically without passwords!

**If you need to add your key to the Pi:**

**On Mac/Linux:**

```bash
# Generate key if you don't have one (with project-specific name)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_smartobjects -C "your_email@example.com"

# Copy to Pi (if SSH config is set up from Part 1)
ssh-copy-id smartobjects1.local

# Or specify the key explicitly
ssh-copy-id -i ~/.ssh/id_ed25519_smartobjects.pub smartobjects1.local
```

**On Windows (PowerShell):**

```powershell
# Generate key if needed (with project-specific name)
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\id_ed25519_smartobjects -C "your_email@example.com"

# Copy to Pi manually
type $env:USERPROFILE\.ssh\id_ed25519_smartobjects.pub | ssh smartobjects1.local "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

Now VS Code will connect without asking for a password!

#### Useful Keyboard Shortcuts

| Action          | Windows/Linux  | Mac           |
| --------------- | -------------- | ------------- |
| Command Palette | `Ctrl+Shift+P` | `Cmd+Shift+P` |
| Open Terminal   | `` Ctrl+` ``   | `` Cmd+` ``   |
| Open File       | `Ctrl+P`       | `Cmd+P`       |
| Find in Files   | `Ctrl+Shift+F` | `Cmd+Shift+F` |
| Toggle Sidebar  | `Ctrl+B`       | `Cmd+B`       |
| Run Code        | `F5`           | `F5`          |
| Save            | `Ctrl+S`       | `Cmd+S`       |

#### Recommended Extensions

Install these on the **remote** (they'll install on the Pi):

| Extension      | Purpose                    |
| -------------- | -------------------------- |
| Python         | Python language support    |
| Pylance        | Better Python IntelliSense |
| GitLens        | Git integration            |
| Error Lens     | Inline error highlighting  |
| indent-rainbow | Visualize indentation      |

---

### Working with Multiple Students

#### Avoid File Conflicts

If multiple people connect to the same Pi:

- **Don't edit the same file simultaneously** — VS Code doesn't merge changes
- Create your own branch or subfolder for experiments
- Communicate with your team!

#### Suggested Workflow

```bash
# Create your own working copy
cd ~/oak-projects
cp person_detector.py person_detector_yourname.py

# Edit your copy
code person_detector_yourname.py
```

---

### VS Code Troubleshooting

#### "No route to host" (macOS) - MOST COMMON

**Problem:** VS Code can't access local network even though terminal SSH works.

**Solution:** Grant VS Code Local Network permission:
1. Open **System Settings** → **Privacy & Security** → **Local Network**
2. Find **Visual Studio Code** and toggle it **ON** ✅
3. Restart VS Code completely
4. Try connecting again

This is **required on macOS** - VS Code needs explicit permission to access local IP addresses (192.168.x.x).

#### "Could not establish connection"

1. Make sure the Pi is powered on and booted
2. Check you're on the same network
3. Try pinging: `ping smartobjects1.local`
4. Try using IP address in SSH config instead of `.local` hostname:
   ```
   Host smartobjects1
       HostName 192.168.1.xxx  # Use actual IP
       User carrie
       IdentityFile ~/.ssh/id_ed25519_smartobjects
   ```

#### "Permission denied (publickey,password)"

- Double-check your password
- Make sure SSH is enabled on the Pi:
  ```bash
  sudo raspi-config
  # Interface Options → SSH → Enable
  ```

#### Extensions Not Working

Remote extensions install on the Pi, not your laptop. If an extension isn't working:

1. Open Extensions sidebar
2. Look for the extension
3. Check if it says "Install in SSH: smartobjects1"
4. Click to install it on the remote

#### Terminal Shows Wrong Python

Make sure you:

1. Activated the venv: `activate-oak`
2. Selected the correct interpreter in VS Code (bottom status bar)

#### Slow Performance

- Use **wired Ethernet** if possible
- Close unnecessary VS Code extensions
- Reduce the number of open files/tabs

---

## Appendix B: Discord Notifications

This appendix covers setting up Discord integration for real-time person detection alerts. There are **two options** available:

### Option 1: Webhooks (Simpler, One-Way)
- Real-time notifications in Discord when people are detected
- Alerts when the area becomes clear
- Timestamped messages showing detection events
- No complex bot hosting required
- **Best for:** Simple notifications only

### Option 2: Discord Bot (Advanced, Two-Way)
- Everything webhooks can do, plus:
- Interactive commands (!status, !detect, !ping)
- Query camera status from Discord
- Bot runs on the Raspberry Pi
- **Best for:** Interactive monitoring and control

**This guide covers webhooks. For bot setup, see [DISCORD_BOT_PLAN.md](DISCORD_BOT_PLAN.md).**

---

## Webhook Setup (Option 1)

### What You'll Get

- Real-time notifications in Discord when people are detected
- Alerts when the area becomes clear
- Timestamped messages showing detection events
- No complex bot hosting required

---

### Prerequisites

- A Discord account
- A Discord server where you have "Manage Webhooks" permission
  - You can create your own server for free if needed

---

### Step 1: Create a Discord Server (If Needed)

If you already have a Discord server where you want notifications, skip to Step 2.

#### Create Your Own Server

1. Open Discord (desktop app or web)
2. Click the **+** button in the left sidebar
3. Select **Create My Own**
4. Choose **For me and my friends** (or customize as needed)
5. Name it something like "Camera Notifications" or "OAK-D Monitor"
6. Click **Create**

---

### Step 2: Create a Webhook

A webhook is a special URL that allows the camera to send messages to Discord without running a full bot.

#### Step-by-Step Instructions

1. **Navigate to Server Settings**
   - Right-click your Discord server name (top-left)
   - Select **Server Settings**

2. **Open Integrations**
   - In the left sidebar, click **Integrations**

3. **Create Webhook**
   - Click **Webhooks** (or **View Webhooks** if webhooks already exist)
   - Click **New Webhook** (or **Create Webhook**)

4. **Configure the Webhook**
   - **Name**: `OAK-D Camera` (or whatever you prefer)
   - **Channel**: Select the channel where you want notifications
     - You can create a dedicated channel like `#camera-alerts`
   - **Icon** (optional): Upload a camera emoji or icon

5. **Copy Webhook URL**
   - Click **Copy Webhook URL** button
   - **IMPORTANT**: Keep this URL secret! Anyone with this URL can send messages to your channel
   - The URL looks like:
     ```
     https://discord.com/api/webhooks/1234567890/AbCdEfGhIjKlMnOpQrStUvWxYz...
     ```

6. **Save Changes**
   - Click **Save Changes** at the bottom

---

### Step 3: Configure the Camera System

Now we'll tell the camera where to send notifications.

#### On Your Raspberry Pi

1. **SSH into your Pi**

   ```bash
   ssh smartobjects1.local
   # or
   ssh smartobjects2.local
   ```

2. **Navigate to the project directory**

   ```bash
   cd ~/oak-projects
   ```

3. **Create environment configuration file**

   ```bash
   nano .env
   ```

4. **Add the webhook URL**
   Paste the following, replacing with your actual webhook URL:

   ```bash
   # Discord Webhook Configuration
   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_URL_HERE
   ```

5. **Save the file**
   - Press `Ctrl+O` then `Enter` to save
   - Press `Ctrl+X` to exit

6. **Secure the file** (so other users can't read your webhook)
   ```bash
   chmod 600 .env
   ```

#### Install Required Dependencies

```bash
# Activate shared virtual environment
activate-oak

# Install required packages (if not already installed during initial setup)
pip install requests aiohttp python-dotenv
```

---

### Step 4: Test the Webhook

Let's verify everything is working before integrating with the camera.

```bash
# Activate shared virtual environment if not already
activate-oak

# Run test notification
python3 discord_notifier.py
```

**Expected output:**

```
🧪 Testing Discord notification...
✅ Notification sent successfully!
   Check your Discord channel to verify
```

**Check Discord** - You should see a message from "OAK-D Camera" with a test notification!

#### Manual Test

You can also send custom messages:

```bash
python3 discord_notifier.py "Hello from the camera system!"
```

---

### Step 5: Run the Camera with Discord Notifications

Now that the webhook is configured, run the person detector:

```bash
# Activate shared virtual environment
activate-oak

# Run with Discord notifications
python3 person_detector.py --discord

# Or with logging
python3 person_detector.py --discord --log

# Or with video display (VNC Pi only)
python3 person_detector.py --discord --display

# Quiet mode (only notify on detection, not when area clears)
python3 person_detector.py --discord --discord-quiet
```

You should now see Discord notifications when:

- ✅ People are detected
- ✅ The area clears (no people)
- ✅ The system starts up
- ✅ The system shuts down

---

### Step 6: Auto-Start with Notifications (Optional)

To have the camera automatically send notifications on boot, update the systemd service:

```bash
sudo nano /etc/systemd/system/person-detector.service
```

Make sure it looks like this:

```ini
[Unit]
Description=OAK-D Person Detector with Discord Notifications
After=network.target

[Service]
Type=simple
User=carrie
WorkingDirectory=/home/carrie/oak-projects
EnvironmentFile=/home/carrie/oak-projects/.env
ExecStart=/opt/oak-shared/venv/bin/python3 /home/carrie/oak-projects/person_detector.py --discord --log
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Note:** Replace `carrie` with your actual username.

Key changes:

- Added `EnvironmentFile=/home/carrie/oak-projects/.env` to load the webhook URL
- Added `--discord` flag to enable notifications
- Uses shared venv at `/opt/oak-shared/venv/`

#### Reload and restart the service

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Restart the service
sudo systemctl restart person-detector

# Check status
sudo systemctl status person-detector

# View logs
journalctl -u person-detector -f
```

---

### Discord Customization

#### Change Notification Settings

Edit `person_detector.py` to customize what gets sent to Discord:

```python
# Example: Only notify on person detection (not when area clears)
if person_detected and not last_status:
    send_notification(f"🟢 PERSON DETECTED (count: {person_count})")
    # Remove the "no person" notification
```

#### Change Username/Icon

Modify calls in `person_detector.py`:

```python
send_notification(
    "🟢 PERSON DETECTED",
    username="Security Camera 1"
)
```

#### Disable Timestamps

```python
send_notification("Message here", add_timestamp=False)
```

---

### Discord Troubleshooting

#### "DISCORD_WEBHOOK_URL not set" Error

**Problem**: The `.env` file isn't being loaded or doesn't exist.

**Solutions**:

1. Make sure you created `.env` in `~/oak-projects/`
2. Check the file contents: `cat ~/oak-projects/.env`
3. Ensure `python-dotenv` is installed: `pip install python-dotenv`

#### "Notification failed: 404" Error

**Problem**: The webhook URL is incorrect or the webhook was deleted.

**Solutions**:

1. Go back to Discord → Server Settings → Integrations → Webhooks
2. Verify the webhook still exists
3. Copy the URL again and update `.env`
4. Make sure there are no extra spaces in the `.env` file

#### "Notification timed out" Error

**Problem**: Network connectivity issue between Pi and Discord.

**Solutions**:

1. Check internet connection: `ping discord.com`
2. Check Pi's network settings
3. Try the test script again: `python3 discord_notifier.py`

#### Messages Not Appearing in Discord

**Checklist**:

- [ ] Webhook exists in Discord settings
- [ ] Webhook points to correct channel
- [ ] `.env` file has correct URL
- [ ] Test script works: `python3 discord_notifier.py`
- [ ] Virtual environment is activated
- [ ] `requests` and `python-dotenv` are installed

#### Permission Issues with .env File

```bash
# Fix file permissions
chmod 600 ~/oak-projects/.env

# Verify ownership
ls -la ~/oak-projects/.env
# Should show: -rw------- 1 [user] [user] ...
```

---

### Security Best Practices

#### Keep Your Webhook Secret

- ✅ **DO**: Store webhook URL in `.env` file with restricted permissions
- ✅ **DO**: Add `.env` to `.gitignore` if using version control
- ❌ **DON'T**: Share webhook URL publicly
- ❌ **DON'T**: Commit `.env` to GitHub/Git repositories
- ❌ **DON'T**: Post webhook URL in Discord or other public places

#### Regenerate Webhook If Leaked

If your webhook URL is accidentally exposed:

1. Go to Discord → Server Settings → Integrations → Webhooks
2. Find your webhook
3. Click the webhook name
4. Scroll to bottom and click **Delete Webhook**
5. Create a new webhook (follow Step 2 again)
6. Update `.env` with new URL

---

### Using Multiple Cameras

If you have multiple cameras (multiple Pis), you can:

#### Option 1: Different Webhooks (Different Channels)

Create separate webhooks for each camera:

```bash
# On Pi 1
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/camera1_webhook

# On Pi 2
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/camera2_webhook
```

#### Option 2: Same Webhook (Same Channel) with Different Names

Use the same webhook but different usernames:

```python
# On Pi 1
send_notification("Person detected", username="Camera 1 - Front Door")

# On Pi 2
send_notification("Person detected", username="Camera 2 - Backyard")
```

---

## Extending This Template - Interactive Smart Objects

This template is designed for **real-time collaboration with cameras as smart objects**. The key idea: you should be able to command, reconfigure, and coordinate cameras dynamically through Discord.

### 🎮 Interactive Control Ideas

**Dynamic Reconfiguration:**
- Add `!set-threshold 0.7` - Change detection sensitivity in real-time
- Add `!detect-mode person` or `!detect-mode car` - Switch what the camera looks for
- Add `!toggle-notifications` - Turn alerts on/off without restarting
- Add `!set-fps 30` - Adjust camera frame rate on the fly

**Physical Coordination:**
- Add `!request-move "point at door"` - Ask someone to reposition the camera
- Add `!camera-location "classroom-front"` - Track where cameras are pointed
- Multiple people collaborate to position cameras optimally

**Multi-Camera Orchestration:**
- `!camera1 detect-person` and `!camera2 detect-car` - Different cameras, different jobs
- `!coordinate-all` - All cameras switch to the same detection mode
- `!status-all` - See what every camera is currently doing
- Cameras report to you when they need adjustment or repositioning

**Live Experimentation:**
- Change models without SSH: `!use-model yolov8-nano`
- Toggle features: `!enable-zones`, `!disable-debouncing`
- A/B test different configurations across cameras
- Students suggest changes via Discord, camera responds

**Rich Communication:**
- Create custom Discord embeds with rich formatting
- Add database logging for detection history
- Implement scheduled reports: `!daily-summary`, `!weekly-stats`
- Video recording on demand: `!record 30` (record 30 seconds)
- Automatic video clips when interesting events occur

### 💡 The Core Concept

Think of cameras as **responsive team members**, not just sensors:
- They can be asked to change what they're doing
- They respond to commands immediately
- Multiple cameras can coordinate their sensing
- Physical repositioning is part of the interaction
- Configuration changes happen through conversation, not SSH

This makes experimentation fast and collaborative - you're having a dialog with your smart objects, not just configuring them once and watching passively.

---

## Next Steps

Once basic detection is working, you might want to explore:

1. **Discord Bot Integration** — Set up interactive bot for two-way communication
   - Follow [DISCORD_BOT_PLAN.md](DISCORD_BOT_PLAN.md) for complete setup guide
   - Query camera status with commands like !status, !detect, !ping

2. **Different models** — Try other YOLO models from Luxonis Hub
   - Browse models at https://models.luxonis.com
   - Change model reference in person_detector.py `--model` argument
   - Example: `--model luxonis/yolov8-nano:r2-coco-640x640`

3. **Depth integration** — Get distance to detected persons
   - Use `depthai-nodes` spatial detection features
   - See [DepthAI 3.x Documentation](https://docs.luxonis.com/software-v3/depthai/)

4. **Recording** — Save video clips when people are detected
   - Use DepthAI 3.x VideoEncoder node
   - See examples in depthai-python repository

5. **Notifications** — See Appendix B for Discord webhook setup

6. **Web dashboard** — Flask/FastAPI server to view status remotely

7. **Multiple cameras** — Run detection on multiple OAK-D cameras

**Resources for learning more:**
- **DepthAI 3.x Documentation**: https://docs.luxonis.com/software-v3/depthai/
- **depthai-nodes GitHub**: https://github.com/luxonis/depthai-nodes
- **Luxonis Hub Models**: https://models.luxonis.com
- **OAK Examples & Experiments**: https://github.com/luxonis/depthai-experiments
- **Tutorials**: https://docs.luxonis.com/software-v3/tutorials/

---

## Contributors

- [Your name here]
- [Student names]

## License

MIT License - Feel free to use and modify for educational purposes.
