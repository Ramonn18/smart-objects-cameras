# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **template repository** for creating Discord bots that communicate with Luxonis OAK-D cameras (DepthAI 3.x). It serves as an educational stepping stone for students learning to build smart object systems with computer vision and interactive communication.

**What This Template Provides:**
- Person detection using YOLO on OAK-D's onboard VPU
- Discord bot integration (commands: !status, !detect, !screenshot)
- Discord webhook notifications (automatic alerts)
- Temporal smoothing and debouncing for stable detection
- Camera frame capture and status file management

**Learning Goals:**
- Browse documentation to understand DepthAI 3.x architecture
- Collaborate on new bot ideas and communication patterns
- Extend to multiple cameras (smart objects network)
- Experiment with different detection models and features
- Build interactive systems that bridge physical sensing and digital communication

**Target Hardware:**
- **Orbit (16GB)**: Desktop environment with VNC access (hostname: `orbit`)
- **Gravity (16GB)**: Desktop environment with VNC access (hostname: `gravity`)
- **Horizon (16GB)**: Desktop environment with VNC access (hostname: `horizon`)

**Note:** All three Raspberry Pis have VNC enabled, but only one user can hold the desktop seat at a time. Multiple users can SSH in simultaneously.

**Students should feel encouraged to:**
- Fork this template for their own projects
- Modify detection logic for different use cases
- Create new Discord commands and bot features
- Experiment with multiple cameras communicating together
- Share discoveries and collaborate on improvements

## Development Environment

### Python Setup
All Python development uses a **shared virtual environment** for efficiency:

```bash
# Activate the shared venv
source /opt/oak-shared/venv/bin/activate
# Or use the alias: activate-oak

# Always activate before running scripts
```

The shared virtual environment includes:
- `depthai` - Luxonis DepthAI 3.x for OAK camera access (low-level pipeline API)
- `depthai-nodes` - High-level neural network parsing nodes for depthai 3.x
- `opencv-python` - Computer vision library
- `numpy` - Numerical computing
- `requests` - HTTP library for Discord webhooks
- `aiohttp` - Async HTTP for Discord notifications
- `python-dotenv` - Environment variable management
- `discord.py` - Discord bot library (optional, for interactive bot)

### Key Paths
```
/opt/oak-shared/
└── venv/                    # Shared Python virtual environment (all users)

~/oak-projects/              # Personal project directory
├── person_detector.py       # Main detection script
├── discord_notifier.py      # Discord webhook notification module
├── discord_bot.py           # Discord bot for two-way interaction (optional)
├── camera_status.json       # Status file for bot integration (auto-generated)
├── .env                     # Environment variables (webhook URL, bot token)
└── person_detection_*.log   # Detection logs (when using --log)
```

### User Accounts
- Main user account created during OS installation (e.g., `carrie`, not `pi`)
- `root` - Superuser accessed via `sudo`
- Modern Raspberry Pi OS doesn't create a default `pi` user

## Running the Person Detector

The main script is `person_detector.py` with the following usage:

```bash
# Basic detection (console output only)
python3 person_detector.py

# With video display (requires X11 display, use on VNC Pi or via X forwarding)
python3 person_detector.py --display

# With file logging
python3 person_detector.py --log

# With Discord notifications
python3 person_detector.py --discord

# Discord notifications (quiet mode - only alerts on detection, not when clear)
python3 person_detector.py --discord --discord-quiet

# Adjust detection threshold (0.0-1.0, default 0.5)
python3 person_detector.py --threshold 0.7

# Combine options
python3 person_detector.py --discord --log --threshold 0.6
```

Stop the script with `Ctrl+C`.

## Architecture (depthai 3.x with depthai-nodes)

### Pipeline-Based Approach
The person detector uses **depthai 3.x** low-level pipeline API with **depthai-nodes** for high-level neural network parsing:

**Key Components:**

1. **Device Connection** - `dai.Device()`
   - Connects to OAK camera
   - Creates pipeline context
   - Manages device lifecycle
   - **API Note**: Use `device.getDeviceId()` not deprecated `getMxId()`

2. **Pipeline** - `dai.Pipeline(device)`
   - Context manager for proper resource cleanup
   - Container for all nodes (camera, NN, outputs)

3. **ColorCamera Node** - `pipeline.create(dai.node.ColorCamera)`
   - RGB camera configuration (512x288 @ 15 FPS)
   - Preview output sized to match model input
   - Non-interleaved RGB format
   - **API Note**: ColorCamera is deprecated (warning shown) but still works
   - New `dai.node.Camera` has a completely different API - use ColorCamera for now

4. **Neural Network with Parser** - `ParsingNeuralNetwork` from depthai-nodes
   - Uses YOLO v6 model from Luxonis Hub: `luxonis/yolov6-nano:r2-coco-512x288`
   - Model description: `dai.NNModelDescription(model_ref, platform=platform)`
   - Model archive: `dai.NNArchive(dai.getModelFromZoo(model_description))`
   - **Automatic parsing**: depthai-nodes handles YOLO output parsing
   - Detects 80 COCO classes (person is class 0)
   - Runs on OAK-D's onboard Myriad X VPU

5. **Output Queues** - Created directly from node outputs
   - **IMPORTANT**: No `XLinkOut` or `XOut` nodes in depthai 3.x!
   - Create queues directly: `output.createOutputQueue(maxSize=4, blocking=False)`
   - Example: `q_det = nn_with_parser.out.createOutputQueue(...)`
   - Example: `q_preview = cam_rgb.preview.createOutputQueue(...)`
   - Must be created before `pipeline.start()`
   - Returns parsed detection objects with label, confidence, bbox

### Detection Logic
The script monitors detection events with temporal smoothing:
- **Debouncing (1.5 seconds)**: State must persist before triggering to prevent flickering
- Filters for person detections only (COCO class 0)
- Tracks when people appear/disappear
- Counts number of people detected
- Only logs on confirmed state changes to reduce noise
- Writes status to JSON file (`camera_status.json`) for Discord bot integration

### Systemd Service
The detector can auto-start on boot via systemd:

```bash
# Service controls
sudo systemctl status person-detector
sudo systemctl start person-detector
sudo systemctl stop person-detector

# View live logs
journalctl -u person-detector -f
```

Service file location: `/etc/systemd/system/person-detector.service`

## Discord Integration

The system supports two types of Discord integration:

### Option 1: Webhooks (One-Way Notifications)
Sends real-time notifications to Discord when detection events occur.

**Setup:**
1. Create a Discord webhook (see `README.md` Appendix B for detailed instructions)
2. Create a `.env` file in `~/oak-projects/`
3. Add your webhook URL to `.env`:
   ```bash
   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_TOKEN
   ```
4. Install dependencies:
   ```bash
   pip install requests aiohttp python-dotenv
   ```

### Option 2: Discord Bot (Two-Way Interaction)
Interactive bot that can respond to commands and query camera status.

**Setup:**
See `DISCORD_BOT_PLAN.md` for complete setup guide (30-45 minutes).

**Key Features:**
- Interactive commands: `!ping`, `!status`, `!detect`, `!help`
- Reads status from `camera_status.json` (written by person_detector.py)
- Bot runs on Raspberry Pi using discord.py library
- Requires Discord bot token in `.env`:
  ```bash
  DISCORD_BOT_TOKEN=your_bot_token_here
  ```

**Integration with person_detector.py:**
- person_detector.py writes detection status to `~/oak-projects/camera_status.json`
- discord_bot.py reads this file when users query `!status` or `!detect`
- Status file updated on startup and whenever detection state changes

### Usage
```bash
# Enable Discord notifications
python3 person_detector.py --discord

# Quiet mode (only notify on detection, not when area clears)
python3 person_detector.py --discord --discord-quiet
```

### Discord Module
The `discord_notifier.py` module provides simple webhook-based notifications:

```python
from discord_notifier import send_notification

# Send a notification
send_notification("🟢 Person detected!")

# Custom username
send_notification("Alert!", username="Camera 1")

# Without timestamp
send_notification("Message", add_timestamp=False)
```

### Testing
```bash
# Test notification system
python3 discord_notifier.py

# Send custom test message
python3 discord_notifier.py "Test message from camera"
```

### Systemd Integration
To enable Discord notifications in the auto-start service, add the `--discord` flag:

```ini
[Service]
EnvironmentFile=/home/pi/oak-projects/.env
ExecStart=/home/pi/oak-projects/venv/bin/python3 /home/pi/oak-projects/person_detector.py --discord --log
```

## Hardware Considerations

### Camera Connection
- OAK-D connects via USB 3.0 (blue ports preferred)
- Powered USB hub recommended for stability
- USB permissions configured via udev rules at `/etc/udev/rules.d/80-movidius.rules`

### Testing Camera
```bash
# Check camera is detected (depthai 3.x)
python3 -c "import depthai as dai; devices = dai.Device.getAllAvailableDevices(); print(f'Found {len(devices)} camera(s)')"

# Get device info
python3 -c "import depthai as dai; device = dai.Device(); print(f'Device: {device.getMxId()}, Platform: {device.getPlatformAsString()}')"

# Should output device count and ID
# If error, check USB connection
```

### Camera Firmware Notes

**USB OAK-D cameras** (OAK-D, OAK-D-Lite, OAK-1) boot from the host computer and don't require firmware updates. They automatically use the bootloader embedded in the depthai library.

**PoE cameras** (OAK-D-POE) have onboard flash and can be updated if needed.

```bash
# Check current bootloader version
python3 -c "import depthai as dai; device = dai.Device(); print(f'Bootloader: {device.getBootloaderVersion()}')"
```

**Expected outputs:**
- **USB cameras**: Will typically show `None` - this is normal and expected
- **PoE cameras**: Will show actual bootloader version

**For most USB setups, you can skip firmware updates entirely.** The depthai library manages bootloader compatibility automatically.

See [INITIAL_SETUP.md](INITIAL_SETUP.md) for detailed firmware information.

### USB Permissions
If getting permission errors:
```bash
# Reload udev rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Or add user to video group
sudo usermod -aG video $USER
# Logout and back in for changes to take effect
```

## Code Modification Guidelines (depthai 3.x)

### Important API Notes for depthai 3.x

**Before modifying pipelines, be aware of these API differences:**

1. **No XLinkOut/XOut nodes** - Create queues directly from outputs: `node.output.createOutputQueue()`
2. **ColorCamera is deprecated but works** - New `Camera` node has different API (not well documented yet)
3. **Use `getDeviceId()` not `getMxId()`** - Old method is deprecated
4. **Queue creation order matters** - Create all queues BEFORE `pipeline.start()`

See "Common Issues → depthai 3.x API Errors" section for detailed examples.

### Changing Detection Models
To use a different model, modify the model reference in `person_detector.py`:

```python
# Use the --model command-line argument:
python3 person_detector.py --model luxonis/yolov8-nano:r2-coco-640x640

# Or modify the default in the script:
parser.add_argument('--model', type=str, default='luxonis/yolov6-nano:r2-coco-512x288')
```

**Available models from Luxonis Hub:**
- `luxonis/yolov6-nano:r2-coco-512x288` (current default, 80 COCO classes)
- `luxonis/yolov8-nano:r2-coco-640x640` (YOLOv8, more accurate)
- `luxonis/yolov5-nano:r2-coco-416x416` (YOLOv5 variant)
- Browse more at: https://models.luxonis.com

**Model architecture:**
```python
# How models are loaded from Luxonis Hub:
model_description = dai.NNModelDescription(args.model, platform=platform)
nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))

# ParsingNeuralNetwork from depthai-nodes handles YOLO parsing
nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
    cam_rgb.preview, nn_archive
)
```

### Adding Depth Information
The OAK-D supports stereo depth. With depthai 3.x, you need to:

1. Create stereo depth nodes (left and right mono cameras)
2. Configure depth alignment
3. Use spatial detection features

This is more complex than the high-level SDK. Refer to:
- https://docs.luxonis.com/software-v3/depthai/
- depthai-python examples for spatial detection

### Adjusting Performance
- **Detection threshold**: Use `--threshold` CLI arg (higher = fewer false positives)
- **Frame rate**: Modify `cam_rgb.setFps(15)` in the script
- **Resolution**: Modify `cam_rgb.setPreviewSize(512, 288)` to match model input size
- **Debounce time**: Adjust `DEBOUNCE_SECONDS = 1.5` to change temporal smoothing

## Networking

### SSH Key-Based Authentication
The Pis are configured with SSH key-based authentication for passwordless, secure access.

**SSH Key Setup:**
- Project-specific key name: `id_ed25519_smartobjects`
- SSH config file configured with host aliases and IP addresses
- See `INITIAL_SETUP.md` for setup details

**Example SSH config (`~/.ssh/config`):**
```
Host orbit
    HostName 192.168.x.x
    User yourusername
    IdentityFile ~/.ssh/id_ed25519_smartobjects

Host gravity
    HostName 192.168.x.x
    User yourusername
    IdentityFile ~/.ssh/id_ed25519_smartobjects

Host horizon
    HostName 192.168.x.x
    User yourusername
    IdentityFile ~/.ssh/id_ed25519_smartobjects
```

**To connect from your computer:**
```bash
# SSH access (no password needed with keys configured)
ssh orbit       # Camera: Orbit
ssh gravity     # Camera: Gravity
ssh horizon     # Camera: Horizon
```

**If you need to add your SSH key to the Pi:**
```bash
# On your Mac/Linux laptop (if SSH config is set up)
ssh-copy-id orbit

# Or specify the key explicitly
ssh-copy-id -i ~/.ssh/id_ed25519_smartobjects.pub orbit

# Or manually copy your public key
cat ~/.ssh/id_ed25519_smartobjects.pub | ssh orbit "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

**For Windows users without ssh-copy-id:**
```powershell
# Copy your public key manually
type $env:USERPROFILE\.ssh\id_ed25519_smartobjects.pub | ssh orbit "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

### VNC Access
All three Raspberry Pis have VNC enabled. Connect via RealVNC Viewer using the host name (e.g., `orbit`, `gravity`, or `horizon`).

**Note:** Only one user can hold the VNC desktop seat at a time, but multiple users can SSH in simultaneously.

### WiFi Network Management
The Pis can be moved between different WiFi networks (e.g., home to classroom).

**Note:** Modern Raspberry Pi OS (Bookworm+) uses **NetworkManager** instead of wpa_supplicant. The `wpa_supplicant.conf` file may not exist.

**Method 1: Using nmtui (Best for pre-configuring networks)**
Text-based menu interface - perfect for adding networks you're not currently connected to:

```bash
# Open NetworkManager Text UI
sudo nmtui

# Navigate with arrow keys, select with Enter
# "Edit a connection" → "Add" → "Wi-Fi"
# Enter SSID and password
# Tab to "OK" and save
```

**Benefits:** Can add classroom WiFi while at home (doesn't need network in range)

**Method 2: Using nmcli (Command line - requires network in range)**
```bash
# List available networks
nmcli device wifi list

# Connect to a network (saves automatically - network must be in range!)
sudo nmcli device wifi connect "ClassroomWiFi" password "classpassword"

# List saved connections
nmcli connection show

# Switch to a saved network
nmcli connection up "HomeWiFi"
```

**Method 3: Using raspi-config**
```bash
sudo raspi-config
# Navigate to: System Options → Wireless LAN
# Enter new SSID and password
```

**Auto-connection behavior:**
Once you've saved multiple networks (via nmtui, nmcli, or raspi-config), the Pi will automatically connect to any available saved network.

**Legacy: wpa_supplicant (may not apply to modern Pi OS)**
If your Pi still uses wpa_supplicant:
```bash
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
# Add network blocks with priority settings
```

### Multi-User Access and Coordination

**Critical concept:** Each Pi has ONE camera. Only ONE person should run `person_detector.py` at a time.

**Smart Object Feature:** The camera **automatically announces** who's using it when run with `--discord`:
- **Startup:** `🎥 **alice** is now running person_detector.py on **orbit**`
- **Shutdown:** `📴 **alice** stopped person_detector.py on **orbit** - camera is free`
- **Status file includes:** Username and hostname (`camera_status.json`)

**Collaboration Model:**
- **One person runs the detector** - Camera auto-announces via Discord (no manual coordination needed)
- **Everyone else can SSH in** - View/edit code simultaneously via VS Code Remote
- **Create personal test copies** - `person_detector_alice.py`, `person_detector_bob.py`
- **Check Discord to see who's using camera** - Or check: `ps aux | grep person_detector.py`

**Implementation Details:**
The script captures username (`getpass.getuser()`) and hostname (`socket.gethostname()`) at startup and includes them in:
1. Discord startup/shutdown notifications
2. Status file (`camera_status.json`) - adds `username` and `hostname` fields
3. Discord bot can display who's currently running the detector

**Best Practices:**
- Always run with `--discord` flag so others know when camera is free
- Check Discord channel before starting to see if camera is in use
- Use VS Code Remote SSH for individual editing sessions (multiple people can edit different files)
- Create personal script copies to test later:
  ```bash
  cp person_detector.py person_detector_yourname.py
  ```
- Use Git branches for feature development (see `GIT_COLLABORATION.md`)
- Each user should add their SSH key to `~/.ssh/authorized_keys`

**Adding Multiple Users (Optional):**
```bash
# Create a new user account
sudo adduser studentname

# Add to necessary groups
sudo usermod -aG video,gpio,i2c,spi studentname

# Copy project access
sudo cp -r /home/pi/oak-projects /home/studentname/
sudo chown -R studentname:studentname /home/studentname/oak-projects
```

**Shared Model Cache (Optional - Development Scenario Only):**

**When needed:** Only if multiple students want to test personal script copies (`person_detector_alice.py`, etc.) at different times and encounter:

```
RuntimeError: filesystem error: cannot remove: Permission denied
[/tmp/yolov6n-r2-288x512.rvc2.tar.xz/config.json]
```

**Root Cause:** DepthAI downloads YOLO models to `/tmp` by default. First user owns the files, subsequent users testing later can't access them.

**Solution (optional):** Run the `setup_shared_model_cache.sh` script (included in repository):

```bash
# Copy script to Pi and run with sudo
scp setup_shared_model_cache.sh orbit:~/
ssh orbit
chmod +x ~/setup_shared_model_cache.sh
sudo ~/setup_shared_model_cache.sh
```

Creates `/opt/depthai-cache` (777) with `DEPTHAI_ZOO_CACHE` environment variable system-wide.

**Note:** For typical collaborative work where one person runs the main script at a time, this isn't necessary. The model cache is only an issue when different users independently test their own script copies.

### VS Code Remote Development
Recommended for code editing:
1. Install "Remote - SSH" extension
2. **macOS ONLY:** Grant VS Code "Local Network" permission in System Settings → Privacy & Security
3. `Ctrl+Shift+P` → "Remote-SSH: Connect to Host"
4. Enter: `orbit`, `gravity`, or `horizon` (should auto-complete from your SSH config)
5. Open folder: `/home/[username]/oak-projects`
6. Select Python interpreter: `~/oak-projects/venv/bin/python` (or `/opt/oak-shared/venv/bin/python` for shared venv)

**Common Issue:** On macOS, VS Code will fail with "No route to host" if Local Network permission isn't granted, even though terminal SSH works fine.

Full setup instructions in `README.md` (Appendix A).

## Common Issues

### depthai 3.x API Errors

#### AttributeError: module 'depthai.node' has no attribute 'XLinkOut' or 'XOut'
**Problem**: Trying to use `XLinkOut` or `XOut` nodes from older DepthAI versions.

**Solution**: depthai 3.x doesn't use explicit XLinkOut nodes. Create output queues directly from node outputs:
```python
# ❌ Old way (doesn't work):
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("preview")
cam.preview.link(xout.input)
q = pipeline.createOutputQueue("preview")

# ✅ New way (depthai 3.x):
q = cam.preview.createOutputQueue(maxSize=4, blocking=False)
```

#### AttributeError: 'Camera' object has no attribute 'setPreviewSize'
**Problem**: The new `dai.node.Camera` has a completely different API than `ColorCamera`.

**Solution**: Use `ColorCamera` for now (it's deprecated but still works):
```python
# ✅ Use ColorCamera (works with familiar API):
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(512, 288)
cam.setInterleaved(False)
cam.setFps(15)

# ⚠️ New Camera node API is different (not documented well yet)
# Stick with ColorCamera until Camera API is stable
```

#### DeprecationWarning: Use getDeviceId() instead
**Problem**: `getMxId()` is deprecated.

**Solution**: Use `getDeviceId()` instead:
```python
# ❌ Old: device.getMxId()
# ✅ New: device.getDeviceId()
device_id = device.getDeviceId()
```

### Camera Not Found
```bash
# Check USB device
lsusb | grep Myriad
# Should show: Intel Movidius MyriadX (ID 03e7:2485)

# If not found:
# - Try different USB port (use USB 3.0)
# - Check cable connection
# - Use powered USB hub
# - Verify udev rules are installed
```

### Import Errors
If getting `ModuleNotFoundError: No module named 'depthai'` or `'depthai_nodes'`:
- Ensure shared virtual environment is activated: `activate-oak` (alias for `source /opt/oak-shared/venv/bin/activate`)
- Prompt should show `(venv)` prefix when activated
- Verify installation: `pip list | grep depthai` should show both `depthai` and `depthai-nodes`

### Display Issues
When using `--display` flag:
- Requires X11 display server
- Works when connected via VNC
- For SSH-only access, requires X forwarding: `ssh -X orbit`
- Press 'q' in OpenCV window to exit (in addition to Ctrl+C)

### Performance Problems
- Check CPU usage: `htop`
- Check memory: `free -h`
- Lower frame rate or resolution if needed
- Ensure adequate power supply (27W official Pi 5 adapter)
- Check for thermal throttling: `vcgencmd get_throttled`

### Discord Notification Issues
If Discord notifications aren't working:

```bash
# Test the notifier directly
python3 discord_notifier.py

# Check .env file exists and has webhook URL
cat ~/oak-projects/.env

# Verify internet connectivity
ping discord.com

# Check for missing dependencies
pip install requests aiohttp python-dotenv
```

Common Discord errors:
- **"DISCORD_WEBHOOK_URL not set"**: Create `.env` file with webhook URL
- **"404 Not Found"**: Webhook was deleted or URL is incorrect
- **"Timeout"**: Network connectivity issue

See `README.md` Appendix B for complete troubleshooting guide.

## Extending This Template - Interactive Smart Objects

### Philosophy: Cameras as Conversational Agents

The key concept is treating cameras as **responsive team members** you can command and reconfigure in real-time through Discord, not just passive sensors you configure once via SSH.

**Interactive Control Patterns:**

**1. Dynamic Reconfiguration (no restart required):**
```python
# Example: Add !set-threshold command
@bot.command(name='set-threshold')
async def set_threshold(ctx, value: float):
    # Write new threshold to config file
    # person_detector.py reads config file periodically
    # Threshold changes without restarting camera
```

**2. Mode Switching:**
```python
# Switch what the camera detects on the fly
@bot.command(name='detect-mode')
async def detect_mode(ctx, mode: str):
    # mode = "person" | "car" | "all"
    # Update filter in real-time
    # Camera responds: "Now detecting: cars"
```

**3. Physical Coordination:**
```python
# Request physical camera repositioning
@bot.command(name='request-move')
async def request_move(ctx, direction: str):
    await ctx.send(f"📍 Please reposition camera: {direction}")
    await ctx.send("Reply with !confirm-moved when done")
```

**4. Multi-Camera Orchestration:**
```python
# Coordinate multiple cameras dynamically
@bot.command(name='coordinate-all')
async def coordinate_all(ctx, command: str):
    # Send command to all cameras simultaneously
    # Each camera responds with acknowledgment
    # Cameras work together on shared task
```

### Example Student Extensions

**Real-Time Configuration Commands:**
- `!set-threshold <value>` - Adjust sensitivity without restart
- `!set-fps <value>` - Change frame rate on the fly
- `!toggle-notifications` - Enable/disable alerts
- `!use-model <model_name>` - Switch detection models
- `!enable-zones` / `!disable-zones` - Toggle zone detection

**Physical Interaction:**
- `!camera-location <name>` - Track where camera is pointed
- `!request-move <description>` - Ask for camera repositioning
- `!confirm-moved` - Acknowledge camera has been repositioned
- `!request-adjustment "tilt up"` - Fine-tune camera angle

**Multi-Camera Control:**
- `!camera1 <command>` - Send command to specific camera
- `!all-cameras <command>` - Broadcast to all cameras
- `!status-all` - Query status of all cameras
- `!coordinate detect-person` - All cameras switch to person detection

**Live Experimentation:**
- `!a-b-test camera1 threshold=0.5 camera2 threshold=0.7` - Compare configurations
- `!show-config` - Display current camera settings
- `!reset-to-default` - Return to template configuration

**Rich Communication & Data:**
- Create custom Discord embeds with rich formatting (status cards, progress bars)
- Add database logging (SQLite/PostgreSQL) for detection history
- Implement scheduled reports: `!daily-summary`, `!weekly-stats`, `!top-detections`
- Video recording on demand: `!record 30` (capture 30 second clip)
- Automatic video clips triggered by detection events
- Export data: `!export-csv today`, `!export-json last-week`

### Implementation Pattern: Config File Polling

For real-time reconfiguration without restart:

```python
# In person_detector.py - add config file monitoring
CONFIG_FILE = Path.home() / "oak-projects" / "camera_config.json"
last_config_check = 0
CONFIG_CHECK_INTERVAL = 2  # Check every 2 seconds

# In detection loop:
if time.time() - last_config_check >= CONFIG_CHECK_INTERVAL:
    # Read config file (written by Discord bot)
    config = json.loads(CONFIG_FILE.read_text())
    threshold = config.get('threshold', 0.5)
    detect_mode = config.get('mode', 'person')
    notifications_enabled = config.get('notifications', True)
    last_config_check = time.time()
```

### Implementation Hints

**Discord Embeds (Rich Formatting):**
```python
import discord

@bot.command(name='status-rich')
async def status_rich(ctx):
    embed = discord.Embed(
        title="📸 Camera Status",
        description="Real-time detection status",
        color=discord.Color.green()
    )
    embed.add_field(name="Status", value="🟢 Online", inline=True)
    embed.add_field(name="Detections", value="2 people", inline=True)
    embed.add_field(name="FPS", value="15", inline=True)
    embed.set_thumbnail(url="attachment://latest_frame.jpg")

    await ctx.send(embed=embed, file=discord.File(SCREENSHOT_FILE))
```

**Video Recording (Using DepthAI VideoEncoder):**
```python
# In person_detector.py - add video encoder node
video_enc = pipeline.create(dai.node.VideoEncoder)
video_enc.setDefaultProfilePreset(15, dai.VideoEncoderProperties.Profile.H264_MAIN)
cam_rgb.video.link(video_enc.input)

# Record when commanded via config file
if config.get('recording', False):
    # Save video frames
    # Stop after duration specified in config
```

**Database Logging (SQLite example):**
```python
import sqlite3
from datetime import datetime

def log_detection(detected, count):
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''INSERT INTO detections (timestamp, detected, count)
                 VALUES (?, ?, ?)''', (datetime.now(), detected, count))
    conn.commit()
    conn.close()

# In detection loop:
if person_detected != last_status:
    log_detection(person_detected, person_count)
```

**Scheduled Reports (Discord scheduled tasks):**
```python
from discord.ext import tasks

@tasks.loop(hours=24)
async def daily_summary():
    # Query database for daily statistics
    # Generate summary embed
    # Send to channel
    channel = bot.get_channel(CHANNEL_ID)
    await channel.send(embed=summary_embed)

@bot.event
async def on_ready():
    daily_summary.start()  # Start scheduled task
```

### The Collaborative Sensing Concept

**Instead of:** Configuring cameras once via SSH and letting them run passively

**Think:** Having a live conversation with your cameras:
- "Hey Camera 1, can you watch the door instead?"
- "All cameras, switch to high sensitivity mode"
- "Camera 2, what do you see right now?"
- "Everyone, let me know if you detect any cars"

This makes experimentation **fast, collaborative, and interactive** - perfect for learning and iterating on smart object systems.

## Additional Resources

### Documentation (This Repository)
- Student guide: `README.md` (for using pre-configured Pis)
- Instructor setup guide: `INITIAL_SETUP.md` (for setting up new Pis from scratch)
- Discord bot setup: `DISCORD_BOT_PLAN.md` (step-by-step bot configuration)
- Quick reference: `CHEATSHEET.md`
- VS Code setup: `README.md` (Appendix A)
- Discord webhooks: `README.md` (Appendix B)

### DepthAI 3.x Resources
- **DepthAI 3.x Documentation**: https://docs.luxonis.com/software-v3/depthai/
- **depthai-nodes GitHub**: https://github.com/luxonis/depthai-nodes (high-level parsing nodes)
- **Luxonis Hub Models**: https://models.luxonis.com (browse available models)
- **OAK Examples & Experiments**: https://github.com/luxonis/depthai-experiments
- **depthai-python Examples**: https://github.com/luxonis/depthai-python/tree/main/examples
- **API Reference**: https://docs.luxonis.com/api/

### Other Resources
- **Raspberry Pi Documentation**: https://www.raspberrypi.com/documentation/
- **VS Code Remote SSH**: https://code.visualstudio.com/docs/remote/ssh
- **discord.py Documentation**: https://discordpy.readthedocs.io/
