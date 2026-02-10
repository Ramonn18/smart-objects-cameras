# Student Quick Start Guide - OAK-D Camera

## Getting Started

### 1. Log In
- **Physical Monitor:** Log in with your username at the connected monitor
- **VNC:** Use VNC Viewer to connect to `orbit` (or `gravity`, `horizon`)
  - Username: your assigned username (e.g., `juju`, `akentou`)
  - Password: `oak2026`

> **Note:** Only one person can use VNC/monitor at a time. If someone else is logged in, you'll need to wait or use SSH.

### 2. Activate the Virtual Environment

Open Terminal and type:
```bash
activate-oak
```

You should see `(venv)` appear before your prompt. This means you're ready!

### 3. Navigate to Your Project Folder

```bash
cd ~/oak-projects
```

---

## Running Person Detection

### With Visual Display (for development/testing)

```bash
activate-oak
cd ~/oak-projects
python3 person_detector_with_display.py --display
```

**What you'll see:**
- A window showing live camera feed
- Green boxes around detected people
- Detection count and confidence threshold
- Press **'q'** to quit (or Ctrl+C in terminal)

### Without Display (for production/monitoring)

```bash
activate-oak
cd ~/oak-projects
python3 person_detector_with_display.py
```

**What happens:**
- Runs in the background
- Saves screenshots every 5 seconds to `latest_frame.jpg`
- Use Discord bot to view: `!screenshot`
- Press **Ctrl+C** to stop

### With Discord Notifications

```bash
python3 person_detector_with_display.py --discord
```

**What it does:**
- Announces when you start/stop the detector
- Sends notifications when people are detected/cleared
- Works great with `!screenshot` command

---

## Discord Bot Commands

The Discord bot can show you what the camera sees:

- `!ping` - Check if bot is alive
- `!status` - Check if camera is running
- `!screenshot` - Get current camera image with detection boxes
- `!detect` - Get current detection status
- `!help` - Show all commands

---

## Common Commands

### Check if Camera is Connected

```bash
activate-oak
python3 -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
```

You should see your camera's serial number.

### Test GUI Support

```bash
activate-oak
python3 -c "import cv2; cv2.namedWindow('test'); cv2.destroyWindow('test'); print('GUI works')"
```

Should print: `GUI works`

### View Your Files

```bash
ls ~/oak-projects
```

You should see:
- `person_detector_with_display.py` - Your main script
- `discord_notifier.py` - Discord integration
- `latest_frame.jpg` - Most recent screenshot
- `camera_status.json` - Current detection status

---

## Troubleshooting

### "Command not found: activate-oak"

You need to reload your bash configuration:
```bash
source ~/.bashrc
```

Then try `activate-oak` again.

### "No module named 'depthai'"

Make sure the virtual environment is activated:
```bash
activate-oak
```

You should see `(venv)` in your prompt.

### "Permission denied" errors

You don't have sudo access. Ask the instructor for help.

### Camera not found

Check:
1. Is the USB cable plugged into a **blue** USB 3.0 port?
2. Is someone else using the camera right now?
3. Try unplugging and replugging the camera

### Black screen in display window

Wait a few seconds - frames take a moment to start flowing. If it stays black for more than 10 seconds, press Ctrl+C and try again.

### VNC shows grey screen or won't connect

Someone else is probably logged in at the monitor. Either:
- Wait for them to log out
- Use SSH instead: `ssh orbit` (if your SSH config is set up)
- Ask them to share the desktop

---

## Tips for Development

### Testing Changes Quickly

1. Make your code changes
2. Run with `--display` to see results live
3. Press 'q' to quit
4. Make more changes
5. Repeat!

### Running Headless

Once your code works:
1. Run without `--display` flag
2. Monitor via Discord (`!screenshot`)
3. Let it run in the background
4. Great for long-term deployments

### Adjusting Detection Sensitivity

```bash
# Lower threshold = more sensitive (may have false positives)
python3 person_detector_with_display.py --display --threshold 0.3

# Higher threshold = less sensitive (fewer false positives)
python3 person_detector_with_display.py --display --threshold 0.7

# Default is 0.5
```

### Logging to File

```bash
python3 person_detector_with_display.py --display --log
```

Creates a timestamped log file like `person_detection_20260204_151030.log`

---

## Your Project Directory

Everything you need is in `~/oak-projects/`:

```
~/oak-projects/
├── person_detector_with_display.py  # Main detection script
├── discord_notifier.py              # Discord webhook helper
├── .env                             # Environment variables (DO NOT commit to git)
├── camera_status.json               # Detection status (auto-generated)
└── latest_frame.jpg                 # Screenshot (auto-generated)
```

**Note:** The actual Python packages are in `/opt/oak-shared/venv/` (shared by everyone).

---

## Need Help?

1. Check the main README: `~/oak-projects/README.md`
2. Check version compatibility: `~/oak-projects/WORKING_VERSIONS.md`
3. Ask your instructor
4. Check Luxonis docs: https://docs.luxonis.com

---

## Working Versions (for reference)

Your system uses these specific versions (tested and working):
- Python 3.13.5
- depthai 3.3.0
- opencv-contrib-python 4.10.0.84
- numpy 1.26.4

See `WORKING_VERSIONS.md` for complete list.
