# OAK-D + Raspberry Pi 5 — Quick Start Cheat Sheet

## 🔌 Network Info

| Pi | Hostname | IP Lookup | Access |
|----|----------|-----------|--------|
| 16GB | `smartobjects1` | `ping smartobjects1.local` | SSH (key-based) + VNC |
| 8GB | `smartobjects2` | `ping smartobjects2.local` | SSH (key-based) only |

**Authentication:** SSH key-based (no password needed if configured)

---

## 🖥️ Connect to Pi

### SSH (Terminal Access)
```bash
ssh smartobjects1.local        # 16GB Pi
ssh smartobjects2.local        # 8GB Pi
```

**Note:** No password needed with SSH keys configured!

### VNC (Desktop Access — 16GB Pi only)
1. Open RealVNC Viewer
2. Connect to: `smartobjects1.local`
3. Login with your username and password

### VS Code Remote (Recommended for coding)
1. **Mac users:** Grant VS Code "Local Network" permission first!
   - System Settings → Privacy & Security → Local Network → VS Code ✅
2. `Ctrl+Shift+P` → "Remote-SSH: Connect to Host"
3. Enter: `smartobjects1.local` or `smartobjects2.local`
4. Open folder: `/home/[username]/oak-projects`

---

## 🐍 Python Environment

**Always activate the shared virtual environment first:**

```bash
cd ~/oak-projects
activate-oak
```

Your prompt should change to show `(venv)`:
```
(venv) carrie@smartobjects1:~/oak-projects $
```

---

## 📷 Quick Camera Test

```bash
# Check if camera is connected
python3 -c "import depthai as dai; devices = dai.Device.getAllAvailableDevices(); print(f'Found {len(devices)} camera(s)')"
```

**Expected output:**
```
Found 1 camera(s)
```

**No output or error?** Check USB connection, try different port, or use powered hub.

---

## 🏃 Run Person Detector

```bash
# Basic (console output only)
python3 person_detector.py

# With video display (VNC Pi only)
python3 person_detector.py --display

# With logging to file
python3 person_detector.py --log

# Adjust sensitivity (0.0 - 1.0, default 0.5)
python3 person_detector.py --threshold 0.7
```

**Stop with:** `Ctrl+C`

---

## 📁 Project Files

```
~/oak-projects/
├── venv/                    # Python virtual environment
├── person_detector.py       # Main detection script
└── person_detection_*.log   # Log files (if using --log)
```

---

## 🔧 Common Commands

| Task | Command |
|------|---------|
| Update system | `sudo apt update && sudo apt upgrade -y` |
| Check memory | `free -h` |
| Check CPU/processes | `htop` |
| Check camera | `python3 -c "import depthai as dai; devices = dai.Device.getAllAvailableDevices(); print(f'Found {len(devices)} camera(s)')"` |
| List USB devices | `lsusb` |
| Reboot | `sudo reboot` |
| Shutdown | `sudo shutdown -h now` |

---

## 🚨 Troubleshooting

| Problem | Solution |
|---------|----------|
| VS Code "No route to host" (Mac) | Grant Local Network permission: Settings → Privacy → Local Network → VS Code ✅ |
| "No module named depthai" | Run `activate-oak` (alias for `source /opt/oak-shared/venv/bin/activate`) |
| Camera not found | Check USB, try powered hub |
| Permission denied | `sudo udevadm control --reload-rules && sudo udevadm trigger` |
| VNC black screen | Set resolution in `raspi-config` → Display Options |
| Can't ping Pi | Check network, try IP address instead of `.local` |
| Script won't start | Check logs: `journalctl -u person-detector -n 20` |

---

## ⌨️ Keyboard Shortcuts

### Terminal (SSH)
| Action | Keys |
|--------|------|
| Cancel running program | `Ctrl+C` |
| Clear screen | `Ctrl+L` or `clear` |
| Previous command | `↑` arrow |
| Search history | `Ctrl+R` |
| Logout | `exit` or `Ctrl+D` |

### VS Code
| Action | Keys |
|--------|------|
| Command Palette | `Ctrl+Shift+P` |
| Open Terminal | `` Ctrl+` `` |
| Run Code | `F5` |
| Save | `Ctrl+S` |
| Find | `Ctrl+F` |

---

## 📝 Making Changes

1. **Edit with VS Code** (recommended):
   - Connect via Remote SSH
   - Edit `person_detector.py`
   - Save (`Ctrl+S`)
   - Run in terminal

2. **Edit on Pi directly**:
   ```bash
   nano person_detector.py    # Edit
   # Ctrl+O, Enter to save
   # Ctrl+X to exit
   ```

3. **Create your own copy first**:
   ```bash
   cp person_detector.py person_detector_myname.py
   nano person_detector_myname.py
   ```

---

## 🌐 WiFi Network Management

### Switch Between Networks (Home ↔ Class)

Modern Raspberry Pi OS uses **NetworkManager**. Here's the easiest way:

**Method 1 — nmtui (BEST for pre-configuring classroom WiFi!):**
```bash
# Open menu interface
sudo nmtui

# Select "Edit a connection" → "Add" → "Wi-Fi"
# Enter SSID and password for classroom
# Save and quit
# Pi will auto-connect when in range!
```

**Method 2 — nmcli (requires network in range):**
```bash
# List available networks
nmcli device wifi list

# Connect to a network (saves automatically)
sudo nmcli device wifi connect "ClassroomWiFi" password "password"

# See saved networks
nmcli connection show

# Switch to a saved network
nmcli connection up "HomeWiFi"
```

**Method 3 — raspi-config:**
```bash
sudo raspi-config
# System Options → Wireless LAN → Enter new SSID/password
```

---

## 👥 Multi-User Access

### Add Your SSH Key
```bash
# From your laptop
ssh-copy-id smartobjects1.local
```

### Multiple People, Same Pi
- Each person adds their SSH key (doesn't overwrite others)
- Create personal script copies to avoid conflicts:
  ```bash
  cp person_detector.py person_detector_yourname.py
  ```
- Coordinate with teammates before editing shared files

---

## 🔗 Useful Links

- DepthAI SDK v3 Docs: https://docs.luxonis.com/software-v3/depthai/
- OAK Examples (v3): https://github.com/luxonis/depthai-experiments
- VS Code Remote: https://code.visualstudio.com/docs/remote/ssh
- Raspberry Pi Docs: https://www.raspberrypi.com/documentation/

---

## ❓ Getting Help

1. Check the full `README.md` for detailed instructions
2. Check `README.md` Appendix A for VS Code Remote setup
3. Check `README.md` Appendix B for Discord notifications
4. Ask your instructor
5. Search Luxonis Discord / GitHub issues

---

*Print this page and keep it at your workstation!*
