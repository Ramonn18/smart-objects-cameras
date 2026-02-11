# Discord Integration

This guide covers setting up Discord integration for real-time person detection alerts and interactive camera control.

---

## Overview: Two Options

### Option 1: Webhooks (Simpler, One-Way)
- Real-time notifications in Discord when people are detected
- Alerts when the area becomes clear
- Timestamped messages showing detection events
- No complex bot hosting required
- **Best for:** Simple notifications only

### Option 2: Discord Bot (Advanced, Two-Way)
- Everything webhooks can do, plus:
- Interactive commands (!status, !detect, !ping, !screenshot)
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

### Prerequisites

- A Discord account
- A Discord server where you have "Manage Webhooks" permission
  - You can create your own server for free if needed

---

## Step 1: Create a Discord Server (If Needed)

If you already have a Discord server where you want notifications, skip to Step 2.

### Create Your Own Server

1. Open Discord (desktop app or web)
2. Click the **+** button in the left sidebar
3. Select **Create My Own**
4. Choose **For me and my friends** (or customize as needed)
5. Name it something like "Camera Notifications" or "OAK-D Monitor"
6. Click **Create**

---

## Step 2: Create a Webhook

A webhook is a special URL that allows the camera to send messages to Discord without running a full bot.

### Step-by-Step Instructions

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

## Step 3: Configure the Camera System

Now we'll tell the camera where to send notifications.

### On Your Raspberry Pi

1. **SSH into your Pi**

   ```bash
   ssh orbit
   # or
   ssh gravity
   # or
   ssh horizon
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

### Install Required Dependencies

```bash
# Activate shared virtual environment
activate-oak

# Install required packages (if not already installed during initial setup)
pip install requests aiohttp python-dotenv
```

---

## Step 4: Test the Webhook

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

### Manual Test

You can also send custom messages:

```bash
python3 discord_notifier.py "Hello from the camera system!"
```

---

## Step 5: Run the Camera with Discord Notifications

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

## Step 6: Auto-Start with Notifications (Optional)

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

### Reload and restart the service

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

## Discord Customization

### Change Notification Settings

Edit `person_detector.py` to customize what gets sent to Discord:

```python
# Example: Only notify on person detection (not when area clears)
if person_detected and not last_status:
    send_notification(f"🟢 PERSON DETECTED (count: {person_count})")
    # Remove the "no person" notification
```

### Change Username/Icon

Modify calls in `person_detector.py`:

```python
send_notification(
    "🟢 PERSON DETECTED",
    username="Security Camera 1"
)
```

### Disable Timestamps

```python
send_notification("Message here", add_timestamp=False)
```

---

## Troubleshooting

### "DISCORD_WEBHOOK_URL not set" Error

**Problem**: The `.env` file isn't being loaded or doesn't exist.

**Solutions**:

1. Make sure you created `.env` in `~/oak-projects/`
2. Check the file contents: `cat ~/oak-projects/.env`
3. Ensure `python-dotenv` is installed: `pip install python-dotenv`

### "Notification failed: 404" Error

**Problem**: The webhook URL is incorrect or the webhook was deleted.

**Solutions**:

1. Go back to Discord → Server Settings → Integrations → Webhooks
2. Verify the webhook still exists
3. Copy the URL again and update `.env`
4. Make sure there are no extra spaces in the `.env` file

### "Notification timed out" Error

**Problem**: Network connectivity issue between Pi and Discord.

**Solutions**:

1. Check internet connection: `ping discord.com`
2. Check Pi's network settings
3. Try the test script again: `python3 discord_notifier.py`

### Messages Not Appearing in Discord

**Checklist**:

- [ ] Webhook exists in Discord settings
- [ ] Webhook points to correct channel
- [ ] `.env` file has correct URL
- [ ] Test script works: `python3 discord_notifier.py`
- [ ] Virtual environment is activated
- [ ] `requests` and `python-dotenv` are installed

### Permission Issues with .env File

```bash
# Fix file permissions
chmod 600 ~/oak-projects/.env

# Verify ownership
ls -la ~/oak-projects/.env
# Should show: -rw------- 1 [user] [user] ...
```

---

## Security Best Practices

### Keep Your Webhook Secret

- ✅ **DO**: Store webhook URL in `.env` file with restricted permissions
- ✅ **DO**: Add `.env` to `.gitignore` if using version control
- ❌ **DON'T**: Share webhook URL publicly
- ❌ **DON'T**: Commit `.env` to GitHub/Git repositories
- ❌ **DON'T**: Post webhook URL in Discord or other public places

### Regenerate Webhook If Leaked

If your webhook URL is accidentally exposed:

1. Go to Discord → Server Settings → Integrations → Webhooks
2. Find your webhook
3. Click the webhook name
4. Scroll to bottom and click **Delete Webhook**
5. Create a new webhook (follow Step 2 again)
6. Update `.env` with new URL

---

## Using Multiple Cameras

If you have multiple cameras (multiple Pis), you can:

### Option 1: Different Webhooks (Different Channels)

Create separate webhooks for each camera:

```bash
# On Pi 1
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/camera1_webhook

# On Pi 2
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/camera2_webhook
```

### Option 2: Same Webhook (Same Channel) with Different Names

Use the same webhook but different usernames:

```python
# On Pi 1
send_notification("Person detected", username="Camera 1 - Front Door")

# On Pi 2
send_notification("Person detected", username="Camera 2 - Backyard")
```

---

## Interactive Bot Setup

For two-way interaction with commands like `!status`, `!screenshot`, and `!detect`, see the complete bot setup guide:

**[DISCORD_BOT_PLAN.md](DISCORD_BOT_PLAN.md)** - Step-by-step Discord bot configuration (30-45 minutes)

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

## Related Documentation

- [README.md](../README.md) - Main documentation
- [DISCORD_BOT_PLAN.md](DISCORD_BOT_PLAN.md) - Complete bot setup guide
- [Multi-User Access](archive/multi-user-access.md) - Collaborative workflows
