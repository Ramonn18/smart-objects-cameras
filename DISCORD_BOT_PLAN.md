# Discord Bot Setup Plan

## Overview

This plan will help you create a **simple Discord bot** that can interact with your OAK-D camera system. Unlike webhooks (one-way notifications), a bot can respond to commands and provide interactive features.

**Time estimate:** 30-45 minutes

---

## What the Bot Will Do (Minimal Version)

✅ **Commands:**

- `!status` - Check if camera is running
- `!detect` - Get current detection status
- `!screenshot` - Get a live image from camera
- `!ping` - Test if bot is alive

✅ **Auto-notifications:** Still send alerts when people are detected (like webhook)

---

## Step 1: Create a Discord Bot (10 minutes)

### 1.1 Go to Discord Developer Portal

Visit: https://discord.com/developers/applications

### 1.2 Create New Application

1. Click **"New Application"**
2. Name it: `{Character}Bot`
3. Click **"Create"**

### 1.3 Create the Bot User

1. In left sidebar, click **"Bot"**
2. Click **"Add Bot"** → **"Yes, do it!"**
3. Under **"Token"**, click **"Reset Token"** → **"Yes, do it!"**
4. **Copy the token** and save it somewhere safe (you'll need this later)
   - ⚠️ **Never share this token publicly!**

### 1.4 Configure Bot Permissions

Still in the Bot page:

1. Scroll to **"Privileged Gateway Intents"**
2. Enable:
   - ✅ **Message Content Intent**
3. Click **"Save Changes"**

### 1.5 Invite Bot to Your Server

1. In left sidebar, click **"OAuth2"** → **"URL Generator"**
2. Under **"Scopes"**, check:
   - ✅ `bot`
3. Under **"Bot Permissions"**, check:
   - ✅ `Send Messages`
   - ✅ `Attach Files` (needed for !screenshot command)
   - ✅ `Read Messages/View Channels`
   - ✅ `Read Message History`
4. Copy the generated URL at the bottom
5. Open that URL in your browser
6. Select your server and authorize

Your bot should now appear in your server (offline)!

---

## Step 2: Install Discord.py on the Pi (5 minutes)

SSH into your Raspberry Pi and install the Discord library:

```bash
# SSH into Pi
ssh smartobjects1.local

# Activate venv
activate-oak

# Install discord.py
pip install discord.py

# Verify installation
python3 -c "import discord; print(f'discord.py version: {discord.__version__}')"
```

---

## Step 3: Set Up Environment Variables (5 minutes)

Add your bot token to the `.env` file:

```bash
cd ~/oak-projects
nano .env
```

Add this line (replace with your actual token):

```bash
# Discord Bot Token
DISCORD_BOT_TOKEN=YOUR_BOT_TOKEN_HERE

# Keep your webhook URL too (optional)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_URL
```

Save: `Ctrl+O`, `Enter`, `Ctrl+X`

Secure the file:

```bash
chmod 600 .env
```

---

## Step 4: Create the Bot Script (5 minutes)

I'll provide a simple bot template in `discord_bot.py`. You can copy it to your Pi:

**On your Pi:**

```bash
cd ~/oak-projects
nano discord_bot.py
```

Paste the bot code (see `discord_bot.py` template).

Save and exit: `Ctrl+O`, `Enter`, `Ctrl+X`

Make it executable:

```bash
chmod +x discord_bot.py
```

---

## Step 5: Test the Bot (5 minutes)

```bash
# Activate venv
activate-oak

# Run the bot
python3 discord_bot.py
```

**Expected output:**

```
Logged in as OAK-D Camera Bot#1234
Bot is ready!
```

**Test commands in Discord:**

- Type `!ping` → Bot should respond "Pong!"
- Type `!status` → Bot should show camera status
- Type `!detect` → Bot should show detection info
- Type `!screenshot` → Bot should send a camera image 📸
- Type `!help` → Bot should list available commands

Press `Ctrl+C` to stop the bot.

**Note:** The `!screenshot` command requires `person_detector.py` to be running, as it saves frames every 5 seconds to `~/oak-projects/latest_frame.jpg`.

---

## Step 6: Integrate with Person Detector (10 minutes)

### Option A: Simple Integration (Recommended)

Modify `person_detector.py` to send updates to the bot via a shared file:

**In person_detector.py**, add at the top:

```python
import json
from pathlib import Path

# Status file for bot
STATUS_FILE = Path.home() / "oak-projects" / "camera_status.json"
```

**In the detection callback**, update status:

```python
def detection_callback(packet: DetectionPacket):
    global last_status, last_count

    person_count = len(packet.detections)
    person_detected = person_count > 0

    # Update status file for bot
    STATUS_FILE.write_text(json.dumps({
        "detected": person_detected,
        "count": person_count,
        "timestamp": datetime.now().isoformat()
    }))

    # ... rest of your callback code
```

The bot will read this file when you use `!status` or `!detect` commands.

**For screenshot functionality**, `person_detector.py` also saves camera frames automatically:

```python
# Screenshot for Discord bot
SCREENSHOT_FILE = Path.home() / "oak-projects" / "latest_frame.jpg"
SCREENSHOT_UPDATE_INTERVAL = 5  # Save screenshot every 5 seconds

# In detection loop:
if preview_frame is not None and current_time - last_screenshot_time >= SCREENSHOT_UPDATE_INTERVAL:
    frame = preview_frame.getCvFrame()
    cv2.imwrite(str(SCREENSHOT_FILE), frame)
    last_screenshot_time = current_time
```

The bot's `!screenshot` command reads this file and sends it to Discord. Frames are captured every 5 seconds automatically, so screenshots are always recent.

### Option B: Advanced Integration (Optional)

Use Python's multiprocessing to run the bot and detector in the same script. This is more complex but allows real-time updates.

---

## Step 7: Auto-Start the Bot (Optional)

Create a systemd service for the bot:

```bash
sudo nano /etc/systemd/system/discord-bot.service
```

Paste:

```ini
[Unit]
Description=Discord Bot for OAK-D Camera
After=network.target

[Service]
Type=simple
User=carrie
WorkingDirectory=/home/carrie/oak-projects
EnvironmentFile=/home/carrie/oak-projects/.env
ExecStart=/opt/oak-shared/venv/bin/python3 /home/carrie/oak-projects/discord_bot.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable discord-bot
sudo systemctl start discord-bot
sudo systemctl status discord-bot
```

---

## Architecture Comparison

### Webhooks (Current):

```
Camera detects person → Python script → POST to webhook → Discord shows message
```

✅ Simple, one-way notifications
❌ No interaction, can't query status

### Bot (New):

```
Camera detects person → Python script → Bot sends message → Discord
User types !status → Discord → Bot reads status → Responds to user
```

✅ Two-way interaction
✅ Can query status, run commands
❌ Slightly more complex setup

---

## Minimal Bot Commands

Here's what the simplest version should support:

| Command       | Description          | Example Response                |
| ------------- | -------------------- | ------------------------------- |
| `!ping`       | Test if bot is alive | "Pong! 🏓"                      |
| `!status`     | Camera status        | "✅ Camera online"              |
| `!detect`     | Current detection    | "🟢 2 people detected"          |
| `!screenshot` | Get camera image     | Sends live photo from camera 📸 |
| `!help`       | List commands        | Shows all commands              |

---

## Optional Advanced Features (For Later)

Once basic bot works, you could add:

- **`!set-threshold <value>`** - Adjust detection threshold remotely (without restart)
- **`!record <seconds>`** - Capture video clip on demand
- **`!stats`** - Show detection statistics from database
- **`!daily-summary`** - Get automated daily reports
- **Discord embeds** - Rich formatted status cards with thumbnails
- **Role-based permissions** - Only admins can control camera
- **Multiple cameras** - `!camera1 status`, `!camera2 screenshot`, `!all-cameras detect-person`

---

## Troubleshooting

### Bot won't connect

- Check token is correct in `.env`
- Make sure Message Content Intent is enabled
- Verify bot has permissions in server

### Bot doesn't respond to commands

- Check bot is online in Discord
- Make sure you're using correct command prefix (`!`)
- Check bot logs: `journalctl -u discord-bot -f`

### "Module discord not found"

- Make sure venv is activated
- Reinstall: `pip install discord.py`

---

## Security Best Practices

✅ **DO:**

- Keep bot token in `.env` file
- Use `chmod 600 .env` to restrict access
- Add `.env` to `.gitignore`

❌ **DON'T:**

- Share bot token publicly
- Commit `.env` to git
- Give bot more permissions than needed

---

## Next Steps After Basic Bot Works

1. ✅ Test all commands work
2. ✅ Integrate with person_detector.py
3. ✅ Set up auto-start (optional)
4. 🎨 Customize bot responses
5. 🚀 Add advanced features (if desired)

---

## Estimated Timeline

- **Step 1-2:** Bot creation and setup - 15 minutes
- **Step 3-4:** Code and configuration - 10 minutes
- **Step 5:** Testing - 5 minutes
- **Step 6:** Integration - 10 minutes
- **Step 7:** Auto-start (optional) - 5 minutes

**Total:** ~30-45 minutes for basic working bot

---

## Resources

- **Discord.py Documentation**: https://discordpy.readthedocs.io/
- **Discord Developer Portal**: https://discord.com/developers/applications
- **Discord.py Examples**: https://github.com/Rapptz/discord.py/tree/master/examples

---

Good luck! Let me know if you get stuck on any step. 🤖
