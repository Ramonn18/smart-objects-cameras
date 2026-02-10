# Integrating Discord Bot with Person Detector

This guide shows how to connect the Discord bot with the person detector so you can query camera status and receive alerts.

---

## Simple Integration (Recommended for Beginners)

This method uses a shared JSON file for communication.

### Step 1: Modify person_detector.py

Add these lines at the **top** of `person_detector.py`:

```python
import json
from pathlib import Path

# Status file for Discord bot
STATUS_FILE = Path.home() / "oak-projects" / "camera_status.json"
```

### Step 2: Update the detection_callback function

Modify the `detection_callback` function to write status updates:

```python
def detection_callback(packet: DetectionPacket):
    """Handle detection events."""
    global last_status, last_count

    # Get number of detected people
    person_count = len(packet.detections)
    person_detected = person_count > 0

    # UPDATE: Write status to file for Discord bot
    try:
        STATUS_FILE.write_text(json.dumps({
            "detected": person_detected,
            "count": person_count,
            "timestamp": datetime.now().isoformat(),
            "running": True
        }))
    except Exception as e:
        print(f"Warning: Could not update status file: {e}")

    # Log status changes (only on presence/absence transition)
    if person_detected != last_status:
        # ... rest of your existing code
```

### Step 3: Test the Integration

**Terminal 1 - Run the person detector:**
```bash
cd ~/oak-projects
source /opt/oak-shared/venv/bin/activate
python3 person_detector.py
```

**Terminal 2 - Run the Discord bot:**
```bash
cd ~/oak-projects
source /opt/oak-shared/venv/bin/activate
python3 discord_bot.py
```

**Discord - Test commands:**
- `!status` → Should show "Camera is ONLINE"
- `!detect` → Should show current detection status
- Wave in front of camera → Use `!detect` again to see it update

---

## Advanced Integration (Real-time Alerts)

This method allows the bot to send messages immediately when detections occur.

### Option A: Run Both in One Script

Create a new file `camera_with_bot.py`:

```python
#!/usr/bin/env python3
"""
Person detector with integrated Discord bot.
Runs both the camera detection and Discord bot in the same process.
"""

import asyncio
import discord
from discord.ext import commands
from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import DetectionPacket
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Discord bot setup
BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Global state
current_detection = {"detected": False, "count": 0, "timestamp": None}


# --- Discord Bot Commands ---

@bot.command(name='ping')
async def ping(ctx):
    await ctx.send('🏓 Pong!')


@bot.command(name='detect')
async def detect(ctx):
    if current_detection["detected"]:
        await ctx.send(f"🟢 {current_detection['count']} person(s) detected")
    else:
        await ctx.send("⚪ No person detected")


# --- Camera Detection ---

def detection_callback(packet: DetectionPacket):
    """Handle detection events - runs in camera thread."""
    global current_detection

    person_count = len(packet.detections)
    person_detected = person_count > 0

    # Update global state
    old_status = current_detection["detected"]
    current_detection = {
        "detected": person_detected,
        "count": person_count,
        "timestamp": datetime.now().isoformat()
    }

    # Send Discord alert on status change
    if person_detected != old_status:
        message = f"🟢 Person detected! (Count: {person_count})" if person_detected else "⚪ Area clear"
        # Schedule sending message to Discord
        asyncio.create_task(send_to_discord(message))


async def send_to_discord(message: str):
    """Send message to Discord channels."""
    for guild in bot.guilds:
        for channel in guild.text_channels:
            if channel.permissions_for(guild.me).send_messages:
                try:
                    await channel.send(message)
                    break
                except:
                    continue


async def run_camera():
    """Run camera detection in async context."""
    with OakCamera() as oak:
        color = oak.create_camera('color')
        nn = oak.create_nn('person-detection-retail-0013', color)
        oak.callback(nn, callback=detection_callback)
        oak.start(blocking=True)


async def main():
    """Main async function that runs both bot and camera."""
    # Start Discord bot in background
    bot_task = asyncio.create_task(bot.start(BOT_TOKEN))

    # Wait for bot to be ready
    await bot.wait_until_ready()
    print(f"Bot ready: {bot.user.name}")

    # Start camera (runs until stopped)
    try:
        await run_camera()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await bot.close()


if __name__ == '__main__':
    asyncio.run(main())
```

### Option B: Use Background Tasks

Modify `discord_bot.py` to periodically check the status file and send alerts:

```python
from discord.ext import tasks

@tasks.loop(seconds=2)
async def check_detection_changes():
    """Background task that checks for detection changes."""
    # Read status file
    # Compare with last known status
    # Send alert if changed
    pass

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')
    check_detection_changes.start()  # Start background task
```

---

## Running Both Services

### Option 1: Separate Terminals (Development)

**Terminal 1:**
```bash
python3 person_detector.py
```

**Terminal 2:**
```bash
python3 discord_bot.py
```

### Option 2: Systemd Services (Production)

**person-detector.service** (already exists)
```ini
[Service]
ExecStart=/opt/oak-shared/venv/bin/python3 /home/carrie/oak-projects/person_detector.py --log
```

**discord-bot.service** (new)
```ini
[Service]
ExecStart=/opt/oak-shared/venv/bin/python3 /home/carrie/oak-projects/discord_bot.py
```

Start both:
```bash
sudo systemctl start person-detector
sudo systemctl start discord-bot
```

### Option 3: Single Service (Advanced)

Use the `camera_with_bot.py` script:

```bash
sudo systemctl stop person-detector  # Stop old service
sudo systemctl stop discord-bot

# Create new combined service
sudo nano /etc/systemd/system/camera-bot.service
```

```ini
[Unit]
Description=OAK-D Camera with Discord Bot
After=network.target

[Service]
Type=simple
User=carrie
WorkingDirectory=/home/carrie/oak-projects
EnvironmentFile=/home/carrie/oak-projects/.env
ExecStart=/opt/oak-shared/venv/bin/python3 /home/carrie/oak-projects/camera_with_bot.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable camera-bot
sudo systemctl start camera-bot
```

---

## Testing the Integration

### Test 1: Status File Method
1. Start person detector
2. Check that `~/oak-projects/camera_status.json` exists
3. Cat the file: `cat ~/oak-projects/camera_status.json`
4. Start Discord bot
5. Use `!status` in Discord - should show "ONLINE"

### Test 2: Real-time Alerts
1. Start both person detector and bot
2. Wave in front of camera
3. Should see alert in Discord
4. Walk away
5. Should see "area clear" alert

### Test 3: Commands
- `!ping` → "Pong!"
- `!status` → "Camera is ONLINE"
- `!detect` → Current detection status
- `!help` → List of commands

---

## Troubleshooting

### Bot shows "Camera OFFLINE"
- Check person_detector.py is running: `ps aux | grep person_detector`
- Check status file exists: `ls ~/oak-projects/camera_status.json`
- Check file permissions: `ls -la ~/oak-projects/camera_status.json`

### Bot doesn't respond to commands
- Check bot is online in Discord (green dot)
- Make sure you're using `!` prefix: `!status` not `status`
- Check bot logs: `journalctl -u discord-bot -f`
- Verify Message Content Intent is enabled in Discord Developer Portal

### Detection updates are delayed
- The simple file-based method has ~1-2 second delay
- For real-time, use the advanced integration option
- Make sure status file is being updated: `watch -n 1 cat ~/oak-projects/camera_status.json`

---

## Security Notes

✅ Never commit bot token to git
✅ Use `chmod 600 .env` to protect credentials
✅ Consider limiting bot commands to specific roles/users
✅ Don't give bot admin permissions in Discord

---

## Next Steps

Once basic integration works:

1. Add more commands (`!threshold`, `!snapshot`)
2. Add role-based permissions
3. Create embeds for prettier messages
4. Add reaction-based controls
5. Support multiple cameras

Happy coding! 🤖📸
