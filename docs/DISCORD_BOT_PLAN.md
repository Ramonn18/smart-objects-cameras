# Discord Bot Setup Plan

## Overview

This plan will help you create a **simple Discord bot** that can interact with your OAK-D camera system. Unlike webhooks (one-way notifications), a bot can respond to commands and provide interactive features.

**Time estimate:** 30-45 minutes

---

## Bot Architecture: Camera Bots vs Personal Bots

**Important:** There are TWO types of bots in this system:

### 1. Camera Bots (Shared - Post to Main Channel)

Each camera has its own dedicated bot that posts to the **main Smart Objects channel**:

- **Orbit Bot** - Runs on the Orbit Pi, represents Camera 1
- **Gravity Bot** - Runs on the Gravity Pi, represents Camera 2
- **Horizon Bot** - Runs on the Horizon Pi, represents Camera 3

**These bots:**
- Post detection alerts to the shared Smart Objects channel
- Respond to commands like `!status`, `!screenshot`, `!detect`
- Allow anyone in the channel to query camera status
- Are always running on their respective Pis

**Setup:** Follow this guide to create camera bots (use names like "OrbitBot", "GravityBot", "HorizonBot")

---

### 2. Personal Bots (Private - For Individual Use)

Each student also created their own **private bot** for personal experiments:

**These bots:**
- Can send you direct messages (DMs)
- Used with `discord_dm_notifier.py` for private notifications
- Great for testing without spamming the class channel
- Optional - only use if you want private alerts

**Setup:** You've already created these! They use a different token in your personal `.env` file.

**Which bot should I set up?**
- **Setting up a camera bot?** Follow this guide (post to Smart Objects channel)
- **Using your personal bot?** Use `discord_dm_notifier.py` instead

---

## What the Bot Will Do (Minimal Version)

✅ **Commands:**

- `!status` - Check if camera is running
- `!detect` - Get current detection status
- `!screenshot` - Get a live image from camera
- `!ping` - Test if bot is alive

✅ **@Mention Commands:** Talk to cameras directly!

- `@OrbitBot status` - Ask a specific camera
- `@GravityBot what do you see?` - Natural language queries
- `@HorizonBot screenshot` - Direct commands to individual cameras

✅ **Multi-Camera Coordination:**

- `!all-cameras status` - Query all cameras at once
- `!orbit status` - Command specific camera by name
- `!compare-cameras` - See all camera states side-by-side

✅ **Auto-notifications:** Still send alerts when people are detected (like webhook)

---

## Step 1: Create a Discord Bot (10 minutes)

### 1.1 Go to Discord Developer Portal

Visit: https://discord.com/developers/applications

### 1.2 Create New Application

1. Click **"New Application"**
2. Name it after your camera:
   - `OrbitBot` (for Orbit Pi)
   - `GravityBot` (for Gravity Pi)
   - `HorizonBot` (for Horizon Pi)
3. Click **"Create"**

**Note:** This creates a **camera bot** that will post to the main Smart Objects channel. If you're setting up your personal bot instead, you've already done this step!

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

### 1.5 Invite Bot to the Smart Objects Server

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
6. Select the **Smart Objects server** and authorize

Your camera bot should now appear in the Smart Objects server (offline)!

**Note:** Camera bots (Orbit, Gravity, Horizon) all post to the main Smart Objects channel. Personal bots can DM you instead.

---

## Step 2: Install Discord.py on the Pi (5 minutes)

SSH into your Raspberry Pi and install the Discord library:

```bash
# SSH into Pi
ssh orbit  # or gravity, or horizon

# Activate venv
activate-oak

# Install discord.py
pip install discord.py

# Verify installation
python3 -c "import discord; print(f'discord.py version: {discord.__version__}')"
```

---

## Step 3: Set Up Environment Variables (5 minutes)

**⚠️ Important for Students:** The camera bot tokens (OrbitBot, GravityBot, HorizonBot) are **already configured** on the Raspberry Pis! This section is for instructors setting up new camera bots from scratch.

**Students:** You only need to add your personal DM bot token if you want private notifications. See [STUDENT_QUICKSTART.md](STUDENT_QUICKSTART.md) for student instructions.

---

### For Instructors: Setting Up Camera Bot Tokens

Add the **camera bot token** to the `.env` file on each Pi:

```bash
cd ~/oak-projects
nano .env
```

The complete `.env` file should contain:

```bash
# Discord Webhook (for person detection --discord notifications)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_URL_HERE

# Discord Bot Configuration (for public camera bot: OrbitBot, GravityBot, or HorizonBot)
DISCORD_APPLICATION_ID=your_application_id_here
DISCORD_PUBLIC_KEY=your_public_key_here
DISCORD_BOT_TOKEN=your_bot_token_here

# Students add their personal tokens below (optional):
# DISCORD_USER_ID=your_discord_user_id_here
# DISCORD_DM_BOT_TOKEN=your_dm_bot_token_here
```

**Token Key:**
- `DISCORD_BOT_TOKEN` = Camera bot (OrbitBot, GravityBot, or HorizonBot) - posts to Smart Objects channel
- `DISCORD_DM_BOT_TOKEN` = Personal student bot - sends private DMs (students add this themselves)
- `DISCORD_APPLICATION_ID` and `DISCORD_PUBLIC_KEY` = From Discord Developer Portal (Application → General Information)
- `DISCORD_USER_ID` = Student's Discord user ID (for DM bot targeting)

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
Logged in as OrbitBot#1234
Bot is ready!
```
(or "GravityBot" or "HorizonBot" depending on which camera you're setting up)

**Test commands in the Smart Objects Discord channel:**

- Type `!ping` → Bot should respond "Pong!"
- Type `!status` → Bot should show camera status
- Type `!detect` → Bot should show detection info
- Type `!screenshot` → Bot should send a camera image 📸
- Type `!help` → Bot should list available commands

**Important:** Make sure you're testing in the **Smart Objects channel** where the camera bot has access!

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

## Step 7: Add @Mention and Multi-Camera Commands (15 minutes)

Now let's make the cameras more conversational! We'll add three command patterns.

### Pattern 1: @Mention Commands (Direct Conversation)

Allow people to talk directly to cameras:

```python
@bot.event
async def on_message(message):
    # Don't respond to yourself
    if message.author == bot.user:
        return

    # Check if bot was mentioned
    if bot.user in message.mentions:
        content = message.content.lower()

        # Remove the mention to get the actual command
        command = content.replace(f'<@{bot.user.id}>', '').strip()

        # Handle natural language
        if 'status' in command or 'running' in command:
            await message.channel.send("🤖 I'm online and watching!")

        elif 'see' in command or 'detect' in command:
            # Read camera_status.json
            status = read_camera_status()
            await message.channel.send(f"👁️ I see {status['count']} people")

        elif 'screenshot' in command or 'picture' in command:
            # Send screenshot
            await message.channel.send(file=discord.File('latest_frame.jpg'))

        else:
            await message.channel.send("I can respond to: status, detect, screenshot")

    # Process normal ! commands
    await bot.process_commands(message)
```

**Usage:**
- `@OrbitBot status` → "🤖 I'm online and watching!"
- `@GravityBot what do you see?` → "👁️ I see 2 people"
- `@HorizonBot send a picture` → *sends screenshot*

---

### Pattern 2: Camera-Specific Commands

Allow commands to specific cameras even when all three bots are in the same channel:

```python
# In discord_bot.py, add these commands:

@bot.command(name='orbit')
async def orbit_command(ctx, *, cmd):
    """Handle !orbit <command>"""
    if ctx.guild and 'orbit' in bot.user.name.lower():
        # This IS the Orbit bot, respond
        await ctx.invoke(bot.get_command(cmd))
    # Otherwise ignore (different camera)

@bot.command(name='gravity')
async def gravity_command(ctx, *, cmd):
    """Handle !gravity <command>"""
    if ctx.guild and 'gravity' in bot.user.name.lower():
        await ctx.invoke(bot.get_command(cmd))

@bot.command(name='horizon')
async def horizon_command(ctx, *, cmd):
    """Handle !horizon <command>"""
    if ctx.guild and 'horizon' in bot.user.name.lower():
        await ctx.invoke(bot.get_command(cmd))
```

**Usage:**
- `!orbit status` → Only OrbitBot responds
- `!gravity screenshot` → Only GravityBot responds
- `!horizon detect` → Only HorizonBot responds

---

### Pattern 3: Multi-Camera Commands

Commands that all cameras respond to:

```python
@bot.command(name='all-cameras')
async def all_cameras(ctx, cmd):
    """Handle !all-cameras <command>"""
    # All bots respond to this
    bot_name = bot.user.name

    if cmd == 'status':
        status = read_camera_status()
        await ctx.send(f"**{bot_name}**: {'🟢 Active' if status['detected'] else '⚪ Idle'}")

    elif cmd == 'screenshot':
        await ctx.send(f"**{bot_name}** view:")
        await ctx.send(file=discord.File('latest_frame.jpg'))

    elif cmd == 'detect':
        status = read_camera_status()
        await ctx.send(f"**{bot_name}**: {status['count']} people detected")

@bot.command(name='compare-cameras')
async def compare_cameras(ctx):
    """Show side-by-side camera comparison"""
    # Each bot contributes its status
    status = read_camera_status()
    bot_name = bot.user.name

    embed = discord.Embed(title=bot_name, color=discord.Color.blue())
    embed.add_field(name="Status", value="🟢 Active" if status['detected'] else "⚪ Idle")
    embed.add_field(name="Count", value=str(status['count']))
    embed.add_field(name="Last Update", value=status.get('timestamp', 'Unknown'))

    await ctx.send(embed=embed)
```

**Usage:**
- `!all-cameras status` → All three bots respond with their status
- `!all-cameras screenshot` → Get screenshots from all three cameras
- `!compare-cameras` → All three send formatted status cards

---

### Complete Example Implementation

Here's how to add all three patterns to your `discord_bot.py`:

```python
import discord
from discord.ext import commands
import json
from pathlib import Path

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix='!', intents=intents)

def read_camera_status():
    """Read status from camera_status.json"""
    status_file = Path.home() / "oak-projects" / "camera_status.json"
    if status_file.exists():
        return json.loads(status_file.read_text())
    return {"detected": False, "count": 0, "timestamp": None}

# Pattern 1: @Mention handling
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Handle @mentions
    if bot.user in message.mentions:
        content = message.content.lower()
        command = content.replace(f'<@{bot.user.id}>', '').strip()

        if 'status' in command or 'running' in command:
            await message.channel.send(f"🤖 **{bot.user.name}**: I'm online and watching!")
        elif 'see' in command or 'detect' in command:
            status = read_camera_status()
            await message.channel.send(f"👁️ **{bot.user.name}**: I see {status['count']} people")
        elif 'screenshot' in command or 'picture' in command:
            screenshot_file = Path.home() / "oak-projects" / "latest_frame.jpg"
            if screenshot_file.exists():
                await message.channel.send(f"📸 **{bot.user.name}** view:",
                                          file=discord.File(screenshot_file))
            else:
                await message.channel.send("❌ No screenshot available (detector not running)")

    await bot.process_commands(message)

# Pattern 2: Camera-specific commands
@bot.command(name='orbit')
async def orbit_command(ctx, *, cmd):
    if 'orbit' in bot.user.name.lower():
        # This is the Orbit bot
        if cmd == 'status':
            await ctx.invoke(bot.get_command('status'))
        elif cmd == 'screenshot':
            await ctx.invoke(bot.get_command('screenshot'))
        elif cmd == 'detect':
            await ctx.invoke(bot.get_command('detect'))

@bot.command(name='gravity')
async def gravity_command(ctx, *, cmd):
    if 'gravity' in bot.user.name.lower():
        if cmd == 'status':
            await ctx.invoke(bot.get_command('status'))
        elif cmd == 'screenshot':
            await ctx.invoke(bot.get_command('screenshot'))
        elif cmd == 'detect':
            await ctx.invoke(bot.get_command('detect'))

@bot.command(name='horizon')
async def horizon_command(ctx, *, cmd):
    if 'horizon' in bot.user.name.lower():
        if cmd == 'status':
            await ctx.invoke(bot.get_command('status'))
        elif cmd == 'screenshot':
            await ctx.invoke(bot.get_command('screenshot'))
        elif cmd == 'detect':
            await ctx.invoke(bot.get_command('detect'))

# Pattern 3: Multi-camera commands
@bot.command(name='all-cameras')
async def all_cameras(ctx, cmd):
    """All cameras respond to this command"""
    bot_name = bot.user.name
    status = read_camera_status()

    if cmd == 'status':
        emoji = "🟢" if status['detected'] else "⚪"
        await ctx.send(f"{emoji} **{bot_name}**: {'Active' if status['detected'] else 'Idle'}")

    elif cmd == 'screenshot':
        screenshot_file = Path.home() / "oak-projects" / "latest_frame.jpg"
        if screenshot_file.exists():
            await ctx.send(f"📸 **{bot_name}**:", file=discord.File(screenshot_file))
        else:
            await ctx.send(f"❌ **{bot_name}**: No screenshot available")

    elif cmd == 'detect':
        await ctx.send(f"👁️ **{bot_name}**: {status['count']} people detected")

@bot.command(name='compare-cameras')
async def compare_cameras(ctx):
    """Show formatted status for this camera"""
    status = read_camera_status()
    bot_name = bot.user.name

    color = discord.Color.green() if status['detected'] else discord.Color.light_grey()
    embed = discord.Embed(title=bot_name, color=color)

    status_text = "🟢 Detecting" if status['detected'] else "⚪ Idle"
    embed.add_field(name="Status", value=status_text, inline=True)
    embed.add_field(name="People Count", value=str(status['count']), inline=True)

    timestamp = status.get('timestamp', 'Unknown')
    embed.add_field(name="Last Update", value=timestamp, inline=False)

    await ctx.send(embed=embed)

# Existing basic commands
@bot.command()
async def status(ctx):
    await ctx.send(f"✅ **{bot.user.name}** is online and monitoring!")

@bot.command()
async def detect(ctx):
    status = read_camera_status()
    if status['detected']:
        await ctx.send(f"🟢 **{bot.user.name}**: {status['count']} people detected")
    else:
        await ctx.send(f"⚪ **{bot.user.name}**: No people detected")

@bot.command()
async def screenshot(ctx):
    screenshot_file = Path.home() / "oak-projects" / "latest_frame.jpg"
    if screenshot_file.exists():
        await ctx.send(f"📸 **{bot.user.name}**:", file=discord.File(screenshot_file))
    else:
        await ctx.send("❌ No screenshot available - is person_detector.py running?")

@bot.command()
async def ping(ctx):
    await ctx.send(f"🏓 Pong! **{bot.user.name}** is alive!")

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')
    print('Bot is ready!')

# Run the bot
if __name__ == '__main__':
    import os
    from dotenv import load_dotenv

    load_dotenv()
    TOKEN = os.getenv('DISCORD_BOT_TOKEN')

    if not TOKEN:
        print("ERROR: DISCORD_BOT_TOKEN not found in .env file!")
        exit(1)

    bot.run(TOKEN)
```

---

### Testing Multi-Camera Commands

Once all three bots are running in the Smart Objects channel:

**Test @mentions:**
```
You: @OrbitBot status
OrbitBot: 🤖 OrbitBot: I'm online and watching!

You: @GravityBot what do you see?
GravityBot: 👁️ GravityBot: I see 2 people

You: @HorizonBot send a picture
HorizonBot: 📸 HorizonBot view: [screenshot]
```

**Test camera-specific commands:**
```
You: !orbit status
OrbitBot: ✅ OrbitBot is online and monitoring!
(only OrbitBot responds)

You: !gravity screenshot
GravityBot: 📸 GravityBot: [screenshot]
(only GravityBot responds)
```

**Test multi-camera commands:**
```
You: !all-cameras status
OrbitBot: 🟢 OrbitBot: Active
GravityBot: ⚪ GravityBot: Idle
HorizonBot: 🟢 HorizonBot: Active
(all three respond!)

You: !compare-cameras
OrbitBot: [status card embed]
GravityBot: [status card embed]
HorizonBot: [status card embed]
(all three send formatted cards)
```

This creates a much more interactive, conversational experience with your cameras as smart objects!

---

## Step 8: Auto-Start the Bot (Future - Not Currently Implemented)

**Currently NOT implemented** - We're still in the exploratory phase!

For now, run the camera bots manually when you need them. In the future, once each camera has a stable configuration, you could set up the bots to auto-start on boot.

### How to Set It Up (When Ready)

Create a systemd service for the camera bot:

```bash
sudo nano /etc/systemd/system/discord-bot.service
```

Paste (replace `carrie` with your username):

```ini
[Unit]
Description=Discord Bot for OAK-D Camera (OrbitBot/GravityBot/HorizonBot)
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

**Note:** This would only apply to camera bots (Orbit, Gravity, Horizon) that post to the Smart Objects channel. Personal bots would be run manually as needed.

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

## Complete Bot Command Reference

### Basic Commands (All Cameras)

| Command       | Description          | Example Response                |
| ------------- | -------------------- | ------------------------------- |
| `!ping`       | Test if bot is alive | "Pong! 🏓"                      |
| `!status`     | Camera status        | "✅ Camera online"              |
| `!detect`     | Current detection    | "🟢 2 people detected"          |
| `!screenshot` | Get camera image     | Sends live photo from camera 📸 |
| `!help`       | List commands        | Shows all commands              |

### @Mention Commands (Direct Conversation)

| Command                      | Response                                        |
| ---------------------------- | ----------------------------------------------- |
| `@OrbitBot status`           | "🤖 OrbitBot: I'm online and watching!"         |
| `@GravityBot what do you see?` | "👁️ GravityBot: I see 2 people"              |
| `@HorizonBot screenshot`     | Sends screenshot from Horizon camera            |

### Camera-Specific Commands

| Command              | Response                                      |
| -------------------- | --------------------------------------------- |
| `!orbit status`      | Only OrbitBot responds                        |
| `!gravity screenshot`| Only GravityBot sends its screenshot          |
| `!horizon detect`    | Only HorizonBot reports detection count       |

### Multi-Camera Commands

| Command                  | Response                                            |
| ------------------------ | --------------------------------------------------- |
| `!all-cameras status`    | All three bots respond with their status           |
| `!all-cameras screenshot`| All three bots send their screenshots              |
| `!all-cameras detect`    | All three bots report detection counts             |
| `!compare-cameras`       | All three bots send formatted status embed cards   |

---

## Optional Advanced Features (For Later)

Once you have the multi-camera system working, you could add:

### Dynamic Configuration

- **`!set-threshold <value>`** - Adjust detection threshold remotely (without restart)
- **`!set-model yolov8`** - Switch YOLO models on the fly
- **`!enable-zones`** / **`!disable-zones`** - Toggle zone detection
- **`!set-fps 30`** - Change camera frame rate

### Recording & Data

- **`!record <seconds>`** - Capture video clip on demand
- **`!stats`** - Show detection statistics from database
- **`!daily-summary`** - Get automated daily reports
- **`!export-csv today`** - Export detection data

### Security & Permissions

- **Role-based permissions** - Only admins can control cameras
- **`!camera-password <password>`** - Require password for sensitive commands
- **Audit logging** - Track who issued which commands

### Advanced Coordination

- **`!synchronized-capture`** - All cameras take screenshot at exact same time
- **`!track-person`** - Coordinate tracking across multiple cameras
- **`!alert-mode on`** - All cameras switch to high-sensitivity detection
- **Smart handoff** - When person leaves one camera's view, alert other cameras

---

## Troubleshooting

### Bot won't connect

- Check token is correct in `.env`
- Make sure Message Content Intent is enabled
- Verify bot has permissions in server

### Bot doesn't respond to commands

- Check bot is online in Discord (should see it in the member list)
- Make sure you're using correct command prefix (`!`)
- Make sure you're in the Smart Objects channel (camera bots only post there)
- Check terminal output where you're running `python3 discord_bot.py` for errors

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

## Summary: Which Bot Am I Setting Up?

Still confused? Here's a quick guide:

### Camera Bots (Follow This Guide)

**You're setting up a camera bot if:**
- ✅ You want the camera to post to the Smart Objects channel
- ✅ You want everyone to see detection alerts
- ✅ You want people to use `!status`, `!screenshot` commands
- ✅ The bot name is OrbitBot, GravityBot, or HorizonBot

**Token to use:** `DISCORD_BOT_TOKEN` (the camera bot you just created)

---

### Personal Bots (Different Setup - Use discord_dm_notifier.py)

**You're using a personal bot if:**
- ✅ You want private DM notifications just for you
- ✅ You're testing without spamming the class
- ✅ Your bot has your own custom name
- ✅ You already created this bot earlier

**Token to use:** `DISCORD_DM_BOT_TOKEN` (your personal bot from before)

**Script to use:** `discord_dm_notifier.py` (not `discord_bot.py`)

---

## Next Steps After Basic Bot Works

1. ✅ Test all commands work in Smart Objects channel
2. ✅ Integrate with person_detector.py
3. ✅ Set up auto-start (future - when stable)
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
