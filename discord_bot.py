#!/usr/bin/env python3
"""
Simple Discord Bot for OAK-D Camera
====================================
Responds to commands and sends camera status updates.

Commands:
    !ping        - Test if bot is alive
    !status      - Check camera status
    !detect      - Get current detection info
    !screenshot  - Get live camera image
    !help        - Show available commands

Setup:
    1. Install discord.py: pip install discord.py
    2. Add DISCORD_BOT_TOKEN to .env file
    3. Run: python3 discord_bot.py
"""

import discord
from discord.ext import commands
import os
import json
from pathlib import Path
from datetime import datetime

# Load environment variables from ~/oak-projects/.env (per-user)
try:
    from dotenv import load_dotenv
    load_dotenv(Path.home() / "oak-projects" / ".env")
except ImportError:
    print("⚠️  python-dotenv not installed - make sure DISCORD_BOT_TOKEN is in environment")

# Configuration
BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
STATUS_FILE = Path.home() / "oak-projects" / "camera_status.json"
SCREENSHOT_FILE = Path.home() / "oak-projects" / "latest_frame.jpg"

# Check token
if not BOT_TOKEN:
    print("❌ Error: DISCORD_BOT_TOKEN not set in .env file")
    print("   Add this line to ~/oak-projects/.env:")
    print("   DISCORD_BOT_TOKEN=your_token_here")
    exit(1)

# Create bot with command prefix
intents = discord.Intents.default()
intents.message_content = True  # Required to read message content
bot = commands.Bot(command_prefix='!', intents=intents)

# Remove default help command (we'll make our own)
bot.remove_command('help')


# --- Event Handlers ---

@bot.event
async def on_ready():
    """Called when bot successfully connects to Discord."""
    print(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    print('Bot is ready!')
    print('------')


@bot.event
async def on_message(message):
    """Called for every message sent in channels the bot can see."""
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Process commands
    await bot.process_commands(message)


# --- Commands ---

@bot.command(name='ping', help='Test if the bot is alive')
async def ping(ctx):
    """Simple ping command to test bot responsiveness."""
    latency = round(bot.latency * 1000)  # Convert to milliseconds
    await ctx.send(f'🏓 Pong! (Latency: {latency}ms)')


@bot.command(name='status', help='Check camera status')
async def status(ctx):
    """Check if camera system is running."""
    try:
        # Try to read status file
        if STATUS_FILE.exists():
            status_data = json.loads(STATUS_FILE.read_text())
            timestamp = status_data.get('timestamp', 'unknown')
            username = status_data.get('username', 'unknown')
            hostname = status_data.get('hostname', 'unknown')

            # Check if status is recent (within last 10 seconds)
            try:
                status_time = datetime.fromisoformat(timestamp)
                age = (datetime.now() - status_time).total_seconds()

                if age < 10:
                    user_info = f"👤 Running: **{username}** on **{hostname}**\n" if username != 'unknown' else ""
                    await ctx.send(f"✅ Camera is **ONLINE**\n{user_info}📊 Last update: {age:.1f}s ago")
                else:
                    await ctx.send(f"⚠️ Camera status is **STALE**\n📊 Last update: {age:.0f}s ago\nCamera may be offline.")
            except:
                await ctx.send(f"✅ Camera status file exists\n📊 Timestamp: {timestamp}")
        else:
            await ctx.send("❌ Camera is **OFFLINE**\n💡 Status file not found. Is person_detector.py running?")

    except Exception as e:
        await ctx.send(f"❌ Error checking status: {str(e)}")


@bot.command(name='detect', help='Get current detection status')
async def detect(ctx):
    """Show current person detection status."""
    try:
        if not STATUS_FILE.exists():
            await ctx.send("❌ No detection data available\n💡 Make sure person_detector.py is running")
            return

        # Read status
        status_data = json.loads(STATUS_FILE.read_text())
        detected = status_data.get('detected', False)
        count = status_data.get('count', 0)
        timestamp = status_data.get('timestamp', 'unknown')
        username = status_data.get('username', 'unknown')
        hostname = status_data.get('hostname', 'unknown')

        # Format response
        if detected:
            emoji = "🟢"
            status_text = f"**PERSON DETECTED**\n👥 Count: {count}"
        else:
            emoji = "⚪"
            status_text = "**No person detected**"

        # Add user info if available
        user_info = f"👤 Camera: **{username}** on **{hostname}**\n" if username != 'unknown' else ""

        await ctx.send(f"{emoji} {status_text}\n{user_info}🕐 Last update: {timestamp}")

    except Exception as e:
        await ctx.send(f"❌ Error reading detection data: {str(e)}")


@bot.command(name='screenshot', help='Get a screenshot from the camera')
async def screenshot(ctx):
    """Send the latest camera frame."""
    try:
        if not SCREENSHOT_FILE.exists():
            await ctx.send("❌ No screenshot available\n💡 Make sure person_detector.py is running")
            return

        # Check screenshot age
        file_age = datetime.now().timestamp() - SCREENSHOT_FILE.stat().st_mtime

        if file_age > 30:
            await ctx.send(f"⚠️ Screenshot is old ({file_age:.0f}s)\nCamera may not be running.")
            return

        # Send the screenshot
        await ctx.send(
            f"📸 **Camera Screenshot**\n🕐 Captured: {file_age:.1f}s ago",
            file=discord.File(str(SCREENSHOT_FILE))
        )

    except Exception as e:
        await ctx.send(f"❌ Error sending screenshot: {str(e)}")


@bot.command(name='help', help='Show available commands')
async def help_command(ctx):
    """Display help message with all available commands."""
    help_text = """
**🤖 OAK-D Camera Bot Commands**

`!ping` - Test if bot is alive
`!status` - Check if camera is running
`!detect` - Get current detection status
`!screenshot` - Get a live image from camera
`!help` - Show this message

**💡 Tips:**
• Camera must be running to get detection data
• Screenshots are captured every 5 seconds
• Use `!status` to check if camera is online
    """
    await ctx.send(help_text)


# --- Helper Function for Person Detector Integration ---

async def send_alert(message: str):
    """
    Send an alert to all channels the bot can access.
    Call this from person_detector.py to send notifications.

    Example:
        await bot.send_alert("🟢 Person detected!")
    """
    for guild in bot.guilds:
        for channel in guild.text_channels:
            if channel.permissions_for(guild.me).send_messages:
                try:
                    await channel.send(message)
                    break  # Only send to first available channel per server
                except:
                    continue


# Add send_alert to bot object so person_detector can access it
bot.send_alert = send_alert


# --- Main ---

if __name__ == '__main__':
    print("Starting Discord bot...")
    print(f"Command prefix: !")
    print("Commands: !ping, !status, !detect, !screenshot, !help")
    print("\nPress Ctrl+C to stop\n")

    try:
        # Start the bot
        bot.run(BOT_TOKEN)
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
