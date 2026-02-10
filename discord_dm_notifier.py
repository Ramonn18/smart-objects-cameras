#!/usr/bin/env python3
"""
Discord DM Bot for Fatigue Detection
=====================================
Persistent bot that watches fatigue_status.json and sends DMs
on state changes. Also accepts DM commands for two-way
conversation with the camera system.

Run in a separate terminal alongside fatigue_detector.py:
    Terminal 1: python3 fatigue_detector.py --display
    Terminal 2: python3 discord_dm_notifier.py

DM Commands (send these to the bot via Discord DM):
    status     - Get current fatigue status
    screenshot - Get latest camera frame
    pause      - Pause DM notifications
    resume     - Resume DM notifications
    help       - Show available commands

Setup:
    1. Create bot at https://discord.com/developers/applications
    2. Add DISCORD_DM_BOT_TOKEN and DISCORD_USER_ID to .env
    3. Invite bot to your server
    4. Send bot a DM to open the DM channel
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
import discord
from discord.ext import commands, tasks

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration
BOT_TOKEN = os.getenv('DISCORD_DM_BOT_TOKEN')
USER_ID = os.getenv('DISCORD_USER_ID')
STATUS_FILE = Path.home() / "oak-projects" / "fatigue_status.json"
SCREENSHOT_FILE = Path.home() / "oak-projects" / "latest_fatigue_frame.jpg"

# State tracking for file watcher
_last_status = {}
_notifications_paused = False


def read_status():
    """Read current fatigue status from file."""
    try:
        if STATUS_FILE.exists():
            return json.loads(STATUS_FILE.read_text())
    except (json.JSONDecodeError, Exception):
        pass
    return None


def format_status(status):
    """Format status data into a readable message."""
    if not status:
        return "No fatigue data available. Is fatigue_detector.py running?"

    running = status.get('running', False)
    if not running:
        return "Fatigue detector is not running."

    faces = status.get('faces_detected', 0)
    fatigued = status.get('fatigue_detected', False)
    eyes = status.get('eyes_closed', False)
    head = status.get('head_tilted', False)
    pct = status.get('fatigue_percent', 0)
    ts = status.get('timestamp', 'unknown')

    if fatigued:
        icon = "🔴"
        state = "FATIGUED"
    elif eyes or head:
        icon = "🟡"
        state = "DROWSY"
    else:
        icon = "🟢"
        state = "ALERT"

    lines = [
        f"{icon} **{state}**",
        f"Faces detected: {faces}",
        f"Eyes: {'closed' if eyes else 'open'}",
        f"Head: {'tilted' if head else 'upright'}",
        f"Fatigue level: {pct:.0%}",
        f"Last update: {ts}",
    ]
    return "\n".join(lines)


def main():
    if not BOT_TOKEN:
        print("Error: DISCORD_DM_BOT_TOKEN not set in .env file")
        print("  Add to .env: DISCORD_DM_BOT_TOKEN=your_token_here")
        return

    if not USER_ID:
        print("Error: DISCORD_USER_ID not set in .env file")
        print("  Add to .env: DISCORD_USER_ID=your_id_here")
        return

    user_id_int = int(USER_ID)

    # Create bot
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix='!', intents=intents)

    @bot.event
    async def on_ready():
        print(f"DM bot logged in as {bot.user.name}")
        print(f"Watching: {STATUS_FILE}")
        print(f"DM target: user {USER_ID}")
        print("Send 'help' to the bot via DM for commands\n")
        watch_status.start()

    @tasks.loop(seconds=1.0)
    async def watch_status():
        """Watch fatigue_status.json for state changes and send DMs."""
        global _last_status, _notifications_paused

        if _notifications_paused:
            return

        status = read_status()
        if not status or not status.get('running', False):
            return

        # First read — just store state, don't notify
        if not _last_status:
            _last_status = status.copy()
            return

        user = await bot.fetch_user(user_id_int)
        if not user:
            return

        # Check for state transitions
        prev_eyes = _last_status.get('eyes_closed', False)
        prev_head = _last_status.get('head_tilted', False)
        prev_fatigue = _last_status.get('fatigue_detected', False)
        curr_eyes = status.get('eyes_closed', False)
        curr_head = status.get('head_tilted', False)
        curr_fatigue = status.get('fatigue_detected', False)

        try:
            # Only DM on sustained fatigue state changes (debounced by detector)
            if curr_fatigue and not prev_fatigue:
                pct = status.get('fatigue_percent', 0)
                reasons = []
                if curr_eyes:
                    reasons.append("eyes closed")
                if curr_head:
                    reasons.append("head tilted")
                reason_str = " / ".join(reasons) if reasons else "sustained drowsiness"
                await user.send(f"🔴 **FATIGUE DETECTED** ({reason_str}, level: {pct:.0%})")
                print(f"  DM sent: FATIGUE DETECTED ({reason_str}, {pct:.0%})")

            if not curr_fatigue and prev_fatigue:
                await user.send("🟢 Attention restored — student alert")
                print("  DM sent: Attention restored")

        except discord.Forbidden:
            print("ERROR: Can't send DMs. Send the bot a message first.")
        except Exception as e:
            print(f"ERROR sending DM: {e}")

        _last_status = status.copy()

    @bot.event
    async def on_message(message):
        """Handle DM commands from the user."""
        global _notifications_paused

        # Only respond to DMs from the configured user
        if message.author.id != user_id_int:
            return
        if message.author == bot.user:
            return
        if message.guild is not None:
            return  # Ignore server messages

        cmd = message.content.strip().lower()

        if cmd == "status":
            status = read_status()
            await message.channel.send(format_status(status))

        elif cmd == "screenshot":
            if SCREENSHOT_FILE.exists():
                age = datetime.now().timestamp() - SCREENSHOT_FILE.stat().st_mtime
                if age > 30:
                    await message.channel.send(
                        f"Screenshot is {age:.0f}s old — camera may not be running."
                    )
                else:
                    await message.channel.send(
                        f"Captured {age:.1f}s ago",
                        file=discord.File(str(SCREENSHOT_FILE))
                    )
            else:
                await message.channel.send(
                    "No screenshot available. Is fatigue_detector.py running?"
                )

        elif cmd == "pause":
            _notifications_paused = True
            await message.channel.send("Notifications paused. Send 'resume' to restart.")
            print("  Notifications paused by user")

        elif cmd == "resume":
            _notifications_paused = False
            _last_status.clear()  # Reset to avoid stale transition alerts
            await message.channel.send("Notifications resumed.")
            print("  Notifications resumed by user")

        elif cmd == "help":
            help_text = (
                "**Fatigue DM Bot Commands**\n"
                "`status` — Current fatigue status\n"
                "`screenshot` — Latest camera frame\n"
                "`pause` — Pause notifications\n"
                "`resume` — Resume notifications\n"
                "`help` — Show this message"
            )
            await message.channel.send(help_text)

        else:
            await message.channel.send(
                f"Unknown command: `{cmd}`\nSend `help` for available commands."
            )

    print("Starting DM bot...")
    print(f"Commands: status, screenshot, pause, resume, help")
    print("Press Ctrl+C to stop\n")

    try:
        bot.run(BOT_TOKEN)
    except KeyboardInterrupt:
        print("\nDM bot stopped")


if __name__ == "__main__":
    main()
