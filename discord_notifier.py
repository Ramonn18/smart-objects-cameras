#!/usr/bin/env python3
"""
Simple Discord Webhook Notifier for OAK-D Camera
================================================
Sends camera detection events to Discord via webhook.

Usage:
    from discord_notifier import send_notification, send_async_notification

    # Synchronous (blocking)
    send_notification("🟢 Person detected!")

    # Asynchronous (non-blocking)
    await send_async_notification("⚪ No person detected")
"""

import os
import requests
import aiohttp
import asyncio
from datetime import datetime
from typing import Optional


def get_webhook_url() -> Optional[str]:
    """Get Discord webhook URL from environment variable."""
    webhook_url = os.getenv('DISCORD_WEBHOOK_URL')

    if not webhook_url:
        print("⚠️  DISCORD_WEBHOOK_URL not set - notifications disabled")
        print("   See DISCORD_SETUP.md for setup instructions")
        return None

    return webhook_url


def send_notification(message: str, username: str = "OAK-D Camera", add_timestamp: bool = True) -> bool:
    """
    Send a notification to Discord (synchronous/blocking).

    Args:
        message: The message to send
        username: The username to display in Discord (default: "OAK-D Camera")
        add_timestamp: Whether to add a timestamp to the message (default: True)

    Returns:
        bool: True if successful, False otherwise
    """
    webhook_url = get_webhook_url()
    if not webhook_url:
        return False

    # Add timestamp if requested
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{timestamp}] {message}"

    payload = {
        "username": username,
        "content": message
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=5)

        if response.status_code == 204:
            return True
        else:
            print(f"❌ Discord notification failed: {response.status_code}")
            return False

    except requests.exceptions.Timeout:
        print("❌ Discord notification timed out")
        return False
    except Exception as e:
        print(f"❌ Discord notification error: {e}")
        return False


async def send_async_notification(message: str, username: str = "OAK-D Camera", add_timestamp: bool = True) -> bool:
    """
    Send a notification to Discord (asynchronous/non-blocking).

    Args:
        message: The message to send
        username: The username to display in Discord (default: "OAK-D Camera")
        add_timestamp: Whether to add a timestamp to the message (default: True)

    Returns:
        bool: True if successful, False otherwise
    """
    webhook_url = get_webhook_url()
    if not webhook_url:
        return False

    # Add timestamp if requested
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"[{timestamp}] {message}"

    payload = {
        "username": username,
        "content": message
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 204:
                    return True
                else:
                    print(f"❌ Discord notification failed: {response.status}")
                    return False

    except asyncio.TimeoutError:
        print("❌ Discord notification timed out")
        return False
    except Exception as e:
        print(f"❌ Discord notification error: {e}")
        return False


def test_notification():
    """Test the Discord notification system."""
    print("🧪 Testing Discord notification...")

    if not get_webhook_url():
        print("❌ No webhook URL configured")
        return False

    success = send_notification("🧪 Test notification from OAK-D Camera system")

    if success:
        print("✅ Notification sent successfully!")
        print("   Check your Discord channel to verify")
        return True
    else:
        print("❌ Notification failed")
        return False


if __name__ == "__main__":
    # Run test when executed directly
    import sys

    # Load environment variables from ~/oak-projects/.env (per-user)
    try:
        from pathlib import Path
        from dotenv import load_dotenv
        load_dotenv(Path.home() / "oak-projects" / ".env")
    except ImportError:
        print("⚠️  python-dotenv not installed - make sure DISCORD_WEBHOOK_URL is set in environment")

    if len(sys.argv) > 1:
        # Send custom message
        message = " ".join(sys.argv[1:])
        success = send_notification(message)
        sys.exit(0 if success else 1)
    else:
        # Run test
        success = test_notification()
        sys.exit(0 if success else 1)
