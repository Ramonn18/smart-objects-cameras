# Camera Bot Tokens Reference

**For Instructor Use Only**

This document tracks the Discord bot tokens for the three camera bots. Students do not need these - they are already configured on each Pi.

---

## Camera Bot Configuration

Each Raspberry Pi has its own camera bot already configured in the `.env` file.

### Orbit Pi (Camera 1)
- **Bot Name:** OrbitBot
- **Token Location:** `orbit:~/oak-projects/.env`
- **Token Variable:** `DISCORD_BOT_TOKEN`
- **Status:** ✅ Configured

To retrieve the token:
```bash
ssh orbit.local "grep DISCORD_BOT_TOKEN ~/oak-projects/.env"
```

---

### Gravity Pi (Camera 2)
- **Bot Name:** GravityBot
- **Token Location:** `gravity:~/oak-projects/.env`
- **Token Variable:** `DISCORD_BOT_TOKEN`
- **Status:** ✅ Configured

To retrieve the token:
```bash
ssh gravity.local "grep DISCORD_BOT_TOKEN ~/oak-projects/.env"
```

---

### Horizon Pi (Camera 3)
- **Bot Name:** HorizonBot
- **Token Location:** `horizon:~/oak-projects/.env`
- **Token Variable:** `DISCORD_BOT_TOKEN`
- **Status:** ✅ Configured

To retrieve the token:
```bash
ssh horizon.local "grep DISCORD_BOT_TOKEN ~/oak-projects/.env"
```

---

## Complete .env Structure on Each Pi

Each Pi's `~/oak-projects/.env` file contains:

```bash
# Discord Webhook (shared by all - for person detection --discord notifications)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Discord Bot Configuration (camera bot - unique per Pi)
DISCORD_APPLICATION_ID=...
DISCORD_PUBLIC_KEY=...
DISCORD_BOT_TOKEN=...

# Students add their personal DM bot tokens below (optional)
# DISCORD_USER_ID=...
# DISCORD_DM_BOT_TOKEN=...
```

---

## Student Instructions

**Students do NOT need to know these camera bot tokens!**

They only need to:
1. Copy Python files to the Pi
2. (Optional) Add their personal `DISCORD_DM_BOT_TOKEN` to `.env` if they want private DM notifications
3. Run the scripts

The camera bots (OrbitBot, GravityBot, HorizonBot) are already configured and ready to use.

---

## Sharing Camera Bot Access

If you need to share the camera bot tokens with another instructor or admin:

1. **Retrieve all tokens:**
   ```bash
   ssh orbit.local "cat ~/oak-projects/.env | grep DISCORD_BOT_TOKEN"
   ssh gravity.local "cat ~/oak-projects/.env | grep DISCORD_BOT_TOKEN"
   ssh horizon.local "cat ~/oak-projects/.env | grep DISCORD_BOT_TOKEN"
   ```

2. **Share securely** - Never post tokens in public channels!
   - Use encrypted messages
   - Share via password-protected files
   - Or regenerate tokens if needed

3. **Update Discord Developer Portal** if tokens are regenerated

---

## Regenerating Camera Bot Tokens

If a token is compromised:

1. **Go to Discord Developer Portal:** https://discord.com/developers/applications
2. **Select the bot:** OrbitBot, GravityBot, or HorizonBot
3. **Go to Bot section** → Click **Reset Token**
4. **Copy new token**
5. **Update .env on the Pi:**
   ```bash
   ssh orbit  # or gravity, or horizon
   nano ~/oak-projects/.env
   # Update DISCORD_BOT_TOKEN with new token
   # Save: Ctrl+O, Enter, Ctrl+X
   ```
6. **Restart any running bots:**
   ```bash
   # If using systemd (future):
   sudo systemctl restart discord-bot

   # If running manually:
   # Press Ctrl+C and restart
   python3 discord_bot.py
   ```

---

## Security Notes

- ✅ `.env` files are secured with `chmod 600` (only owner can read)
- ✅ `.env` is in `.gitignore` (never committed to GitHub)
- ✅ Each Pi has its own bot token (tokens are not shared)
- ✅ Students only see their personal DM bot tokens, not camera bot tokens

---

## Testing Camera Bots

To test if a camera bot is working:

1. **SSH into the Pi:**
   ```bash
   ssh orbit  # or gravity, or horizon
   ```

2. **Run the bot manually:**
   ```bash
   activate-oak
   cd ~/oak-projects
   python3 discord_bot.py
   ```

3. **Test in Discord Smart Objects channel:**
   - `!ping` - Should respond
   - `!status` - Should show camera status
   - `@OrbitBot status` - Should respond to mention

4. **Stop the bot:** Press `Ctrl+C`

---

## Backup Strategy

**Recommended:** Keep a secure backup of all camera bot tokens in a password-protected file on your local computer (not in the GitHub repo!).

Example structure:
```
smart-objects-tokens.txt (encrypted/password-protected)
├── OrbitBot Token: MTQ2N...
├── GravityBot Token: MTQ2N...
└── HorizonBot Token: MTQ2N...
```

Store this file securely and separately from the GitHub repository.
