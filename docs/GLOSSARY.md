# Glossary

A quick-reference glossary of terms, tools, and concepts from the [Student Quick Start Guide](STUDENT_QUICKSTART.md).

---

## Hardware

| Term | Definition |
|------|-----------|
| **Raspberry Pi 5** | A small, affordable single-board computer used as the host machine for the OAK-D camera. The class has three: Orbit, Gravity, and Horizon (each 16GB RAM). |
| **OAK-D** | A Luxonis depth camera with an onboard AI chip (Myriad X VPU) that can run neural networks directly on the device without needing the host CPU. Connects via USB 3.0. |
| **Myriad X VPU** | The Vision Processing Unit built into the OAK-D camera. It runs YOLO and other neural network models on-device, offloading work from the Raspberry Pi. |
| **USB 3.0 (blue port)** | The faster USB standard. Always plug the OAK-D into a blue USB port for reliable data transfer. |
| **VNC** | Virtual Network Computing. A way to remotely view and control the Pi's graphical desktop. Only one user can hold the desktop seat at a time. |

---

## Software & Libraries

| Term | Definition |
|------|-----------|
| **depthai (3.x)** | The Luxonis Python library (version 3) for communicating with OAK-D cameras. It uses a pipeline-based architecture to configure camera nodes, neural networks, and output queues. |
| **depthai-nodes** | A companion library to depthai that provides high-level neural network parsing nodes (e.g., `ParsingNeuralNetwork`), so you don't have to manually decode YOLO output. |
| **OpenCV (`cv2`)** | An open-source computer vision library used for displaying video feeds, drawing bounding boxes, and image processing. |
| **discord.py** | A Python library for building Discord bots. Used for two-way interaction between users and cameras via commands like `!status` and `!screenshot`. |
| **python-dotenv** | A library that loads environment variables from a `.env` file into your Python script, keeping secrets like bot tokens out of your code. |
| **NumPy** | A numerical computing library for Python. Used for array operations in image processing and detection logic. |

---

## Networking & Access

| Term | Definition |
|------|-----------|
| **SSH (Secure Shell)** | A protocol for securely connecting to a remote computer's terminal over a network. Used to log into the Raspberry Pis from your laptop (e.g., `ssh orbit`). |
| **`scp` (Secure Copy)** | A command-line tool that copies files between your local computer and a remote machine over SSH. Example: `scp file.py orbit:~/oak-projects/`. |
| **SSH Config (`~/.ssh/config`)** | A file on your local computer that stores shortcuts for SSH connections (hostnames, IP addresses, usernames, key files) so you can type `ssh orbit` instead of the full connection details. |
| **SSH Key** | A cryptographic key pair used for passwordless authentication. The class uses `id_ed25519_smartobjects` keys. |
| **Hostname** | The name identifying a machine on the network. The three Pis are named `orbit`, `gravity`, and `horizon`. Type `hostname` in a terminal to see which machine you're on. |

---

## Python & Terminal Concepts

| Term | Definition |
|------|-----------|
| **Virtual Environment (venv)** | An isolated Python environment with its own installed packages. The class uses a shared venv at `/opt/oak-shared/venv/`. Activate it with `activate-oak`. |
| **`activate-oak`** | A bash alias that activates the shared virtual environment. Equivalent to `source /opt/oak-shared/venv/bin/activate`. You'll see `(venv)` in your prompt when active. |
| **Flags / Command-Line Arguments** | Options passed to a script when you run it, prefixed with `--`. Examples: `--display` (show video window), `--log` (save to file), `--discord` (enable notifications), `--threshold 0.7` (set sensitivity). |
| **`Ctrl+C`** | Keyboard shortcut to stop a running script in the terminal. |
| **File Path** | The location of a file on the system, written as a continuous string with `/` separating directories. Example: `~/oak-projects/.env`. No spaces allowed in the path. |
| **`~` (Tilde)** | Shorthand for your home directory. `~/oak-projects` means `/home/yourusername/oak-projects`. |
| **`nano`** | A simple terminal-based text editor. Save with `Ctrl+O`, exit with `Ctrl+X`. |

---

## Detection & AI

| Term | Definition |
|------|-----------|
| **YOLO (You Only Look Once)** | A family of real-time object detection neural network models. The project uses YOLOv6-nano by default, which can detect 80 object classes from the COCO dataset. |
| **COCO Classes** | The 80 object categories the YOLO model can detect (person, car, dog, chair, etc.). Person is class 0. |
| **Detection Threshold** | A confidence value between 0.0 and 1.0. Only detections above this threshold are reported. Higher = fewer false positives, lower = more sensitive. Default is 0.5. |
| **Debouncing** | A technique to prevent rapid on/off flickering of detection states. The detector requires a state to persist for 1.5 seconds before triggering, ensuring stable results. |
| **Temporal Smoothing** | Filtering detection results over time to reduce noise and produce more consistent readings. Works together with debouncing. |
| **EAR (Eye Aspect Ratio)** | A measurement used in fatigue detection. It calculates the openness of the eye from face landmarks. When EAR drops below a threshold, the system flags drowsiness. |
| **Gaze Estimation** | The process of determining where a person is looking based on eye position and head pose. Outputs direction labels like "looking left" or "looking right." |
| **Bounding Box** | A rectangle drawn around a detected object in the video feed, showing its location and size. |
| **Face Landmarks** | Specific points on a face (eyes, nose, mouth, jawline) detected by a neural network. Used for fatigue detection and gaze estimation. |

---

## Pipeline Architecture (depthai 3.x)

| Term | Definition |
|------|-----------|
| **Pipeline** | The core structure in depthai that connects camera inputs, neural network processing, and data outputs. Created with `dai.Pipeline(device)` and started with `pipeline.start()`. |
| **Node** | A building block inside a pipeline. Examples: `ColorCamera` (captures video), `ParsingNeuralNetwork` (runs AI model). Nodes are linked together to form a processing chain. |
| **ColorCamera Node** | The RGB camera node that captures video frames. Configured with resolution, FPS, and preview size to match the AI model's expected input. |
| **ParsingNeuralNetwork** | A node from depthai-nodes that runs a neural network and automatically parses the output into usable detection objects (labels, confidence scores, bounding boxes). |
| **Output Queue** | A data queue created directly from a node's output (e.g., `cam.preview.createOutputQueue()`). Used to retrieve frames and detection results in your Python code. Must be created before `pipeline.start()`. |
| **Model Archive / NNArchive** | A packaged neural network model downloaded from Luxonis Hub. Contains the model weights and configuration needed to run inference on the VPU. |
| **Luxonis Hub** | An online repository of pre-trained AI models optimized for OAK-D cameras. Browse at https://models.luxonis.com. |

---

## Discord Integration

| Term | Definition |
|------|-----------|
| **Webhook** | A one-way notification mechanism. The detector sends messages to a Discord channel via a webhook URL. No bot needed, but it can't receive commands. |
| **Discord Bot** | A two-way interactive program that responds to user commands in Discord (e.g., `!status`, `!screenshot`, `!detect`). Requires a bot token. |
| **Bot Token** | A secret string that authenticates your bot with Discord's API. Stored in the `.env` file. Never share or commit it to GitHub. |
| **DM Bot (Personal)** | A Discord bot that sends private direct messages to your account. Uses `DISCORD_DM_BOT_TOKEN` and `DISCORD_USER_ID` in `.env`. |
| **Camera Bot** | The class-shared bots (OrbitBot, GravityBot, HorizonBot) that post to the public Smart Objects channel. Each Pi has its own camera bot token. |
| **`.env` File** | A configuration file that stores secret values (tokens, webhook URLs) as environment variables. Located at `~/oak-projects/.env` on the Pi. Never commit to GitHub. |
| **`chmod 600`** | A command that restricts file permissions so only the owner can read/write it. Used on `.env` to keep tokens private. |

---

## Project Scripts

| Script | Purpose |
|--------|---------|
| **`person_detector_with_display.py`** | Detects people using YOLO on the OAK-D camera. Supports display, logging, and Discord notifications. |
| **`fatigue_detector.py`** | Monitors eye aspect ratio to detect drowsiness. Uses face landmarks and sends alerts via personal DM bot. |
| **`gaze_detector.py`** | Estimates where a person is looking and outputs gaze direction to the terminal. |
| **`discord_bot.py`** | The public camera bot that responds to commands (`!ping`, `!status`, `!screenshot`, `!detect`). |
| **`discord_dm_notifier.py`** | Your personal DM bot that sends private notifications to your Discord account. |
| **`discord_notifier.py`** | A helper module for sending webhook-based notifications to a Discord channel. |

---

## Project Directories

| Path | Description |
|------|-------------|
| `~/oak-projects/` | Your working project directory on the Raspberry Pi. Contains scripts, config, and auto-generated files. |
| `/opt/oak-shared/venv/` | The shared Python virtual environment with all required libraries pre-installed. |
| `~/oak-projects/.env` | Environment file with Discord tokens and webhook URLs (secret, never commit). |
| `~/oak-projects/camera_status.json` | Auto-generated JSON file with current detection status. Read by the Discord bot for `!status` queries. |
| `~/oak-projects/latest_frame.jpg` | The most recent screenshot from the camera. Sent when a user runs `!screenshot`. |
| `utils/` | Helper Python modules (face landmarks, keypoint processing, etc.) needed by fatigue and gaze detectors. |
| `depthai_models/` | YAML configuration files that tell the camera which AI models to download and use. |

---

## Bot Commands

| Command | What It Does |
|---------|-------------|
| `!ping` | Checks if the camera bot is online and responsive. |
| `!status` | Reports whether the camera is running and what it's detecting. |
| `!screenshot` | Captures and sends the current camera frame to Discord. |
| `!detect` | Returns the current detection state (people detected, count, etc.). |
| `!help` | Lists all available bot commands. |

---

## Common Flags

| Flag | What It Does |
|------|-------------|
| `--display` | Opens a live video window showing the camera feed with detection overlays. Requires VNC or a physical monitor. |
| `--log` | Saves detection events to a timestamped log file. |
| `--discord` | Enables webhook notifications to the class Discord channel. |
| `--discord-quiet` | Only sends notifications when something is detected (not when the area clears). |
| `--threshold <value>` | Sets detection confidence threshold (0.0-1.0). Default is 0.5. |
| `--model <model_ref>` | Specifies which YOLO model to use from Luxonis Hub. |

---

## Claude Code

| Term | Definition |
|------|-----------|
| **Claude Code** | Anthropic's CLI tool that provides AI assistance for coding. Run `claude` from your project directory to get context-aware help with your scripts. |
| **Slash Commands** | Shortcuts like `/research-dev` and `/help-me-code` that trigger structured workflows in Claude Code. |
| **`/research-dev`** | A slash command that guides Claude through a research-first development process: research existing examples, compare approaches, design, implement, and review. |
| **`/help-me-code`** | A student-friendly slash command for quick coding help. Explains plans before building and teaches as it goes. |

---

## Workflow Summary

| Step | Where | What |
|------|-------|------|
| 1. Clone repo | Local computer | `git clone` the GitHub repository |
| 2. Edit code | Local computer | Modify scripts on your laptop |
| 3. Copy files | Local computer | `scp` files to the Pi |
| 4. Set up `.env` | Raspberry Pi | Add tokens if needed |
| 5. Activate venv | Raspberry Pi | Run `activate-oak` |
| 6. Run scripts | Raspberry Pi | Execute detection scripts |
| 7. Monitor | Discord / Terminal | Check notifications and output |
