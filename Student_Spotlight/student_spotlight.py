#!/usr/bin/env python3
"""
Student Spotlight - Classroom Presenter Recognition System
==========================================================
Uses an OAK-D camera to detect when a student enters a designated
"spotlight zone" (e.g., standing at the front to present) and announces
them to the class via Discord with a live snapshot.

Concept:
    A camera watches the front of the classroom. When a student steps
    into the spotlight zone to present or answer a question, the system:
      - Detects their presence with temporal smoothing
      - Sends a Discord notification with a snapshot
      - Tracks how long they've been in the spotlight
      - Announces when the spotlight zone is vacated

Problems this addresses:
    - Automatically acknowledging students who step up to present
    - Reducing the overhead of manually triggering Discord announcements
    - Creating a visual record of who presented (snapshot per session)
    - Encouraging participation by making it feel like a real spotlight

Usage:
    python3 student_spotlight.py                    # Basic detection
    python3 student_spotlight.py --display          # Live camera feed (VNC/X11)
    python3 student_spotlight.py --log              # Log events to file
    python3 student_spotlight.py --discord          # Discord announcements
    python3 student_spotlight.py --discord-quiet    # Only notify on entry, not exit
    python3 student_spotlight.py --threshold 0.6    # Detection sensitivity
    python3 student_spotlight.py --cooldown 5       # Seconds between repeat alerts
"""

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork
import argparse
import time
import os
import json
import cv2
import numpy as np
import socket
import getpass
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Environment & optional imports
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
    load_dotenv(Path.home() / "oak-projects" / ".env")
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    from discord_notifier import send_notification
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False

# ---------------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Student Spotlight - Classroom Presenter Recognition (DepthAI 3.x)"
)
parser.add_argument("--log",            action="store_true", help="Log events to file")
parser.add_argument("--display",        action="store_true", help="Show live camera feed (requires X11/VNC)")
parser.add_argument("--discord",        action="store_true", help="Send spotlight events to Discord")
parser.add_argument("--discord-quiet",  action="store_true", help="Only notify on spotlight entry, not exit")
parser.add_argument("--threshold",      type=float, default=0.5, help="Detection confidence threshold (0–1)")
parser.add_argument("--cooldown",       type=float, default=10.0, help="Seconds before re-alerting same event")
parser.add_argument("--model",          type=str,
                    default="luxonis/yolov6-nano:r2-coco-512x288",
                    help="Luxonis Hub model reference")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

DEBOUNCE_SECONDS       = 2.0   # State must persist this long before triggering
STATUS_UPDATE_INTERVAL = 10    # Seconds between periodic status file writes
SCREENSHOT_INTERVAL    = 5     # Seconds between auto-saved preview frames

# Spotlight zone (normalized 0–1 of the frame).
# This defines the region at the front/center of the room to watch.
# Adjust these values to match your camera angle and room layout.
SPOTLIGHT_ZONE = {
    "x_min": 0.2,
    "x_max": 0.8,
    "y_min": 0.1,
    "y_max": 0.9,
}

# Output file paths
STATUS_FILE     = Path.home() / "oak-projects" / "spotlight_status.json"
SCREENSHOT_FILE = Path.home() / "oak-projects" / "spotlight_frame.jpg"
ZONE_FILE       = Path(__file__).parent / "spotlight_zone.json"

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

in_spotlight           = False      # Confirmed current state
pending_state          = None       # Candidate state during debounce
pending_state_time     = 0.0
spotlight_start_time   = None       # When student entered spotlight
last_alert_time        = 0.0
last_status_time       = 0.0
last_screenshot_time   = 0.0
log_file               = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log_event(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    if log_file:
        log_file.write(line + "\n")
        log_file.flush()


def notify(message: str):
    if args.discord and DISCORD_AVAILABLE and os.getenv("DISCORD_WEBHOOK_URL"):
        send_notification(message, add_timestamp=False)


def format_duration(seconds: float) -> str:
    """Convert seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def write_status(in_zone: bool, count: int, username: str, hostname: str):
    try:
        duration = None
        if in_zone and spotlight_start_time:
            duration = round(time.time() - spotlight_start_time, 1)
        data = {
            "student_in_spotlight": in_zone,
            "person_count":         count,
            "spotlight_duration_s": duration,
            "timestamp":            datetime.now().isoformat(),
            "username":             username,
            "hostname":             hostname,
        }
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATUS_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        log_event(f"WARNING: Could not write status file: {e}")


def load_zone():
    """Load saved spotlight zone from file if it exists."""
    global SPOTLIGHT_ZONE
    if ZONE_FILE.exists():
        try:
            SPOTLIGHT_ZONE = json.loads(ZONE_FILE.read_text())
            log_event(f"Loaded spotlight zone from {ZONE_FILE}")
        except Exception as e:
            log_event(f"WARNING: Could not load zone file: {e}")


# ---------------------------------------------------------------------------
# Zone check
# ---------------------------------------------------------------------------


def person_in_spotlight(bbox) -> bool:
    """Return True if the person's bounding-box center falls inside the spotlight zone."""
    cx = (bbox.xmin + bbox.xmax) / 2
    cy = (bbox.ymin + bbox.ymax) / 2
    return (
        SPOTLIGHT_ZONE["x_min"] <= cx <= SPOTLIGHT_ZONE["x_max"] and
        SPOTLIGHT_ZONE["y_min"] <= cy <= SPOTLIGHT_ZONE["y_max"]
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run():
    global log_file, in_spotlight, spotlight_start_time
    global pending_state, pending_state_time
    global last_alert_time, last_status_time, last_screenshot_time

    # Identity
    try:    username = getpass.getuser()
    except: username = os.getenv("USER", "unknown")
    try:    hostname = socket.gethostname()
    except: hostname = "unknown"

    # Logging
    if args.log:
        fname = f"student_spotlight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file = open(fname, "w")
        log_event(f"Logging to {fname}")

    # Load saved zone config
    load_zone()

    log_event("Student Spotlight system starting…")
    log_event(f"Model: {args.model}  |  Threshold: {args.threshold}")
    log_event(f"Spotlight zone: {SPOTLIGHT_ZONE}")
    log_event("Press Ctrl+C to exit\n")

    notify(f"🎤 **{username}** started Student Spotlight on **{hostname}**")
    write_status(False, 0, username, hostname)
    last_status_time = time.time()

    try:
        device = dai.Device()
        platform = device.getPlatformAsString()
        log_event(f"Connected: {device.getDeviceId()} ({platform})")

        with dai.Pipeline(device) as pipeline:

            # Camera input
            cam = pipeline.create(dai.node.ColorCamera)
            cam.setPreviewSize(512, 288)
            cam.setInterleaved(False)
            cam.setFps(15)

            # Load model & create neural network with parser
            model_desc = dai.NNModelDescription(args.model, platform=platform)
            nn_archive  = dai.NNArchive(dai.getModelFromZoo(model_desc))
            nn          = pipeline.create(ParsingNeuralNetwork).build(cam.preview, nn_archive)

            # Output queues (must be created before pipeline.start())
            q_det     = nn.out.createOutputQueue(maxSize=4, blocking=False)
            q_preview = cam.preview.createOutputQueue(maxSize=4, blocking=False)

            pipeline.start()
            log_event("Pipeline running — watching spotlight zone…\n")

            while pipeline.isRunning():
                now           = time.time()
                det_msg       = q_det.tryGet()
                preview_frame = q_preview.tryGet()

                # ── Detection ───────────────────────────────────────────────
                if det_msg is not None and hasattr(det_msg, "detections"):
                    people = [
                        d for d in det_msg.detections
                        if d.label == 0 and d.confidence >= args.threshold
                    ]

                    person_count = len(people)

                    # Check if any detected person is inside the spotlight zone
                    spotlight_occupied = any(person_in_spotlight(p) for p in people)

                    # ── Debounce ─────────────────────────────────────────────
                    if spotlight_occupied != in_spotlight:
                        if pending_state == spotlight_occupied:
                            if now - pending_state_time >= DEBOUNCE_SECONDS:
                                # State confirmed — commit change
                                in_spotlight   = spotlight_occupied
                                pending_state  = None

                                if in_spotlight:
                                    spotlight_start_time = now
                                    log_event("SPOTLIGHT ON — student stepped into spotlight zone")
                                    notify("🎤 A student has stepped into the **spotlight**!")
                                    last_alert_time = now
                                else:
                                    duration = now - (spotlight_start_time or now)
                                    log_event(f"SPOTLIGHT OFF — student left after {format_duration(duration)}")
                                    if not args.discord_quiet:
                                        notify(f"👋 Spotlight cleared after **{format_duration(duration)}**")
                                    spotlight_start_time = None
                                    last_alert_time = now

                                write_status(in_spotlight, person_count, username, hostname)
                        else:
                            # New candidate — start debounce timer
                            pending_state      = spotlight_occupied
                            pending_state_time = now
                    else:
                        # State is stable — reset pending
                        pending_state      = None
                        pending_state_time = 0.0

                    # ── Repeat alert if spotlight held for a long time ────────
                    if (in_spotlight and spotlight_start_time and
                            now - last_alert_time >= args.cooldown):
                        held = format_duration(now - spotlight_start_time)
                        log_event(f"Still in spotlight — {held}")
                        notify(f"⏱️ Student still in spotlight — **{held}**")
                        last_alert_time = now

                # ── Optional live display ────────────────────────────────────
                if args.display and preview_frame is not None:
                    frame = preview_frame.getCvFrame()
                    h, w  = frame.shape[:2]

                    # Draw spotlight zone
                    x1 = int(SPOTLIGHT_ZONE["x_min"] * w)
                    x2 = int(SPOTLIGHT_ZONE["x_max"] * w)
                    y1 = int(SPOTLIGHT_ZONE["y_min"] * h)
                    y2 = int(SPOTLIGHT_ZONE["y_max"] * h)
                    color = (0, 255, 200) if in_spotlight else (100, 100, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Status label
                    status_text = "IN SPOTLIGHT" if in_spotlight else "zone empty"
                    cv2.putText(frame, status_text, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Duration overlay
                    if in_spotlight and spotlight_start_time:
                        held = format_duration(now - spotlight_start_time)
                        cv2.putText(frame, f"Duration: {held}", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    cv2.imshow("Student Spotlight", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # ── Periodic screenshot ──────────────────────────────────────
                if preview_frame is not None and now - last_screenshot_time >= SCREENSHOT_INTERVAL:
                    try:
                        cv2.imwrite(str(SCREENSHOT_FILE), preview_frame.getCvFrame())
                        last_screenshot_time = now
                    except Exception as e:
                        log_event(f"WARNING: Screenshot failed: {e}")

                # ── Periodic status heartbeat ────────────────────────────────
                if now - last_status_time >= STATUS_UPDATE_INTERVAL:
                    write_status(in_spotlight, 0, username, hostname)
                    last_status_time = now

                time.sleep(0.01)

    except KeyboardInterrupt:
        log_event("\nStudent Spotlight stopped.")
        notify(f"📴 **{username}** stopped Student Spotlight on **{hostname}**")

    finally:
        if args.display:
            cv2.destroyAllWindows()
        if log_file:
            log_file.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if args.discord and not DISCORD_AVAILABLE:
        print("ERROR: --discord requested but discord_notifier.py not found.")
        import sys
        sys.exit(1)

    run()
