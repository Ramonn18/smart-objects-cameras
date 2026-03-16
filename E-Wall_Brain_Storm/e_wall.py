#!/usr/bin/env python3
"""
E-Wall (E-CA) - Interactive Classroom Projection System
========================================================
Uses an OAK-D camera to track the presenter's position and gestures on stage,
driving dynamic visual cues projected onto the classroom wall/whiteboard.

Concept:
    A singular ultra-short-throw projector faces the wall. The OAK-D camera
    reads the presenter's actions from the stage. Purposeful gestures trigger
    visual queues on the projection surface, making the classroom more dynamic
    and interactive.

Problems this addresses:
    - Highlighting the human presenter as the focal point
    - Distinguishing purposeful gestures from mundane body language
    - Calibrating the area of focus (stage zone)
    - Handling low-light / variable lighting conditions

Usage:
    python3 e_wall.py                    # Basic tracking (console output)
    python3 e_wall.py --display          # Show live camera feed (requires display)
    python3 e_wall.py --log              # Log events to file
    python3 e_wall.py --discord          # Send gesture events to Discord
    python3 e_wall.py --threshold 0.6    # Adjust detection sensitivity
    python3 e_wall.py --calibrate        # Run stage zone calibration
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

parser = argparse.ArgumentParser(description="E-Wall Interactive Classroom System (DepthAI 3.x)")
parser.add_argument("--log",        action="store_true", help="Log events to file")
parser.add_argument("--display",    action="store_true", help="Show live camera feed (requires X11/VNC)")
parser.add_argument("--discord",    action="store_true", help="Send gesture events to Discord")
parser.add_argument("--calibrate",  action="store_true", help="Run stage zone calibration on startup")
parser.add_argument("--threshold",  type=float, default=0.5,  help="Detection confidence threshold (0–1)")
parser.add_argument("--model",      type=str,
                    default="luxonis/yolov6-nano:r2-coco-512x288",
                    help="Luxonis Hub model reference")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

DEBOUNCE_SECONDS       = 1.5   # Seconds a gesture must persist before triggering
STATUS_UPDATE_INTERVAL = 10    # Seconds between periodic status file writes
SCREENSHOT_INTERVAL    = 5     # Seconds between saved preview frames

# Stage zone (normalized 0–1). Presenter must be inside this box to trigger.
# Adjust via --calibrate or edit manually after calibration run.
STAGE_ZONE = {
    "x_min": 0.1,
    "x_max": 0.9,
    "y_min": 0.2,
    "y_max": 1.0,
}

# Gesture cooldown: minimum seconds between successive gesture events
GESTURE_COOLDOWN = 3.0

# Output file paths
STATUS_FILE     = Path.home() / "oak-projects" / "e_wall_status.json"
SCREENSHOT_FILE = Path.home() / "oak-projects" / "e_wall_frame.jpg"
CALIBRATION_FILE = Path(__file__).parent / "stage_zone.json"

# ---------------------------------------------------------------------------
# Gesture definitions
# (Expand these as you experiment — these are starting placeholders)
# ---------------------------------------------------------------------------
# A "gesture" here is a simple spatial heuristic based on the presenter's
# bounding box position in the frame. Replace with pose estimation or a
# dedicated gesture model for more sophisticated detection.

GESTURE_LABELS = {
    "center_stage":    "Presenter at center — highlight mode",
    "left_wing":       "Presenter moved left — pan projection left",
    "right_wing":      "Presenter moved right — pan projection right",
    "close_to_wall":   "Presenter near wall — zoom in on projection",
    "back_of_stage":   "Presenter far back — ambient / background mode",
}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

last_gesture          = None
last_gesture_time     = 0.0
pending_gesture       = None
pending_gesture_time  = 0.0
last_status_time      = 0.0
last_screenshot_time  = 0.0
log_file              = None

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


def write_status(gesture: str | None, presenter_on_stage: bool, username: str, hostname: str):
    try:
        data = {
            "presenter_on_stage": presenter_on_stage,
            "current_gesture":    gesture,
            "gesture_label":      GESTURE_LABELS.get(gesture, "—") if gesture else "—",
            "timestamp":          datetime.now().isoformat(),
            "username":           username,
            "hostname":           hostname,
        }
        STATUS_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        log_event(f"WARNING: Could not write status file: {e}")


def load_calibration():
    """Load saved stage zone from calibration file if it exists."""
    global STAGE_ZONE
    if CALIBRATION_FILE.exists():
        try:
            STAGE_ZONE = json.loads(CALIBRATION_FILE.read_text())
            log_event(f"Loaded stage zone from {CALIBRATION_FILE}")
        except Exception as e:
            log_event(f"WARNING: Could not load calibration: {e}")


def save_calibration():
    """Save current stage zone to file."""
    try:
        CALIBRATION_FILE.write_text(json.dumps(STAGE_ZONE, indent=2))
        log_event(f"Stage zone saved to {CALIBRATION_FILE}")
    except Exception as e:
        log_event(f"WARNING: Could not save calibration: {e}")


# ---------------------------------------------------------------------------
# Gesture classification
# ---------------------------------------------------------------------------

def classify_gesture(bbox) -> str | None:
    """
    Classify presenter position into a named gesture/zone.

    bbox: object with .xmin, .xmax, .ymin, .ymax (normalized 0–1)

    Returns a gesture key from GESTURE_LABELS, or None if outside stage zone.
    """
    cx = (bbox.xmin + bbox.xmax) / 2  # Horizontal center of presenter
    cy = (bbox.ymin + bbox.ymax) / 2  # Vertical center of presenter

    # Check if presenter is within the calibrated stage zone
    in_zone = (
        STAGE_ZONE["x_min"] <= cx <= STAGE_ZONE["x_max"] and
        STAGE_ZONE["y_min"] <= cy <= STAGE_ZONE["y_max"]
    )
    if not in_zone:
        return None

    # Determine relative position within zone
    zone_width  = STAGE_ZONE["x_max"] - STAGE_ZONE["x_min"]
    zone_height = STAGE_ZONE["y_max"] - STAGE_ZONE["y_min"]
    rel_x = (cx - STAGE_ZONE["x_min"]) / zone_width   # 0 = far left, 1 = far right
    rel_y = (cy - STAGE_ZONE["y_min"]) / zone_height  # 0 = top/back, 1 = bottom/front

    # Classify horizontal position
    if rel_x < 0.3:
        return "left_wing"
    elif rel_x > 0.7:
        return "right_wing"

    # Classify depth / distance from wall
    if rel_y < 0.3:
        return "back_of_stage"
    elif rel_y > 0.7:
        return "close_to_wall"

    return "center_stage"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run():
    global log_file, last_gesture, last_gesture_time
    global pending_gesture, pending_gesture_time
    global last_status_time, last_screenshot_time

    # Identity
    try:    username = getpass.getuser()
    except: username = os.getenv("USER", "unknown")
    try:    hostname = socket.gethostname()
    except: hostname = "unknown"

    # Logging
    if args.log:
        fname = f"e_wall_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file = open(fname, "w")
        log_event(f"Logging to {fname}")

    # Load previous calibration
    load_calibration()

    log_event("E-Wall system starting…")
    log_event(f"Model: {args.model}  |  Threshold: {args.threshold}")
    log_event(f"Stage zone: {STAGE_ZONE}")
    log_event("Press Ctrl+C to exit\n")

    notify(f"📽️ **{username}** started E-Wall on **{hostname}**")
    write_status(None, False, username, hostname)
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
            model_desc   = dai.NNModelDescription(args.model, platform=platform)
            nn_archive   = dai.NNArchive(dai.getModelFromZoo(model_desc))
            nn           = pipeline.create(ParsingNeuralNetwork).build(cam.preview, nn_archive)

            # Output queues (must be created before pipeline.start())
            q_det     = nn.out.createOutputQueue(maxSize=4, blocking=False)
            q_preview = cam.preview.createOutputQueue(maxSize=4, blocking=False)

            pipeline.start()
            log_event("Pipeline running — monitoring presenter stage…\n")

            while pipeline.isRunning():
                now           = time.time()
                det_msg       = q_det.tryGet()
                preview_frame = q_preview.tryGet()

                # ── Detection & gesture classification ──────────────────────
                if det_msg is not None and hasattr(det_msg, "detections"):
                    people = [
                        d for d in det_msg.detections
                        if d.label == 0 and d.confidence >= args.threshold
                    ]

                    presenter_on_stage = len(people) > 0

                    # Use the most-confident detection for gesture classification
                    gesture = None
                    if people:
                        best = max(people, key=lambda d: d.confidence)
                        gesture = classify_gesture(best)

                    # ── Debounce: confirm gesture only if stable ─────────────
                    if gesture != last_gesture:
                        if pending_gesture == gesture:
                            if now - pending_gesture_time >= DEBOUNCE_SECONDS:
                                # Confirm gesture change
                                last_gesture = gesture
                                last_gesture_time = now
                                pending_gesture = None

                                if gesture:
                                    label = GESTURE_LABELS.get(gesture, gesture)
                                    log_event(f"GESTURE: {gesture} → {label}")
                                    notify(f"🖐️ **Gesture:** {label}")
                                else:
                                    if presenter_on_stage:
                                        log_event("Presenter on stage — outside named zone")
                                    else:
                                        log_event("Stage empty")
                                        notify("🪑 Stage is empty")

                                write_status(gesture, presenter_on_stage, username, hostname)
                        else:
                            # New candidate gesture — start debounce timer
                            pending_gesture      = gesture
                            pending_gesture_time = now
                    else:
                        # Gesture is stable — reset pending
                        pending_gesture      = None
                        pending_gesture_time = 0.0

                # ── Optional live display ────────────────────────────────────
                if args.display and preview_frame is not None:
                    frame = preview_frame.getCvFrame()

                    # Draw stage zone overlay
                    h, w = frame.shape[:2]
                    x1 = int(STAGE_ZONE["x_min"] * w)
                    x2 = int(STAGE_ZONE["x_max"] * w)
                    y1 = int(STAGE_ZONE["y_min"] * h)
                    y2 = int(STAGE_ZONE["y_max"] * h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)

                    # Label current gesture
                    label_text = last_gesture or "—"
                    cv2.putText(frame, f"Zone: {label_text}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

                    cv2.imshow("E-Wall Preview", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # ── Periodic screenshot ──────────────────────────────────────
                if preview_frame is not None and now - last_screenshot_time >= SCREENSHOT_INTERVAL:
                    try:
                        cv2.imwrite(str(SCREENSHOT_FILE), preview_frame.getCvFrame())
                        last_screenshot_time = now
                    except Exception as e:
                        log_event(f"WARNING: screenshot failed: {e}")

                # ── Periodic status file heartbeat ───────────────────────────
                if now - last_status_time >= STATUS_UPDATE_INTERVAL:
                    write_status(last_gesture, last_gesture is not None, username, hostname)
                    last_status_time = now

                time.sleep(0.01)

    except KeyboardInterrupt:
        log_event("\nE-Wall stopped.")
        notify(f"📴 **{username}** stopped E-Wall on **{hostname}**")

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

    if args.calibrate:
        print("TODO: Interactive calibration not yet implemented.")
        print(f"      Edit STAGE_ZONE in this file, or update {CALIBRATION_FILE}")

    run()
