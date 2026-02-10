#!/usr/bin/env python3
"""
Student Fatigue Detector for OAK-D (DepthAI 3.x)
=================================================
Two-stage pipeline: YuNet face detection + MediaPipe face landmarks.
Detects closed eyes and head tilting to monitor student alertness.
Sends private notifications via Discord DMs (not to shared channels).

Adapted from Luxonis oak-examples fatigue detection.

Usage:
    python3 fatigue_detector.py                    # Basic detection
    python3 fatigue_detector.py --log              # Log to file
    python3 fatigue_detector.py --dm               # Enable Discord DM notifications
    python3 fatigue_detector.py --dm --dm-quiet    # Only DM on fatigue, not when alert
"""

from pathlib import Path
from collections import deque
from datetime import datetime
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, ImgDetectionsBridge, GatherData
from depthai_nodes.node.utils import generate_script_content
import argparse
import time
import os
import json
import cv2
import numpy as np

from utils.face_landmarks import determine_fatigue

# Load environment variables for Discord
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Parse arguments
parser = argparse.ArgumentParser(
    description='OAK-D Student Fatigue Detector (DepthAI 3.x)')
parser.add_argument('--log', action='store_true', help='Log events to file')
parser.add_argument('--fps-limit', type=int, default=None,
                    help='FPS limit (default: 5 for RVC2, 30 for RVC4)')
parser.add_argument('--pitch-threshold', type=int, default=20,
                    help='Head tilt angle threshold in degrees (default: 20)')
parser.add_argument('--ear-threshold', type=float, default=0.15,
                    help='Eye aspect ratio threshold (default: 0.15)')
parser.add_argument('--device', type=str, default=None,
                    help='Optional DeviceID or IP of the camera')
parser.add_argument('--display', action='store_true',
                    help='Show live video window (requires display)')
args = parser.parse_args()

# Requested camera resolution (larger than model input to keep detail for landmarks)
REQ_WIDTH, REQ_HEIGHT = 1024, 768

# Global state tracking
log_file = None

# Fatigue state tracking
last_fatigue_status = None  # None = unknown, True = fatigued, False = alert
last_eyes_closed = None
last_head_tilted = None

# Temporal smoothing (rolling windows like Luxonis example)
closed_eye_history = deque(maxlen=30)
head_tilted_history = deque(maxlen=30)
FATIGUE_THRESHOLD = 0.75  # 75% of frames must show fatigue

# Debouncing for Discord notifications
pending_state = None
pending_state_time = None
DEBOUNCE_SECONDS = 1.5

# Status file for Discord bot integration
STATUS_FILE = Path.home() / "oak-projects" / "fatigue_status.json"
STATUS_UPDATE_INTERVAL = 10
last_status_update_time = 0

# Screenshot for Discord bot
SCREENSHOT_FILE = Path.home() / "oak-projects" / "latest_fatigue_frame.jpg"
SCREENSHOT_UPDATE_INTERVAL = 5
last_screenshot_time = 0


def log_event(message: str):
    """Print and optionally log an event."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)

    if log_file:
        log_file.write(line + "\n")
        log_file.flush()


def update_status_file(faces_detected: int, fatigue_detected: bool,
                       eyes_closed: bool, head_tilted: bool,
                       fatigue_percent: float, running: bool = True):
    """Update status file for Discord bot integration."""
    try:
        status_data = {
            "faces_detected": faces_detected,
            "fatigue_detected": fatigue_detected,
            "eyes_closed": eyes_closed,
            "head_tilted": head_tilted,
            "fatigue_percent": round(fatigue_percent, 2),
            "timestamp": datetime.now().isoformat(),
            "running": running
        }
        STATUS_FILE.write_text(json.dumps(status_data, indent=2))
    except Exception as e:
        log_event(f"WARNING: Could not update status file: {e}")


def run_detection():
    """Main fatigue detection loop using DepthAI 3.x two-stage pipeline."""
    global log_file, last_fatigue_status, last_eyes_closed, last_head_tilted
    global pending_state, pending_state_time
    global last_status_update_time, last_screenshot_time

    # Open log file if requested
    if args.log:
        log_filename = f"fatigue_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file = open(log_filename, 'w')
        log_event(f"Logging to {log_filename}")

    startup_msg = "Fatigue detector started (DepthAI 3.x, YuNet + MediaPipe landmarks)"
    log_event(startup_msg)
    log_event(f"EAR threshold: {args.ear_threshold}, Pitch threshold: {args.pitch_threshold}")
    log_event("Press Ctrl+C to exit (or 'q' in display window)\n")

    # Initialize status file
    update_status_file(
        faces_detected=0, fatigue_detected=False,
        eyes_closed=False, head_tilted=False,
        fatigue_percent=0.0, running=True
    )
    last_status_update_time = time.time()

    try:
        # Connect to device
        if args.device:
            device = dai.Device(dai.DeviceInfo(args.device))
        else:
            device = dai.Device()
        platform = device.getPlatform().name
        log_event(f"Connected to device: {device.getDeviceId()}")
        log_event(f"Platform: {platform}")

        # Set FPS limit based on platform
        fps_limit = args.fps_limit
        if fps_limit is None:
            fps_limit = 5 if platform == "RVC2" else 30
            log_event(f"FPS limit set to {fps_limit} for {platform}")

        frame_type = (
            dai.ImgFrame.Type.BGR888i if platform == "RVC4"
            else dai.ImgFrame.Type.BGR888p
        )

        with dai.Pipeline(device) as pipeline:
            log_event("Creating pipeline...")

            # --- Stage 1: Face Detection (YuNet) ---
            det_model_description = dai.NNModelDescription.fromYamlFile(
                str(Path(__file__).parent / "depthai_models" / f"yunet.{platform}.yaml")
            )
            det_model_nn_archive = dai.NNArchive(
                dai.getModelFromZoo(det_model_description)
            )
            det_model_w, det_model_h = det_model_nn_archive.getInputSize()

            # --- Stage 2: Face Landmarks (MediaPipe) ---
            rec_model_description = dai.NNModelDescription.fromYamlFile(
                str(Path(__file__).parent / "depthai_models" / f"mediapipe_face_landmarker.{platform}.yaml")
            )
            rec_model_nn_archive = dai.NNArchive(
                dai.getModelFromZoo(rec_model_description)
            )
            rec_model_w, rec_model_h = rec_model_nn_archive.getInputSize()

            # --- Camera input ---
            cam = pipeline.create(dai.node.Camera).build()
            cam_out = cam.requestOutput(
                size=(REQ_WIDTH, REQ_HEIGHT), type=frame_type, fps=fps_limit
            )

            # Resize to face detection model input size
            resize_node = pipeline.create(dai.node.ImageManip)
            resize_node.initialConfig.setOutputSize(det_model_w, det_model_h)
            resize_node.initialConfig.setReusePreviousImage(False)
            resize_node.inputImage.setBlocking(True)
            cam_out.link(resize_node.inputImage)

            # Face detection neural network
            det_nn = pipeline.create(ParsingNeuralNetwork).build(
                resize_node.out, det_model_nn_archive
            )

            # Bridge detection output format for Script node
            det_bridge = pipeline.create(ImgDetectionsBridge).build(det_nn.out)

            # Script node to coordinate face cropping
            script_node = pipeline.create(dai.node.Script)
            det_bridge.out.link(script_node.inputs["det_in"])
            cam_out.link(script_node.inputs["preview"])
            script_content = generate_script_content(
                resize_width=rec_model_w,
                resize_height=rec_model_h,
            )
            script_node.setScript(script_content)

            # Crop node for face regions
            crop_node = pipeline.create(dai.node.ImageManip)
            crop_node.inputConfig.setWaitForMessage(True)
            script_node.outputs["manip_cfg"].link(crop_node.inputConfig)
            script_node.outputs["manip_img"].link(crop_node.inputImage)

            # Face landmark neural network
            landmark_nn = pipeline.create(ParsingNeuralNetwork).build(
                crop_node.out, rec_model_description
            )

            # Sync detections with landmark results
            gather_data_node = pipeline.create(GatherData).build(fps_limit)
            landmark_nn.out.link(gather_data_node.input_data)
            det_nn.out.link(gather_data_node.input_reference)

            # Output queues (must be created before pipeline.start())
            q_gather = gather_data_node.out.createOutputQueue(
                maxSize=4, blocking=False
            )
            q_preview = cam_out.createOutputQueue(
                maxSize=4, blocking=False
            )

            log_event("Pipeline created.")
            pipeline.start()
            log_event("Detection started. Monitoring for fatigue...\n")

            while pipeline.isRunning():
                # Get gathered data (synced detections + landmarks)
                gather_msg = q_gather.tryGet()

                # Get preview frame for screenshots
                preview_frame = q_preview.tryGet()

                if gather_msg is not None:
                    from depthai_nodes import ImgDetectionsExtended, Keypoints

                    detections_msg = gather_msg.reference_data
                    landmarks_list = gather_msg.gathered
                    src_w, src_h = detections_msg.transformation.getSize()

                    faces_detected = len(detections_msg.detections)
                    current_eyes_closed = False
                    current_head_tilted = False

                    for detection, landmarks in zip(
                        detections_msg.detections, landmarks_list
                    ):
                        if isinstance(landmarks, Keypoints):
                            head_tilted, eyes_closed = determine_fatigue(
                                (src_h, src_w), landmarks,
                                pitch_angle=args.pitch_threshold
                            )

                            closed_eye_history.append(eyes_closed)
                            head_tilted_history.append(head_tilted)

                            if eyes_closed:
                                current_eyes_closed = True
                            if head_tilted:
                                current_head_tilted = True

                    # Calculate fatigue percentages from rolling window
                    percent_eyes_closed = (
                        sum(closed_eye_history) / len(closed_eye_history)
                        if closed_eye_history else 0.0
                    )
                    percent_head_tilted = (
                        sum(head_tilted_history) / len(head_tilted_history)
                        if head_tilted_history else 0.0
                    )
                    fatigue_percent = max(percent_eyes_closed, percent_head_tilted)
                    fatigue_detected = fatigue_percent >= FATIGUE_THRESHOLD

                    current_time = time.time()

                    # Console status line (overwrite in place)
                    eyes_str = "CLOSED" if current_eyes_closed else "open"
                    head_str = "TILTED" if current_head_tilted else "up"
                    print(
                        f"\r  Faces: {faces_detected} | "
                        f"Eyes: {eyes_str} ({percent_eyes_closed:.0%}) | "
                        f"Head: {head_str} ({percent_head_tilted:.0%}) | "
                        f"Fatigue: {fatigue_percent:.0%}  ",
                        end="", flush=True
                    )

                    # Update status file on state transitions
                    # (DM bot in separate process watches this file)
                    state_changed = False
                    if current_eyes_closed != last_eyes_closed and last_eyes_closed is not None:
                        state_changed = True
                    if current_head_tilted != last_head_tilted and last_head_tilted is not None:
                        state_changed = True

                    if state_changed:
                        update_status_file(
                            faces_detected, fatigue_detected,
                            current_eyes_closed, current_head_tilted,
                            fatigue_percent
                        )

                    last_eyes_closed = current_eyes_closed
                    last_head_tilted = current_head_tilted

                    # --- Fatigue state change with debouncing ---
                    if fatigue_detected != last_fatigue_status:
                        if pending_state == fatigue_detected:
                            if current_time - pending_state_time >= DEBOUNCE_SECONDS:
                                if fatigue_detected:
                                    reasons = []
                                    if percent_eyes_closed >= FATIGUE_THRESHOLD:
                                        reasons.append("eyes closed")
                                    if percent_head_tilted >= FATIGUE_THRESHOLD:
                                        reasons.append("head tilted")
                                    reason_str = " / ".join(reasons)
                                    log_event(f"\nFATIGUE DETECTED ({reason_str})")
                                else:
                                    log_event("\nAttention restored - student alert")

                                last_fatigue_status = fatigue_detected
                                pending_state = None
                                pending_state_time = None

                                update_status_file(
                                    faces_detected, fatigue_detected,
                                    current_eyes_closed, current_head_tilted,
                                    fatigue_percent
                                )
                        else:
                            pending_state = fatigue_detected
                            pending_state_time = current_time
                    else:
                        pending_state = None
                        pending_state_time = None

                # Periodic status file update
                current_time = time.time()
                if current_time - last_status_update_time >= STATUS_UPDATE_INTERVAL:
                    fatigue = last_fatigue_status if last_fatigue_status is not None else False
                    eyes = last_eyes_closed if last_eyes_closed is not None else False
                    head = last_head_tilted if last_head_tilted is not None else False
                    pct = max(
                        sum(closed_eye_history) / len(closed_eye_history) if closed_eye_history else 0.0,
                        sum(head_tilted_history) / len(head_tilted_history) if head_tilted_history else 0.0,
                    )
                    update_status_file(0, fatigue, eyes, head, pct)
                    last_status_update_time = current_time

                # Periodic screenshot save + live display
                if preview_frame is not None:
                    try:
                        frame = preview_frame.getCvFrame()

                        # Save screenshot periodically
                        if current_time - last_screenshot_time >= SCREENSHOT_UPDATE_INTERVAL:
                            cv2.imwrite(str(SCREENSHOT_FILE), frame)
                            last_screenshot_time = current_time

                        # Show live video window
                        if args.display:
                            eyes = last_eyes_closed if last_eyes_closed is not None else False
                            head = last_head_tilted if last_head_tilted is not None else False
                            fatigue = last_fatigue_status if last_fatigue_status is not None else False

                            # Status text overlay
                            color = (0, 0, 255) if fatigue else (0, 255, 0)
                            status = "FATIGUED" if fatigue else "ALERT"
                            cv2.putText(frame, status, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                            if eyes:
                                cv2.putText(frame, "Eyes Closed", (10, 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            if head:
                                cv2.putText(frame, "Head Tilted", (10, 85),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                            cv2.imshow("Fatigue Detector", frame)
                            if cv2.waitKey(1) == ord('q'):
                                break
                    except Exception as e:
                        log_event(f"WARNING: Could not process frame: {e}")

                time.sleep(0.01)

    except KeyboardInterrupt:
        shutdown_msg = "Fatigue detector stopped"
        log_event(f"\n{shutdown_msg}")

    finally:
        if args.display:
            cv2.destroyAllWindows()
        # Mark as not running in status file
        update_status_file(0, False, False, False, 0.0, running=False)
        if log_file:
            log_file.close()


if __name__ == "__main__":
    run_detection()
