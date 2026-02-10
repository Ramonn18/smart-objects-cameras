#!/usr/bin/env python3
"""
Gaze Direction Detector for OAK-D (DepthAI 3.x)
=================================================
Three-stage pipeline: YuNet face detection + head pose estimation
+ gaze estimation ADAS. Detects where a person is looking.

Adapted from Luxonis oak-examples gaze estimation.

Usage:
    python3 gaze_detector.py                    # Basic detection
    python3 gaze_detector.py --display          # Show live video with gaze vectors
    python3 gaze_detector.py --log              # Log to file
"""

from pathlib import Path
from datetime import datetime
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, GatherData
import argparse
import time
import json
import cv2
import numpy as np

from utils.process_keypoints import LandmarksProcessing
from utils.node_creators import create_crop_node
from utils.host_concatenate_head_pose import ConcatenateHeadPose

# Parse arguments
parser = argparse.ArgumentParser(
    description='OAK-D Gaze Direction Detector (DepthAI 3.x)')
parser.add_argument('--log', action='store_true', help='Log events to file')
parser.add_argument('--fps-limit', type=int, default=None,
                    help='FPS limit (default: 15 for RVC2, 30 for RVC4)')
parser.add_argument('--device', type=str, default=None,
                    help='Optional DeviceID or IP of the camera')
parser.add_argument('--display', action='store_true',
                    help='Show live video window with gaze vectors (requires display)')
args = parser.parse_args()

# Requested camera resolution
REQ_WIDTH, REQ_HEIGHT = 640, 480

# Global state
log_file = None

# Status file for integration
STATUS_FILE = Path.home() / "oak-projects" / "gaze_status.json"
STATUS_UPDATE_INTERVAL = 2
last_status_update_time = 0

# Screenshot
SCREENSHOT_FILE = Path.home() / "oak-projects" / "latest_gaze_frame.jpg"
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


def classify_gaze(gaze_x, gaze_y):
    """Classify gaze vector into a human-readable direction.

    Args:
        gaze_x: Horizontal gaze component (positive = right)
        gaze_y: Vertical gaze component (positive = up)

    Returns:
        Direction string like "center", "left", "right", "up", "down",
        or combinations like "up-left".
    """
    threshold = 0.15

    h_dir = ""
    v_dir = ""

    if gaze_x < -threshold:
        h_dir = "left"
    elif gaze_x > threshold:
        h_dir = "right"

    if gaze_y > threshold:
        v_dir = "up"
    elif gaze_y < -threshold:
        v_dir = "down"

    if h_dir and v_dir:
        return f"{v_dir}-{h_dir}"
    elif h_dir:
        return h_dir
    elif v_dir:
        return v_dir
    else:
        return "center"


def update_status_file(faces_detected, gaze_direction, gaze_x, gaze_y, gaze_z,
                       head_yaw, head_pitch, head_roll, running=True):
    """Update status file for external integration."""
    try:
        status_data = {
            "faces_detected": faces_detected,
            "gaze_direction": gaze_direction,
            "gaze_x": round(float(gaze_x), 4),
            "gaze_y": round(float(gaze_y), 4),
            "gaze_z": round(float(gaze_z), 4),
            "head_yaw": round(float(head_yaw), 1),
            "head_pitch": round(float(head_pitch), 1),
            "head_roll": round(float(head_roll), 1),
            "timestamp": datetime.now().isoformat(),
            "running": running,
        }
        STATUS_FILE.write_text(json.dumps(status_data, indent=2))
    except Exception as e:
        log_event(f"WARNING: Could not update status file: {e}")


def draw_gaze_vector(frame, eye_x, eye_y, gaze_vector, src_w, src_h, color=(0, 255, 0)):
    """Draw a gaze direction arrow on the frame.

    Args:
        frame: OpenCV frame to draw on
        eye_x, eye_y: Normalized eye position (0-1)
        gaze_vector: Raw 3D gaze vector from the model
        src_w, src_h: Source frame dimensions
        color: Arrow color in BGR
    """
    # Scale gaze vector to pixel coordinates
    gaze_scaled = (gaze_vector * 640)[:2]
    start = (int(eye_x * src_w), int(eye_y * src_h))
    end = (
        int(start[0] + gaze_scaled[0]),
        int(start[1] - gaze_scaled[1]),  # Y is inverted
    )
    cv2.arrowedLine(frame, start, end, color, 2, tipLength=0.3)


def run_detection():
    """Main gaze detection loop using DepthAI 3.x three-stage pipeline."""
    global log_file, last_status_update_time, last_screenshot_time

    if args.log:
        log_filename = f"gaze_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file = open(log_filename, 'w')
        log_event(f"Logging to {log_filename}")

    log_event("Gaze detector started (DepthAI 3.x, YuNet + Head Pose + Gaze ADAS)")
    log_event("Press Ctrl+C to exit (or 'q' in display window)\n")

    # Initialize status file
    update_status_file(0, "unknown", 0, 0, 0, 0, 0, 0, running=True)
    last_status_update_time = time.time()

    # Track last gaze for console output when no faces detected
    last_gaze_direction = "unknown"
    last_gaze_x = 0.0
    last_gaze_y = 0.0
    last_gaze_z = 0.0
    last_head_yaw = 0.0
    last_head_pitch = 0.0
    last_head_roll = 0.0

    try:
        # Connect to device
        if args.device:
            device = dai.Device(dai.DeviceInfo(args.device))
        else:
            device = dai.Device()
        platform = device.getPlatform().name
        log_event(f"Connected to device: {device.getDeviceId()}")
        log_event(f"Platform: {platform}")

        fps_limit = args.fps_limit
        if fps_limit is None:
            fps_limit = 15 if platform == "RVC2" else 30
            log_event(f"FPS limit set to {fps_limit} for {platform}")

        frame_type = (
            dai.ImgFrame.Type.BGR888i if platform == "RVC4"
            else dai.ImgFrame.Type.BGR888p
        )

        with dai.Pipeline(device) as pipeline:
            log_event("Creating pipeline...")

            models_dir = Path(__file__).parent / "depthai_models"

            # --- Stage 1: Face Detection (YuNet 320x240) ---
            det_model_description = dai.NNModelDescription.fromYamlFile(
                str(models_dir / f"yunet_gaze.{platform}.yaml")
            )
            det_model_nn_archive = dai.NNArchive(
                dai.getModelFromZoo(det_model_description)
            )

            # --- Stage 2: Head Pose Estimation ---
            head_pose_model_description = dai.NNModelDescription.fromYamlFile(
                str(models_dir / f"head_pose_estimation.{platform}.yaml")
            )
            head_pose_model_description.platform = platform
            head_pose_model_nn_archive = dai.NNArchive(
                dai.getModelFromZoo(head_pose_model_description)
            )

            # --- Stage 3: Gaze Estimation ADAS ---
            gaze_model_description = dai.NNModelDescription.fromYamlFile(
                str(models_dir / f"gaze_estimation_adas.{platform}.yaml")
            )
            gaze_model_description.platform = platform
            gaze_model_nn_archive = dai.NNArchive(
                dai.getModelFromZoo(gaze_model_description)
            )

            # --- Camera input ---
            cam = pipeline.create(dai.node.Camera).build()
            cam_out = cam.requestOutput(
                size=(REQ_WIDTH, REQ_HEIGHT), type=frame_type, fps=fps_limit
            )

            # Resize to face detection model input size
            resize_node = pipeline.create(dai.node.ImageManip)
            resize_node.initialConfig.setOutputSize(
                det_model_nn_archive.getInputWidth(),
                det_model_nn_archive.getInputHeight(),
            )
            resize_node.setMaxOutputFrameSize(
                det_model_nn_archive.getInputWidth()
                * det_model_nn_archive.getInputHeight() * 3
            )
            resize_node.initialConfig.setReusePreviousImage(False)
            resize_node.inputImage.setBlocking(True)
            cam_out.link(resize_node.inputImage)

            # Face detection NN
            det_nn = pipeline.create(ParsingNeuralNetwork).build(
                resize_node.out, det_model_nn_archive
            )
            det_nn.input.setBlocking(True)

            # Detection processing: extract eye and face crop configs
            detection_process_node = pipeline.create(LandmarksProcessing)
            detection_process_node.set_source_size(REQ_WIDTH, REQ_HEIGHT)
            detection_process_node.set_target_size(
                head_pose_model_nn_archive.getInputWidth(),
                head_pose_model_nn_archive.getInputHeight(),
            )
            det_nn.out.link(detection_process_node.detections_input)

            # Crop nodes for left eye, right eye, and face
            left_eye_crop_node = create_crop_node(
                pipeline, cam_out, detection_process_node.left_config_output
            )
            right_eye_crop_node = create_crop_node(
                pipeline, cam_out, detection_process_node.right_config_output
            )
            face_crop_node = create_crop_node(
                pipeline, cam_out, detection_process_node.face_config_output
            )

            # Head pose estimation NN
            head_pose_nn = pipeline.create(ParsingNeuralNetwork).build(
                face_crop_node.out, head_pose_model_nn_archive
            )
            head_pose_nn.input.setBlocking(True)

            # Concatenate yaw/pitch/roll into single tensor
            head_pose_concatenate_node = pipeline.create(ConcatenateHeadPose).build(
                head_pose_nn.getOutput(0),
                head_pose_nn.getOutput(1),
                head_pose_nn.getOutput(2),
            )

            # Gaze estimation NN (multi-input: left eye, right eye, head pose)
            gaze_estimation_node = pipeline.create(dai.node.NeuralNetwork)
            gaze_estimation_node.setNNArchive(gaze_model_nn_archive)
            head_pose_concatenate_node.output.link(
                gaze_estimation_node.inputs["head_pose_angles_yaw_pitch_roll"]
            )
            left_eye_crop_node.out.link(
                gaze_estimation_node.inputs["left_eye_image"]
            )
            right_eye_crop_node.out.link(
                gaze_estimation_node.inputs["right_eye_image"]
            )
            gaze_estimation_node.inputs["head_pose_angles_yaw_pitch_roll"].setBlocking(True)
            gaze_estimation_node.inputs["left_eye_image"].setBlocking(True)
            gaze_estimation_node.inputs["right_eye_image"].setBlocking(True)
            gaze_estimation_node.inputs["left_eye_image"].setMaxSize(5)
            gaze_estimation_node.inputs["right_eye_image"].setMaxSize(5)
            gaze_estimation_node.inputs["head_pose_angles_yaw_pitch_roll"].setMaxSize(5)

            # Sync detections with gaze estimations
            gather_data_node = pipeline.create(GatherData).build(fps_limit)
            gaze_estimation_node.out.link(gather_data_node.input_data)
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
            log_event("Detection started. Monitoring gaze direction...\n")

            while pipeline.isRunning():
                gather_msg = q_gather.tryGet()
                preview_frame = q_preview.tryGet()

                if gather_msg is not None:
                    from depthai_nodes import ImgDetectionsExtended

                    detections_msg = gather_msg.reference_data
                    gaze_list = gather_msg.gathered
                    src_w, src_h = detections_msg.transformation.getSize()

                    faces_detected = len(detections_msg.detections)

                    # Process first detected face for status
                    if faces_detected > 0 and len(gaze_list) > 0:
                        detection = detections_msg.detections[0]
                        gaze_data = gaze_list[0]

                        gaze_tensor = gaze_data.getFirstTensor(dequantize=True).flatten()
                        gaze_x = float(gaze_tensor[0])
                        gaze_y = float(gaze_tensor[1])
                        gaze_z = float(gaze_tensor[2]) if len(gaze_tensor) > 2 else 0.0

                        # Extract head pose from the concatenated tensor
                        # The head pose values were sent through the pipeline
                        # We can get approximate values from the gaze vector direction
                        # For accurate values, we'd need a separate queue
                        head_yaw = last_head_yaw
                        head_pitch = last_head_pitch
                        head_roll = last_head_roll

                        gaze_direction = classify_gaze(gaze_x, gaze_y)

                        last_gaze_direction = gaze_direction
                        last_gaze_x = gaze_x
                        last_gaze_y = gaze_y
                        last_gaze_z = gaze_z

                        # Console status line
                        print(
                            f"\r  Faces: {faces_detected} | "
                            f"Gaze: {gaze_direction:>10} "
                            f"(x:{gaze_x:+.2f} y:{gaze_y:+.2f} z:{gaze_z:+.2f})  ",
                            end="", flush=True
                        )
                    else:
                        print(
                            f"\r  Faces: 0 | Gaze: --           "
                            f"                              ",
                            end="", flush=True
                        )

                    # Update status file periodically
                    current_time = time.time()
                    if current_time - last_status_update_time >= STATUS_UPDATE_INTERVAL:
                        update_status_file(
                            faces_detected, last_gaze_direction,
                            last_gaze_x, last_gaze_y, last_gaze_z,
                            last_head_yaw, last_head_pitch, last_head_roll,
                        )
                        last_status_update_time = current_time

                # Screenshot and display
                if preview_frame is not None:
                    try:
                        frame = preview_frame.getCvFrame()
                        current_time = time.time()

                        # Save screenshot periodically
                        if current_time - last_screenshot_time >= SCREENSHOT_UPDATE_INTERVAL:
                            cv2.imwrite(str(SCREENSHOT_FILE), frame)
                            last_screenshot_time = current_time

                        # Live display with gaze vectors
                        if args.display and gather_msg is not None:
                            detections_msg = gather_msg.reference_data
                            gaze_list = gather_msg.gathered
                            src_w, src_h = detections_msg.transformation.getSize()

                            for detection, gaze_data in zip(
                                detections_msg.detections, gaze_list
                            ):
                                keypoints = detection.keypoints
                                gaze_tensor = gaze_data.getFirstTensor(
                                    dequantize=True
                                ).flatten()

                                # Draw face bounding box
                                bbox = detection.rotated_rect.getPoints()
                                pts = np.array(
                                    [[int(p.x * src_w), int(p.y * src_h)] for p in bbox],
                                    dtype=np.int32,
                                )
                                cv2.polylines(frame, [pts], True, (255, 255, 0), 2)

                                # Draw gaze vectors from both eyes
                                if len(keypoints) >= 2:
                                    # Left eye (keypoints[1] in YuNet)
                                    draw_gaze_vector(
                                        frame,
                                        keypoints[1].x, keypoints[1].y,
                                        gaze_tensor, src_w, src_h,
                                        color=(0, 255, 0),
                                    )
                                    # Right eye (keypoints[0] in YuNet)
                                    draw_gaze_vector(
                                        frame,
                                        keypoints[0].x, keypoints[0].y,
                                        gaze_tensor, src_w, src_h,
                                        color=(0, 255, 0),
                                    )

                            # Gaze direction text overlay
                            direction = last_gaze_direction.upper()
                            color = (0, 255, 0) if direction == "CENTER" else (0, 165, 255)
                            cv2.putText(
                                frame, f"Gaze: {direction}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
                            )

                        if args.display:
                            cv2.imshow("Gaze Detector", frame)
                            if cv2.waitKey(1) == ord('q'):
                                break

                    except Exception as e:
                        log_event(f"WARNING: Could not process frame: {e}")

                time.sleep(0.01)

    except KeyboardInterrupt:
        log_event(f"\nGaze detector stopped")

    finally:
        if args.display:
            cv2.destroyAllWindows()
        update_status_file(0, "unknown", 0, 0, 0, 0, 0, 0, running=False)
        if log_file:
            log_file.close()


if __name__ == "__main__":
    run_detection()
