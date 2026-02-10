"""
Node Creators for Gaze Estimation Pipeline
===========================================
Helper functions to create crop nodes using Script + ImageManip pairs.

Adapted from Luxonis oak-examples gaze estimation.
"""

import depthai as dai
from pathlib import Path


def create_crop_node(
    pipeline: dai.Pipeline,
    input_frame: dai.Node.Output,
    configs_message: dai.Node.Output,
) -> dai.node.ImageManip:
    """Create a Script + ImageManip pair for cropping regions from frames.

    The Script node iterates over MessageGroup configs (one per detection)
    and sends each config + frame pair to the ImageManip for cropping.

    Args:
        pipeline: The DepthAI pipeline
        input_frame: Camera output to crop from
        configs_message: MessageGroup of ImageManipConfig crops

    Returns:
        ImageManip node whose .out produces cropped regions
    """
    script_path = Path(__file__).parent / "config_sender_script.py"
    with script_path.open("r") as script_file:
        script_content = script_file.read()

    config_sender_script = pipeline.create(dai.node.Script)
    config_sender_script.setScript(script_content)
    config_sender_script.inputs["frame_input"].setBlocking(True)
    config_sender_script.inputs["config_input"].setBlocking(True)

    img_manip_node = pipeline.create(dai.node.ImageManip)
    img_manip_node.initialConfig.setReusePreviousImage(False)
    img_manip_node.inputConfig.setReusePreviousMessage(False)
    img_manip_node.inputImage.setReusePreviousMessage(False)
    img_manip_node.inputConfig.setBlocking(True)
    img_manip_node.inputImage.setBlocking(True)

    input_frame.link(config_sender_script.inputs["frame_input"])
    configs_message.link(config_sender_script.inputs["config_input"])

    config_sender_script.outputs["output_config"].link(img_manip_node.inputConfig)
    config_sender_script.outputs["output_frame"].link(img_manip_node.inputImage)

    return img_manip_node
