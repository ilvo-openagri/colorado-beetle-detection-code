"""Module to perform all AI related steps for the OpenAgri project.
"""

from .constants import *
from .model import TensorFlowVersion, ModelVersion, ModelResult
from .model import YoloModel, PipelineModel, HailoModel
from .settings import get_config
from .util.text import GREEN, RED, RESET
from pathlib import Path
import argparse
import atexit
import cv2
import logging
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import platform
import time

# Got an error that the physical cores could not be identified
# This code removes the error, but might impact performance?
if platform.system() == "Windows":
    os.environ["OMP_NUM_THREADS"] = "1"

if platform.machine() != "aarch64":
    matplotlib.use("tkagg")


ai_model = None


def exit_handler():
    """Called on exit of the python program"""
    global ai_model
    if ai_model is not None:
        ai_model.close()


def visualise(
    image_name: str,
    source_image: np.ndarray,
    output_image_path: Path,
    result: ModelResult,
):
    """Visualise the results of the model in an annotated image file.

    Args:
        image_name (str): Name of the image
        source_image (np.ndarray): Source image
        output_image_path (Path): The path to the output image
        result (ModelResult): Result of the inference
    """
    patch_colors = {64: "darkred", 128: "orange", 256: "yellow"}
    _fig, ax = plt.subplots(1, 1, figsize=(30, 15))
    ax.imshow(source_image)
    ax.set_title(f"Identified beetles on {image_name}")
    ax.axis("off")

    # Draw boxes around the identified leaves:
    if result.leaves:
        size = len(result.leaves[0].image)
        for leave in result.leaves:
            rect = patches.Rectangle(
                xy=(
                    leave.center_x - size / 2,
                    leave.center_y - size / 2,
                ),
                width=size,
                height=size,
                edgecolor="aqua",
                facecolor="aqua",
                alpha=0.25,
            )
            ax.add_patch(rect)
            if leave.annotation:
                ax.text(
                    rect.get_x(),
                    rect.get_y() + 25,
                    leave.annotation,
                    fontsize=5,
                )

    # Draw boxes around the model matches:
    for model_size, model_matches in result.beetles.items():
        for model_match in model_matches:
            rect = patches.Rectangle(
                xy=(
                    model_match.center_x - model_size / 2,
                    model_match.center_y - model_size / 2,
                ),
                width=model_size,
                height=model_size,
                facecolor=patch_colors[model_size],
                alpha=0.25,
            )
            ax.add_patch(rect)
            ax.text(
                rect.get_x(),
                rect.get_y() - 5,
                str(round(model_match.confidence, 2)),
                fontsize=5,
                color=patch_colors[model_size],
            )

    plt.savefig(output_image_path, dpi=500)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        format="[%(levelname)s %(name)s @ %(asctime)s] %(message)s", level=logging.DEBUG
    )
    logger = logging.getLogger(__name__)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("colormath").setLevel(logging.ERROR)
    logging.getLogger("h5py").setLevel(logging.ERROR)

    # Register exit handler
    atexit.register(exit_handler)

    # Parse the script arguments:
    parser = argparse.ArgumentParser(
        description="Evaluate image(s) from Colorado Beetles"
    )
    parser.add_argument(
        "--input",
        dest="input",
        metavar="path",
        required=True,
        help="The path to the image",
    )
    parser.add_argument(
        "--output",
        dest="output",
        metavar="path",
        required=False,
        default="",
        help="The path where an annotated output image can be stored.",
    )
    parser.add_argument(
        "--config",
        dest="scriptconfig",
        metavar="path",
        required=False,
        default="default",
        help="Configuration option for the evaluation script [default, raspi, pipeline]",
    )
    script_arguments = parser.parse_args()

    # Update the config_name environment variable
    os.environ["CONFIG_NAME"] = script_arguments.scriptconfig
    config = get_config()

    # Initialise and start the OpenAgri AI parsing:
    input_path = Path(script_arguments.input)
    input_image = cv2.imread(input_path.as_posix())

    # Select the model
    ai_model = None
    config.DETECTION_MODEL = ModelVersion(config.DETECTION_MODEL)
    if config.DETECTION_MODEL == ModelVersion.YOLO:
        ai_model = YoloModel(0.33)
    elif config.DETECTION_MODEL == ModelVersion.HAILO:
        ai_model = HailoModel(0.4)
    elif config.DETECTION_MODEL == ModelVersion.PIPELINE:
        ai_model = PipelineModel(0.6)

    # Perform the evaluation
    start_time = time.time()
    model_result = ai_model.evaluate(input_image)
    model_execution_time = round(time.time() - start_time, 2)
    logger.info(
        "%sEvaluation took %s seconds%s",
        GREEN if model_execution_time < 1 else RED,
        model_execution_time,
        RESET,
    )

    # Write the results to an image for visualisation:
    output_path = Path(script_arguments.output)
    if output_path.name:
        visualise(
            image_name=input_path.name,
            source_image=cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
            output_image_path=output_path,
            result=model_result,
        )
        logger.info("Written to %s", output_path)
