"""Module to perform all AI related steps for the OpenAgri project.
"""

from .constants import *
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, sRGBColor
from enum import Enum
from pathlib import Path
from skimage.color import deltaE_ciede2000
from sklearn.cluster import KMeans
from ultralytics import YOLO
import argparse
import cv2
import keras
import logging
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import platform
import tensorflow as tf
import time
import typing

# Got an error that the physical cores could not be identified
# This code removes the error, but might impact performance?
if platform.system() == "Windows":
    os.environ["OMP_NUM_THREADS"] = "1"


class ModelVersion(Enum):
    """Versions of the models to parse 256x256 image parts"""

    PIPELINE = "PIPELINE"
    YOLO = "YOLO"


class TensorFlowVersion(Enum):
    """Tensorflow versions that can be selected for the model"""

    DEFAULT = "default"
    LITE = "lite"


class OpenAgriAI:
    """Class to perform all AI related steps for the OpenAgri project."""

    def __init__(
        self, model_version: ModelVersion, tensorflow_version: TensorFlowVersion
    ):
        self.logger = logging.getLogger("OpenAgriAIModule")
        self.logger.setLevel(logging.INFO)

        self._model_version = model_version
        self._tensorflow_version = tensorflow_version
        if self._model_version == ModelVersion.PIPELINE:
            if self._tensorflow_version == TensorFlowVersion.DEFAULT:
                self._pipeline_models = {
                    256: keras.models.load_model(MODEL_256),
                    128: keras.models.load_model(MODEL_128),
                    64: keras.models.load_model(MODEL_64),
                }
                self._pipeline_models_sizes = [256, 128, 64]
            elif self._tensorflow_version == TensorFlowVersion.LITE:
                self._pipeline_lite_interpreters = {
                    256: tf.lite.Interpreter(MODEL_LITE_256),
                    128: tf.lite.Interpreter(MODEL_LITE_128),
                    64: tf.lite.Interpreter(MODEL_LITE_64),
                }
                self._pipeline_models_sizes = [256, 128, 64]
                for (
                    pipeline_lite_interpreter
                ) in self._pipeline_lite_interpreters.values():
                    pipeline_lite_interpreter.allocate_tensors()
            self._pipeline_destination_size = self._pipeline_models_sizes[-1]
        elif self._model_version == ModelVersion.YOLO:
            if self._tensorflow_version == TensorFlowVersion.DEFAULT:
                self._yolo_model = YOLO(MODEL_YOLO, task="detect")
            else:
                self._yolo_model = YOLO(MODEL_YOLO_TFLITE, task="detect")

        self._gpu_workload_callback = None

    def _visualise(
        self,
        image_name: str,
        source_image: np.ndarray,
        output_image_path: Path,
        leave_images: list,
        model_images: dict,
    ):
        """Visualise the results of the model in an annotated image file.

        Args:
            image_name (str): Name of the image
            source_image (np.ndarray): Source image
            output_image_path (Path): The path to the output image
            leave_images (list): Parts of the image where leaves were detected
            model_images (dict): Parts of the image where the pipeline detected Colorado beetles
        """
        patch_colors = {64: "darkred", 128: "orange", 256: "yellow"}
        _fig, ax = plt.subplots(1, 1, figsize=(30, 15))
        ax.imshow(source_image)
        ax.set_title(f"Identified beetles on {image_name}")
        ax.axis("off")

        # Draw boxes around the identified leaves:
        size = len(leave_images[0]["image"])
        for leave_image in leave_images:
            rect = patches.Rectangle(
                xy=(
                    leave_image["center"]["x"] - size / 2,
                    leave_image["center"]["y"] - size / 2,
                ),
                width=size,
                height=size,
                edgecolor="aqua",
                facecolor="aqua",
                alpha=0.25,
            )
            ax.add_patch(rect)

        # Draw boxes around the model matches:
        for model_size, model_matches in model_images.items():
            for model_match in model_matches:
                rect = patches.Rectangle(
                    xy=(
                        model_match["center"]["x"] - model_size / 2,
                        model_match["center"]["y"] - model_size / 2,
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
                    str(round(model_match["confidence"], 2)),
                    fontsize=5,
                    color=patch_colors[model_size],
                )

        plt.savefig(output_image_path, dpi=500)

    def _split_image(
        self,
        source_image: np.ndarray,
        piece_size: int,
        x_offset: int = 0,
        y_offset: int = 0,
        margin: int = 0,
    ) -> list:
        """Split image into equal sized pieces

        Args:
            source_image (np.ndarray): Source image
            piece_size (int): Dimension of each piece
            x_offset (int): X offset to the center of the piece
            y_offset (int): Y offset to the center of the piece
            margin (int): Some margin around the image

        Returns:
            list: The equal sized pieces in which the image was split.
        """

        pieces = []
        y = 0

        x_max_dim = len(source_image[0])
        y_max_dim = len(source_image)
        self.logger.debug("Dimensions: %d/%d", x_max_dim, y_max_dim)

        # Remove edges if deemed too small
        while y < y_max_dim:
            x = 0
            # Remove edges if deemed too small
            while x < x_max_dim:
                x_end = x + piece_size if x + piece_size < x_max_dim else x_max_dim
                y_end = y + piece_size if y + piece_size < y_max_dim else y_max_dim

                # Cut out a piece
                img_piece = source_image[
                    max(y - margin, 0) : y_end + margin,
                    max(x - margin, 0) : x_end + margin,
                ]
                pieces.append(
                    {
                        "image": img_piece,
                        "center": {
                            "x": x + piece_size / 2 + x_offset,
                            "y": y + piece_size / 2 + y_offset,
                        },
                        "confidence": 0,
                    }
                )
                x += piece_size
            y += piece_size
        if len(pieces) != 4:
            self.logger.error("Unexpected image split size of %d", len(pieces))

        return pieces

    def _filter_kmeans(self, source_image: np.ndarray) -> list[np.ndarray]:
        """Do a kmeans classification to identify the leaves vs soil

        Args:
            source_image (np.ndarray): Source image

        Returns:
            list[np.ndarray]: Matching image fragments of 512x512
        """
        # Split the image into its RGB components
        b, g, r = cv2.split(source_image)

        # Convert channels to float to avoid integer division issues
        r = r.astype(np.float32)
        g = g.astype(np.float32)
        b = b.astype(np.float32)

        # Avoid division by zero by setting zero values in g to a small number
        g[g == 0] = 1
        # Apply the formula (r + b) / g
        result = (r + b) / g

        # Convert the result back to uint8

        resized_image = cv2.resize(
            result,
            (result.shape[1] // 256, result.shape[0] // 256),
            interpolation=cv2.INTER_AREA,
        )

        (h, w) = resized_image.shape
        flat_image = resized_image.reshape(h * w, 1)
        kmeans_model = KMeans(n_clusters=2)
        cluster_labels = kmeans_model.fit_predict(flat_image)
        clustered_image = np.reshape(cluster_labels, (h, w))

        # Determine the cluster with a green color:
        green_cluster = np.argmin(kmeans_model.cluster_centers_)

        # Create the resultset
        result_set = []
        for y_index, _ in enumerate(clustered_image):
            for x_index, classification in enumerate(clustered_image[y_index]):
                if classification != green_cluster:
                    continue
                x_offset = x_index * 256
                y_offset = y_index * 256
                result_set.append(
                    {
                        "image": source_image[
                            y_offset : y_offset + 256, x_offset : x_offset + 256
                        ],
                        "center": {"x": x_offset + 128, "y": y_offset + 128},
                        "confidence": 0,
                    }
                )

        return result_set

    def _filter_pipeline(self, source_image: np.ndarray) -> bool:
        """Filter the image based on the pipeline provided through the thesis of Robin

        Args:
            source_image (np.ndarray): Source image

        Returns:
            bool: Return if the image matches the pipeline stage
        """
        filter_result = 0
        size = len(source_image)
        start_pred = time.time()
        reshaped_image = tf.reshape(source_image, [1, size, size, 3])
        if self._tensorflow_version == TensorFlowVersion.DEFAULT:
            prediction = self._pipeline_models[size].predict(reshaped_image, verbose=0)
            filter_result = prediction[0][1]
        elif self._tensorflow_version == TensorFlowVersion.LITE:
            interpreter = self._pipeline_lite_interpreters[size]
            interpreter.set_tensor(
                tensor_index=interpreter.get_input_details()[0]["index"],
                value=reshaped_image,
            )
            interpreter.invoke()
            filter_result = interpreter.get_tensor(
                tensor_index=interpreter.get_output_details()[0]["index"]
            )
            filter_result = filter_result[0][1]

        pred_time = time.time() - start_pred

        return filter_result

    def yolo_mp_worker(
        self,
        _worker_index: int,
        model: str,
        confidence: float,
        task_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
    ):
        yolo_model = YOLO(model, task="detect")
        while True:
            # Get a new task (an image) from the task queue
            image = task_queue.get()

            if image is None:  # If `None` is received, terminate the worker
                break

            yolo_match = yolo_model.predict(
                image["image"], conf=confidence, verbose=False
            )
            yolo_results = []
            for match in yolo_match:
                if match.summary():
                    x, y, w, h = match.boxes.xywh[0]
                    yolo_results.append(
                        {
                            "x": image["center"]["x"] - 128 + int(x),
                            "y": image["center"]["y"] - 128 + int(y),
                            "w": int(w),
                            "h": int(h),
                            "confidence": float(match.boxes.conf[0]),
                        }
                    )
            result_queue.put(yolo_results)
        result_queue.put(None)
        while result_queue.qsize():
            time.sleep(0.1)
        return 0

    def register_gpu_workload_callback(self, callback: typing.Callable):
        """Register a callback which is called when starting and stopping a GPU workload

        Args:
            callback (typing.Callacble): The callback function
        """
        self._gpu_workload_callback = callback

    def evaluate(
        self, image_name: str, image_file: cv2.typing.MatLike, output_path: Path
    ):
        """Evaluate the images from the file path against the openagri AI model

        Args:
            image_name (str): Name of the source image.
            image_file (cv2.typing.MatLike): Source image
            output_path (Path): Path to the output path, where an annotated image will be stored.
        """
        self.logger.info("Start evaluation")

        # Crop the image:
        x_size = image_file.shape[0] // 256
        y_size = image_file.shape[1] // 256
        self.logger.info("Crop image to %dx%d", x_size * 256, y_size * 256)
        image_file = image_file[: x_size * 256, : y_size * 256]

        # Filter out the leaves through kmeans clustering:
        self.logger.info("Start kmeans filtering to extract the leaves")
        leave_images = self._filter_kmeans(image_file)
        self.logger.info(
            "Kmeans matches %d out of %d image pieces",
            len(leave_images),
            x_size * y_size,
        )

        active_image_set = leave_images
        matching_image_set = []
        matching_images_per_resolution = {}
        if self._model_version == ModelVersion.PIPELINE:
            # We do the beetle matching based on three different models:
            # - input: 256x256 images
            # - input: 128x128 images
            # - input: 64x64 images
            # Only the pieces that passed the previous stage, are parsed to the next model.
            # The result will be the matching 64x64 pieces matching the Colorado Beetle.
            # Do some preprocessing:
            for active_image in active_image_set:
                active_image["image"] = (
                    np.array(active_image["image"], dtype=np.float32) / 255
                )

            if self._gpu_workload_callback is not None:
                self._gpu_workload_callback(True)
            for pipeline_model in self._pipeline_models_sizes:
                matching_image_set = []
                self.logger.info("Starting pipeline")
                for active_image in active_image_set:
                    pipeline_confidence = self._filter_pipeline(active_image["image"])
                    if pipeline_confidence < 0.5:
                        continue
                    active_image["confidence"] = pipeline_confidence
                    matching_image_set.append(active_image)
                self.logger.info(
                    "Pipeline matches %d out of %d image pieces (size %d)",
                    len(matching_image_set),
                    len(active_image_set) * 4,
                    pipeline_model,
                )

                active_image_set = matching_image_set
                matching_images_per_resolution[pipeline_model] = matching_image_set

                # Iteration completed. Do some post=processing for next iteration.
                if pipeline_model != self._pipeline_destination_size:
                    active_image_set_split = []
                    for active_image in active_image_set:
                        active_image_set_split.extend(
                            self._split_image(
                                source_image=active_image["image"],
                                piece_size=pipeline_model // 2,
                                x_offset=active_image["center"]["x"]
                                - pipeline_model // 2,
                                y_offset=active_image["center"]["y"]
                                - pipeline_model // 2,
                            )
                        )
                    active_image_set = active_image_set_split
            if self._gpu_workload_callback is not None:
                self._gpu_workload_callback(False)
        elif self._model_version == ModelVersion.YOLO:
            matching_images_per_resolution[64] = []
            task_queue = multiprocessing.Queue()
            result_queue = multiprocessing.Queue()
            workers = []
            PROCESS_COUNT = 4
            for worker_index in range(PROCESS_COUNT):
                p = multiprocessing.Process(
                    target=self.yolo_mp_worker,
                    args=(
                        worker_index,
                        MODEL_YOLO_TFLITE,
                        0.33,
                        task_queue,
                        result_queue,
                    ),
                )
                p.start()
                workers.append(p)

            for active_image in active_image_set:
                if self._gpu_workload_callback is not None:
                    self._gpu_workload_callback(True)
                task_queue.put(active_image)
                if self._gpu_workload_callback is not None:
                    self._gpu_workload_callback(False)
            self.logger.info("Waiting for workers to finish")
            # Stop all workers by sending `None` to each
            for _ in workers:
                task_queue.put(None)

            # Collect the results
            processes_finished = 0
            results = []
            while processes_finished < PROCESS_COUNT:
                result = result_queue.get()
                if result is None:
                    processes_finished += 1
                    continue
                results.extend(result)

            task_queue.close()
            task_queue.join_thread()
            result_queue.close()
            result_queue.join_thread()

            for p in workers:
                p.join()

            for result in results:
                matching_images_per_resolution[64].append(
                    {
                        # "image": None,
                        "center": {
                            "x": result["x"],
                            "y": result["y"],
                        },
                        "confidence": result["confidence"],
                    }
                )

        self.logger.info("Completed")
        if output_path.name:
            self._visualise(
                image_name=image_name,
                source_image=image_file,
                output_image_path=output_path,
                leave_images=leave_images,
                model_images=matching_images_per_resolution,
            )
            self.logger.info("Written to %s", output_path)
        else:
            self.logger.error("No output name: %s", output_path)

        return matching_images_per_resolution[64]


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        format="[%(levelname)s %(name)s @ %(asctime)s] %(message)s", level=logging.DEBUG
    )
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("colormath").setLevel(logging.ERROR)

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
        "--tensorflow",
        dest="tensorflow",
        metavar="path",
        required=False,
        choices=[TensorFlowVersion.DEFAULT.value, TensorFlowVersion.LITE.value],
        default=TensorFlowVersion.DEFAULT.value,
        help="Select the tensorflow version (lite or default)",
    )
    parser.add_argument(
        "--model",
        dest="model",
        required=False,
        choices=[ModelVersion.PIPELINE.value, ModelVersion.YOLO.value],
        default=ModelVersion.PIPELINE.value,
        help="The type of model used on the 256x256 images of the crops.",
    )
    script_arguments = parser.parse_args()

    # Initialise and start the OpenAgri AI module:
    ai_module = OpenAgriAI(
        ModelVersion(script_arguments.model),
        TensorFlowVersion(script_arguments.tensorflow),
    )
    input_image = cv2.imread(Path(script_arguments.input))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    ai_module.evaluate(
        image_name="",
        image_file=input_image,
        output_path=Path(script_arguments.output),
    )
