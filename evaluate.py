"""Module to perform all AI related steps for the OpenAgri project.
"""

import os
import argparse
import platform
import logging
import time
from enum import Enum
from pathlib import Path
from sklearn.cluster import KMeans
from skimage.color import deltaE_ciede2000
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, sRGBColor
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec
import keras
import constants

# Got an error that the physical cores could not be identified
# This code removes the error, but might impact performance?
if platform.system() == 'Windows':
    os.environ['OMP_NUM_THREADS'] = '1'


class TensorFlowVersion(Enum):
    """Tensorflow versions that can be selected for the model
    """
    DEFAULT = "default"
    LITE = "lite"


class OpenAgriAI:
    """Class to perform all AI related steps for the OpenAgri project.
    """

    def __init__(self, tensorflow_version: TensorFlowVersion):
        self.logger = logging.getLogger("OpenAgriAIModule")
        self.logger.setLevel(logging.INFO)
        self._tensorflow_version = tensorflow_version
        if self._tensorflow_version == TensorFlowVersion.DEFAULT:
            self._pipeline_models = {
                256: keras.models.load_model(constants.MODEL_256),
                128: keras.models.load_model(constants.MODEL_128),
                # 64: keras.models.load_model(constants.MODEL_64),
            }
            self._pipeline_models_sizes = [256, 128]
            self._pipeline_destination_size = 128
        elif self._tensorflow_version == TensorFlowVersion.LITE:
            self._pipeline_lite_interpreters = {
                256: tf.lite.Interpreter(constants.MODEL_LITE_256),
                128: tf.lite.Interpreter(constants.MODEL_LITE_128),
                64: tf.lite.Interpreter(constants.MODEL_LITE_64),
            }
            self._pipeline_models_sizes = [256, 128, 64]
            self._pipeline_destination_size = 64
            for pipeline_lite_interpreter in self._pipeline_lite_interpreters.values():
                pipeline_lite_interpreter.allocate_tensors()

    def _visualise(self,
                   image_name: str,
                   source_image: np.ndarray,
                   output_image_path: Path,
                   leave_images: list,
                   model_images: dict):
        """Visualise the results of the model in an annotated image file.

        Args:
            image_name (str): Name of the image
            source_image (np.ndarray): Source image
            output_image_path (Path): The path to the output image
            leave_images (list): Parts of the image where leaves were detected
            model_images (dict): Parts of the image where the pipeline detected Colorado beetles
        """
        patch_colors = {
            64: "darkred",
            128: "orange",
            256: "yellow"
        }
        _fig, ax = plt.subplots(1, 1, figsize=(30, 15))
        ax.imshow(source_image)
        ax.set_title(f'Identified beetles on {image_name}')
        ax.axis('off')

        # Draw boxes around the identified leaves:
        size = len(leave_images[0]['image'])
        for leave_image in leave_images:
            rect = patches.Rectangle(
                xy=(
                    leave_image['center']['x'] - size / 2,
                    leave_image['center']['y'] - size / 2,
                ),
                width=size,
                height=size,
                edgecolor='aqua',
                facecolor='aqua',
                alpha=0.25
            )
            ax.add_patch(rect)

        # Draw boxes around the model matches:
        for model_size, model_matches in model_images.items():
            for model_match in model_matches:
                rect = patches.Rectangle(
                    xy=(
                        model_match['center']['x'] - model_size / 2,
                        model_match['center']['y'] - model_size / 2,
                    ),
                    width=model_size,
                    height=model_size,
                    edgecolor=patch_colors[model_size],
                    facecolor=patch_colors[model_size],
                    alpha=0.25
                )
                ax.add_patch(rect)

        plt.savefig(output_image_path, dpi=500)

    def _split_image(self, source_image: np.ndarray, piece_size: int,
                     x_offset: int = 0, y_offset: int = 0) -> list:
        """Split image into equal sized pieces

        Args:
            source_image (np.ndarray): Source image
            piece_size (int): Dimension of each piece
            x_offset (int): X offset to the center of the piece
            y_offset (int): Y offset to the center of the piece

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
                x_end = x + piece_size if x + \
                    piece_size < x_max_dim else x_max_dim
                y_end = y + piece_size if y + \
                    piece_size < y_max_dim else y_max_dim

                # Cut out a piece
                img_piece = source_image[y:y_end, x:x_end]
                pieces.append({
                    'image': img_piece,
                    'center': {
                        'x': x + piece_size / 2 + x_offset,
                        'y': y + piece_size / 2 + y_offset
                    }
                })
                x += piece_size
            y += piece_size
        if len(pieces) != 4:
            self.logger.info(len(pieces))

        return pieces

    def _filter_kmeans(self, source_image: np.ndarray) -> list[np.ndarray]:
        """Do a kmeans classification to identify the leaves vs soil

        Args:
            source_image (np.ndarray): Source image

        Returns:
            list[np.ndarray]: Matching image fragments of 512x512
        """
        # Reshape the input image. Each pixel will represent on 512x512 piece of the source image
        start_reshape = time.time()
        resized_image = cv2.resize(
            source_image,
            (
                source_image.shape[1] // 256,
                source_image.shape[0] // 256
            ),
            interpolation=cv2.INTER_AREA
        )
        (h, w, c) = resized_image.shape
        flat_image = resized_image.reshape(h*w, c)
        reshape_time = time.time() - start_reshape

        # Predict the parts the kmeans classification:
        kmeans_model = KMeans(n_clusters=2)
        cluster_labels = kmeans_model.fit_predict(flat_image)
        color_columns = kmeans_model.cluster_centers_.round(0).astype(int)
        clustered_image = np.reshape(cluster_labels, (h, w))

        # Determine the cluster with a green color:
        lab_green = convert_color(
            sRGBColor(0, 128, 0), LabColor).get_value_tuple()
        lab_cluster_colors = [
            convert_color(sRGBColor(*rgb_color), LabColor).get_value_tuple()
            for rgb_color in color_columns
        ]
        delta_e_values = np.array(
            [deltaE_ciede2000(lab_color, lab_green) for lab_color in lab_cluster_colors])
        green_cluster = np.argmin(delta_e_values)

        # Create the resultset
        result_set = []
        for y_index, _ in enumerate(clustered_image):
            for x_index, classification in enumerate(clustered_image[y_index]):
                if classification != green_cluster:
                    continue
                x_offset = x_index * 256
                y_offset = y_index * 256
                result_set.append({
                    'image': source_image[
                        y_offset: y_offset + 256,
                        x_offset: x_offset + 256
                    ],
                    'center': {
                        "x": x_offset + 128,
                        "y": y_offset + 128
                    }
                })

        return result_set

    def _filter_pipeline(self, source_image: np.ndarray) -> bool:
        """Filter the image based on the pipeline provided through the thesis of Robin

        Args:
            source_image (np.ndarray): Source image

        Returns:
            bool: Return if the image matches the pipeline stage
        """
        filter_result = False
        size = len(source_image)
        start_pred = time.time()
        reshaped_image = tf.reshape(source_image, [1, size, size, 3])
        if self._tensorflow_version == TensorFlowVersion.DEFAULT:
            prediction = self._pipeline_models[size].predict(
                reshaped_image, verbose=0)
            filter_result = prediction[0][1] > 0.5
        elif self._tensorflow_version == TensorFlowVersion.LITE:
            interpreter = self._pipeline_lite_interpreters[size]
            interpreter.set_tensor(
                tensor_index=interpreter.get_input_details()[0]['index'],
                value=reshaped_image
            )
            interpreter.invoke()
            filter_result = interpreter.get_tensor(
                tensor_index=interpreter.get_output_details()[0]['index']
            )
            filter_result = filter_result[0][1] > 0.9

        pred_time = time.time() - start_pred

        return filter_result

    def evaluate(self, image_path: Path, output_path: Path):
        """Evaluate the images from the file path against the openagri AI model

        Args:
            image_path (Path): Path containing the evaluation image
            output_path (Path): Path to the output path, where an annotated image will be stored.
        """
        self.logger.info("Start evaluation on path %s", image_path)
        self.logger.info("Reading image into memory")
        input_image = cv2.imread(image_path)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        # Crop the image:
        x_size = input_image.shape[0]//256
        y_size = input_image.shape[1]//256
        self.logger.info("Crop image to %dx%d", x_size * 256, y_size * 256)
        input_image = input_image[:x_size * 256, :y_size * 256]

        # Filter out the leaves through kmeans clustering:
        self.logger.info("Start kmeans filtering to extract the leaves")
        leave_images = self._filter_kmeans(input_image)
        self.logger.info("Kmeans matches %d out of %d image pieces",
                         len(leave_images), x_size * y_size)

        # Do some preprocessing:
        for leave_image in leave_images:
            leave_image['image'] = np.array(
                leave_image['image'], dtype=np.float32) / 255

        # We do the beetle matching based on three different models:
        # - input: 256x256 images
        # - input: 128x128 images
        # - input: 64x64 images
        # Only the pieces that passed the previous stage, are parsed to the next model.
        # The result will be the matching 64x64 pieces matching the Colorado Beetle.
        active_image_set = leave_images
        matching_image_set = []
        matching_images_per_resolution = {}
        for pipeline_model in self._pipeline_models_sizes:
            matching_image_set = []
            for active_image in active_image_set:
                pipeline_match = self._filter_pipeline(
                    active_image['image'])
                if not pipeline_match:
                    continue
                matching_image_set.append(active_image)
            self.logger.info("Pipeline matches %d out of %d image pieces (size %d)",
                             len(matching_image_set), len(active_image_set) * 4, pipeline_model)

            active_image_set = matching_image_set
            matching_images_per_resolution[pipeline_model] = matching_image_set

            # Iteration completed. Do some post=processing for next iteration.
            if pipeline_model != self._pipeline_destination_size:
                active_image_set_split = []
                for active_image in active_image_set:
                    active_image_set_split.extend(
                        self._split_image(
                            source_image=active_image['image'],
                            piece_size=pipeline_model // 2,
                            x_offset=active_image['center']['x'] -
                            pipeline_model // 2,
                            y_offset=active_image['center']['y'] -
                            pipeline_model // 2
                        )
                    )
                active_image_set = active_image_set_split

        self.logger.info("Completed")
        if output_path.name:
            self._visualise(
                image_name=image_path.name,
                source_image=input_image,
                output_image_path=output_path,
                leave_images=leave_images,
                model_images=matching_images_per_resolution
            )


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(format='[%(levelname)s %(name)s @ %(asctime)s] %(message)s',
                        level=logging.DEBUG)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("colormath").setLevel(logging.ERROR)

    # Parse the script arguments:
    parser = argparse.ArgumentParser(
        description='Evaluate image(s) from Colorado Beetles')
    parser.add_argument('--input', dest="input", metavar='path', required=True,
                        help='The path to the image')
    parser.add_argument('--output', dest="output", metavar='path', required=False, default="",
                        help='The path where an annotated output image can be stored.')
    parser.add_argument('--tensorflow', dest="tensorflow", metavar='path', required=True,
                        choices=[
                            TensorFlowVersion.DEFAULT.value,
                            TensorFlowVersion.LITE.value
                        ],
                        default=TensorFlowVersion.DEFAULT.value,
                        help='Select the tensorflow version (lite or default)')
    script_arguments = parser.parse_args()

    # Initialise and start the OpenAgri AI module:
    ai_module = OpenAgriAI(TensorFlowVersion(script_arguments.tensorflow))
    ai_module.evaluate(image_path=Path(script_arguments.input),
                       output_path=Path(script_arguments.output))
