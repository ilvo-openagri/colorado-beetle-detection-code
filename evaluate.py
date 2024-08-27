"""Module to perform all AI related steps for the OpenAgri project.
"""

# Standard imports
import argparse
import logging
import time
from pathlib import Path

# Third party imports
import joblib
import keras
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
import skimage.filters
import skimage.io
import skimage.transform
import tensorflow as tf

# Local imports
import constants


class OpenAgriAI:
    """Class to perform all AI related steps for the OpenAgri project.
    """

    def __init__(self):
        self.logger = logging.getLogger("OpenAgriAIModule")
        self.logger.setLevel(logging.INFO)
        self._kmeans_model = joblib.load(constants.MODEL_KMEANS)
        self._pipeline_models = {
            # 64: keras.models.load_model(constants.MODEL_64),
            128: keras.models.load_model(constants.MODEL_128),
            256: keras.models.load_model(constants.MODEL_256),
        }

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

        y_max_dim = len(source_image)
        x_max_dim = len(source_image[0])
        self.logger.debug("Dimensions: %d/%d", x_max_dim, y_max_dim)

        # Remove edges if deemed too small
        while y <= y_max_dim - piece_size:
            x = 0
            # Remove edges if deemed too small
            while x <= x_max_dim - piece_size:
                x_end = x + piece_size if x + \
                    piece_size < x_max_dim else x_max_dim
                y_end = y + piece_size if y + \
                    piece_size < y_max_dim else y_max_dim

                # Cut out a piece
                img_piece = source_image[y:y_end, x:x_end]
                pieces.append({
                    'image': img_piece,
                    "center": (
                        x + piece_size / 2 + x_offset,
                        y + piece_size / 2 + y_offset
                    )
                })
                x += piece_size
            y += piece_size

        return pieces

    def _filter_kmeans(self, source_image: np.ndarray):
        # Reshape the input image
        start_reshape = time.time()
        resized_image = skimage.transform.resize(
            source_image, (64, 64))
        resized_image = resized_image.reshape((1, -1))
        resized_image = np.array(resized_image, dtype=np.float32)
        reshape_time = time.time() - start_reshape

        # Predict through the kmeans model
        start_pred = time.time()
        prediction = self._kmeans_model.predict(resized_image)
        pred_time = time.time() - start_pred

        # Return if the kmeans model matches:
        return prediction == 0

    def _filter_pipeline(self, source_image: np.ndarray) -> bool:
        """Filter the image based on the pipeline provided through the thesis of Robin

        Args:
            source_image (np.ndarray): Source image

        Returns:
            bool: Return if the image matches the pipeline stage
        """
        size = len(source_image)
        start_pred = time.time()
        reshaped_image = tf.reshape(source_image, [1, size, size, 3])
        prediction = self._pipeline_models[size].predict(
            reshaped_image, verbose=0)
        pred_time = time.time() - start_pred

        return prediction[0][1] > 0.5

    def evaluate(self, image_path: Path):
        """Evaluate the images from the file path against the openagri AI model

        Args:
            image_path (Path): Path containing the evaluation images
        """
        self.logger.info("Start evaluation on path %s", image_path)
        input_image = skimage.io.imread(image_path)
        image_pieces = self._split_image(input_image, 512)

        # Filter out the leaves through kmeans clustering:
        leave_images = []
        for image_piece in image_pieces:
            kmeans_match = self._filter_kmeans(image_piece['image'])
            if not kmeans_match:
                continue

            leave_images.append(image_piece)
        self.logger.info("Kmeans matches %d out of %d image pieces",
                         len(leave_images), len(image_pieces))

        # Do some preprocessing:
        for leave_image in leave_images:
            leave_image["image"] = np.array(leave_image["image"]) / 255

        # Split the images in four parts:
        active_image_set = leave_images
        matching_image_set = []
        pipeline_models = [256, 128]
        for pipeline_model in pipeline_models:
            matching_image_set = []
            for active_image in active_image_set:
                active_images_split = self._split_image(
                    source_image=active_image['image'],
                    piece_size=pipeline_model,
                    x_offset=active_image['center'][0] - pipeline_model,
                    y_offset=active_image['center'][1] - pipeline_model)
                for active_image_split in active_images_split:
                    pipeline_match = self._filter_pipeline(
                        active_image_split['image'])
                    if not pipeline_match:
                        continue
                    matching_image_set.append(active_image_split)
            self.logger.info("Pipeline matches %d out of %d image pieces (size %d)",
                             len(matching_image_set), len(active_image_set) * 4, pipeline_model)
            active_image_set = matching_image_set

        self.logger.info("Completed")


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(format='[%(levelname)s %(name)s @ %(asctime)s] %(message)s',
                        level=logging.DEBUG)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    # Parse the script arguments:
    parser = argparse.ArgumentParser(
        description='Evaluate image(s) from Colorado Beetles')
    parser.add_argument('--input', dest="input", metavar='path', required=True,
                        help='The path to the images')
    script_arguments = parser.parse_args()

    # Initialise and start the OpenAgri AI module:
    ai_module = OpenAgriAI()
    ai_module.evaluate(image_path=Path(script_arguments.input))
