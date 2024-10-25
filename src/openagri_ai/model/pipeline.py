from ..constants import (
    MODEL_64,
    MODEL_128,
    MODEL_256,
    MODEL_LITE_64,
    MODEL_LITE_128,
    MODEL_LITE_256,
)
from ..leave_filter import KmeansFilter
from ..settings import get_config
from .base import _BaseAIModel
from .result import AiResult, ModelResult
from .version import TensorFlowVersion
import cv2
import numpy as np
import tensorflow as tf


class PipelineModel(_BaseAIModel):
    """AI model for the original pipeline model (see Thesis_Robin.pdf)
    The execution has been improved compared to the available implementation
    """

    def __init__(
        self,
        confidence: float,
    ):
        super().__init__()
        config = get_config()
        self._confidence = confidence
        self._tensorflow_version = TensorFlowVersion(config.TF_VERSION)
        self._output_dimension = 256
        self._filters = []
        self.leave_filter = KmeansFilter()

        if self._tensorflow_version == TensorFlowVersion.DEFAULT:
            self._pipeline_models = {
                256: tf.saved_model.load(MODEL_256),
                128: tf.saved_model.load(MODEL_128),
                64: tf.saved_model.load(MODEL_64),
            }
        if self._tensorflow_version == TensorFlowVersion.LITE:
            self._pipeline_models = {
                256: tf.lite.Interpreter(MODEL_LITE_256),
                128: tf.lite.Interpreter(MODEL_LITE_128),
                64: tf.lite.Interpreter(MODEL_LITE_64),
            }
            for interpreter in self._pipeline_models.values():
                interpreter.allocate_tensors()

    def _filter(self, confidence: float, image_description: AiResult) -> bool:
        """Filter the image based on the pipeline provided through the thesis of Robin

        Args:
            confidence (float): Minimum confidence level of the results
            source_image (np.ndarray): Source image

        Returns:
            bool: Return if the image matches the pipeline stage
        """
        filter_result = 0
        size = len(image_description.image)
        reshaped_image = tf.reshape(image_description.image, [1, size, size, 3])
        if self._tensorflow_version == TensorFlowVersion.DEFAULT:
            prediction = self._pipeline_models[size].signatures["serving_default"](
                input_1=reshaped_image
            )
            filter_result = np.array(
                next(iter(prediction.values()))
            )  # dense_2, dense_1, ...
            filter_result = filter_result[0][1]
        elif self._tensorflow_version == TensorFlowVersion.LITE:
            interpreter = self._pipeline_models[size]
            interpreter.set_tensor(
                tensor_index=interpreter.get_input_details()[0]["index"],
                value=reshaped_image,
            )
            interpreter.invoke()
            filter_result = interpreter.get_tensor(
                tensor_index=interpreter.get_output_details()[0]["index"]
            )
            filter_result = filter_result[0][1]

        if filter_result >= confidence:
            image_description.confidence = filter_result
            return image_description
        else:
            return None

    def evaluate(self, image_file: cv2.typing.MatLike) -> ModelResult:
        """Evaluate the images from the file path against the openagri pipeline AI model

        Args:
            image_file (cv2.typing.MatLike): Source image
        """
        self.logger.info("Start evaluation")

        leave_images = self._collect_leaves(image_file, self._output_dimension)
        matching_image_set = []
        matching_images_per_resolution = {}

        # We do the beetle matching based on three different models:
        # - input: 256x256 images
        # - input: 128x128 images
        # - input: 64x64 images
        # Only the pieces that passed the previous stage, are parsed to the next model.
        # The result will be the matching 64x64 pieces matching the Colorado Beetle.
        # Do some preprocessing:
        for leave_image in leave_images:
            leave_image.image = np.array(leave_image.image, dtype=np.float32) / 255

        active_image_set = leave_images
        pipeline_models = [256, 128, 64]
        for pipeline_model in pipeline_models:
            matching_image_set = []
            self.logger.info("Starting pipeline")

            # Send the images to the workers:
            for active_image in active_image_set:
                result = self._filter(self._confidence, active_image)
                if result is not None:
                    matching_image_set.append(result)

            self.logger.info(
                "Pipeline matches %d out of %d image pieces (size %d)",
                len(matching_image_set),
                len(active_image_set),
                pipeline_model,
            )

            active_image_set = matching_image_set
            matching_images_per_resolution[pipeline_model] = matching_image_set

            # Iteration completed. Do some post=processing for next iteration.
            if pipeline_model != pipeline_models[-1]:
                active_image_set_split = []
                for active_image in active_image_set:
                    active_image_set_split.extend(
                        self._split_image(
                            source_image=active_image.image,
                            piece_size=pipeline_model // 2,
                            x_offset=active_image.center_x - pipeline_model // 2,
                            y_offset=active_image.center_y - pipeline_model // 2,
                        )
                    )
                active_image_set = active_image_set_split

        self.logger.info("Completed")

        return ModelResult(leaves=leave_images, beetles=matching_images_per_resolution)
