from .base import _BaseAIModel
from .result import AiResult, ModelResult
from .version import TensorFlowVersion
from pathlib import Path
from ultralytics import YOLO
import cv2
import keras
import multiprocessing
import numpy as np
import tensorflow as tf


class PipelineModel(_BaseAIModel):

    def __init__(
        self,
        tensorflow_version: TensorFlowVersion,
        model_paths: dict[int:Path],
        amount_of_workers: int,
        confidence: float,
    ):
        super().__init__()
        self._tensorflow_version = tensorflow_version
        self._model_paths = model_paths
        self._output_dimension = 256
        self._mp_task_queue = multiprocessing.Queue()
        self._mp_result_queue = multiprocessing.Queue()
        self._mp_workers = []
        for worker_index in range(amount_of_workers):
            p = multiprocessing.Process(
                target=self._mp_worker,
                args=(
                    worker_index,
                    confidence,
                    self._mp_task_queue,
                    self._mp_result_queue,
                ),
            )
            p.start()
            self._mp_workers.append(p)

    def _mp_worker(
        self,
        _worker_index: int,
        confidence: float,
        task_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
    ) -> bool:
        """Filter the image based on the pipeline provided through the thesis of Robin

        Args:
            source_image (np.ndarray): Source image

        Returns:
            bool: Return if the image matches the pipeline stage
        """
        if self._tensorflow_version == TensorFlowVersion.DEFAULT:
            pipeline_models = {
                256: keras.models.load_model(self._model_paths[256]),
                128: keras.models.load_model(self._model_paths[128]),
                64: keras.models.load_model(self._model_paths[64]),
            }
        elif self._tensorflow_version == TensorFlowVersion.LITE:
            pipeline_lite_interpreters = {
                256: tf.lite.Interpreter(self._model_paths[256]),
                128: tf.lite.Interpreter(self._model_paths[128]),
                64: tf.lite.Interpreter(self._model_paths[64]),
            }
            for interpreter in pipeline_lite_interpreters.values():
                interpreter.allocate_tensors()

        while True:
            filter_result = 0
            image_description = task_queue.get()
            size = len(image_description.image)
            reshaped_image = tf.reshape(image_description.image, [1, size, size, 3])
            if self._tensorflow_version == TensorFlowVersion.DEFAULT:
                prediction = pipeline_models[size].predict(reshaped_image, verbose=0)
                filter_result = prediction[0][1]
            elif self._tensorflow_version == TensorFlowVersion.LITE:
                interpreter = pipeline_lite_interpreters[size]
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
                result_queue.put(image_description)
            else:
                result_queue.put(None)

    def evaluate(self, image_file: cv2.typing.MatLike) -> ModelResult:
        self.logger.info("Start evaluation")

        leave_images = self._collect_leaves(image_file, self._output_dimension)
        matching_image_set = []
        matching_images_per_resolution = {}
        task_done_count = 0

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
                self._mp_task_queue.put(active_image)

            # Wait for all workers to complete, and gather the results:
            self.logger.info(
                "Waiting for workers to finish for model %d", pipeline_model
            )
            task_done_count = 0
            while task_done_count < len(active_image_set):
                process_result = self._mp_result_queue.get()
                task_done_count += 1
                if process_result is not None:
                    matching_image_set.append(process_result)

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
