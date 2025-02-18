from ..constants import MODEL_YOLO, MODEL_YOLO_TFLITE
from ..settings import get_config
from .base import _BaseAIModel
from .result import AiResult, ModelResult
from .version import TensorFlowVersion
from pathlib import Path
from ultralytics import YOLO
import cv2
import time


class YoloModel(_BaseAIModel):
    """YoloV8 model for Colorado Beetle detection"""

    def __init__(
        self,
        confidence: float,
        device: str = "cpu",
    ):
        super().__init__(device)
        config = get_config()
        self._confidence = confidence
        self._output_dimension = config.DETECTION_MODEL_DIM
        self._tensorflow_version = TensorFlowVersion(config.TF_VERSION)
        self._device = device

        self._model_path = MODEL_YOLO
        if self._tensorflow_version == TensorFlowVersion.LITE:
            self._model_path = MODEL_YOLO_TFLITE

        self._yolo_model = YOLO(self._model_path, task="detect")
        if self._device == "GPU":
            self.logger.info("CB detection using GPU")
            self._yolo_model.to("cuda")

    def _filter(self, image_description: AiResult) -> list[dict]:
        """Filter out Colorado Beetle using the YoloV8 model

        Args:
            image_description (AiResult): Source image description

        Returns:
            list[dict]: Matches within the source image
        """
        yolo_match = self._yolo_model.predict(
            image_description.image,
            conf=self._confidence,
            verbose=False,
            device=self._device,
        )
        yolo_results = []
        for match in yolo_match:
            for match_entry in match.summary():
                yolo_results.append(
                    {
                        "x": image_description.loc_x + int(match_entry["box"]["x1"]),
                        "y": image_description.loc_y + int(match_entry["box"]["y1"]),
                        "w": int(match_entry["box"]["x2"] - match_entry["box"]["x1"]),
                        "h": int(match_entry["box"]["y2"] - match_entry["box"]["y1"]),
                        "confidence": match_entry["confidence"],
                    }
                )
        return yolo_results

    def evaluate(self, image_file: cv2.typing.MatLike) -> ModelResult:
        """Evaluate the images from the file path against the openagri YoloV8 AI model

        Args:
            image_file (cv2.typing.MatLike): Source image
        """
        self.logger.info("Start evaluation")
        start_time = time.time()
        leave_images = self._collect_leaves(image_file, self._output_dimension)
        matches = []
        for active_image in leave_images:
            matches.extend(self._filter(active_image))

        matching_image_set = []

        for match in matches:
            matching_image_set.append(
                AiResult(
                    image=image_file[
                        max(match["y"] - 16, 0) : min(
                            match["y"] + 48, image_file.shape[0]
                        ),  # match["h"],
                        max(match["x"] - 16, 0) : min(
                            match["x"] + 48, image_file.shape[1]
                        ),  # match["w"],
                    ],
                    loc_x=match["x"],
                    loc_y=match["y"],
                    width=match["w"],
                    height=match["h"],
                    confidence=match["confidence"],
                )
            )
        end_time = time.time()

        self.logger.info("Completed")

        return ModelResult(
            execution_time=end_time - start_time,
            leaves=leave_images,
            beetles={64: matching_image_set},
        )
