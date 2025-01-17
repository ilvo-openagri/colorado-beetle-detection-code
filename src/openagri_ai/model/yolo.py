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
        super().__init__()
        config = get_config()
        self._confidence = confidence
        self._output_dimension = config.DETECTION_MODEL_DIM
        self._tensorflow_version = TensorFlowVersion(config.TF_VERSION)
        self._device = device

        self._model_path = MODEL_YOLO
        if self._tensorflow_version == TensorFlowVersion.LITE:
            self._model_path = MODEL_YOLO_TFLITE

        self._yolo_model = YOLO(self._model_path, task="detect")
        if self._device == "gpu":
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
            if match.summary():
                x, y, w, h = match.boxes.xywh[0]
                yolo_results.append(
                    {
                        "x": image_description.center_x - 128 + int(x),
                        "y": image_description.center_y - 128 + int(y),
                        "w": int(w),
                        "h": int(h),
                        "confidence": float(match.boxes.conf[0]),
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
                    image=None,
                    center_x=match["x"],
                    center_y=match["y"],
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
