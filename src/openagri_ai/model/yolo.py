from .base import _BaseAIModel
from .result import AiResult, ModelResult
from pathlib import Path
from ultralytics import YOLO
import cv2
import multiprocessing
from .version import TensorFlowVersion


class YoloModel(_BaseAIModel):

    def __init__(
        self,
        tensorflow_version: TensorFlowVersion,
        model_path: Path,
        amount_of_workers: int,
        confidence: float,
        device: str = "cpu",
    ):
        super().__init__()
        self._confidence = confidence
        self._model = None
        self._model_path = model_path
        self._mp_result_queue = multiprocessing.Queue()
        self._mp_task_queue = multiprocessing.Queue()
        self._mp_workers = []
        self._output_dimension = 256
        self._tensorflow_version = tensorflow_version
        self._device = device
        if amount_of_workers == 0:
            self._yolo_model = YOLO(self._model_path, task="detect")
            self._yolo_model.to("cuda")
        else:
            for worker_index in range(amount_of_workers):
                p = multiprocessing.Process(
                    target=self._mp_worker,
                    args=(
                        worker_index,
                        self._model_path,
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
        model: str,
        confidence: float,
        task_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
    ):
        yolo_model = YOLO(model, task="detect")
        while True:
            image_description = task_queue.get()
            yolo_results = self._mp_worker_work(
                yolo_model, image_description, confidence
            )
            result_queue.put(yolo_results)

    def _mp_worker_work(
        self, model: YOLO, image_description: AiResult, confidence: float
    ) -> list:
        yolo_match = model.predict(
            image_description.image,
            conf=confidence,
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
        self.logger.info("Start evaluation")

        leave_images = self._collect_leaves(image_file, self._output_dimension)
        matches = []

        if not self._mp_workers:
            for active_image in leave_images:
                matches.extend(
                    self._mp_worker_work(
                        self._yolo_model, active_image, self._confidence
                    )
                )
        else:
            for active_image in leave_images:
                self._mp_task_queue.put(active_image)

            self.logger.info("Waiting for workers to finish")

            # Collect the results
            task_done_count = 0
            while task_done_count < len(leave_images):
                process_result = self._mp_result_queue.get()
                task_done_count += 1
                matches.extend(process_result)

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

        self.logger.info("Completed")

        return ModelResult(leaves=leave_images, beetles={64: matching_image_set})
