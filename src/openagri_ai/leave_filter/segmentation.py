from ..model.result import AiResult
from ..tools.visualise import write_image
from ..settings import get_config
from ..config import OpenagriConfig
from ..constants import MODEL_CLASS_NCNN, MODEL_CLASS_PYTORCH

from ultralytics import YOLO
import cv2
import logging
import numpy as np


class SegmentationFilter:
    """Leave filter using a YoloV8 segmentation"""

    def __init__(
        self, device: str, do_warmup: bool = False, config: OpenagriConfig = None
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self._device = device
        if config is None:
            config = get_config()

        self.logger.info(
            "Using segmentation model: %s", config.LEAVE_SEGMENTATION_MODEL
        )
        if config.LEAVE_SEGMENTATION_MODEL == "NCNN":
            model_file = MODEL_CLASS_NCNN
        else:
            model_file = MODEL_CLASS_PYTORCH

        self._model = YOLO(
            model_file,
            task="segment",
        )
        self._leave_mask = None

        # Do a warmup detection. The first prediction takes longer do to some additional loading.
        if do_warmup:
            self._model.predict("test_images/sectie32.JPG", verbose=False)

    def filter(self, source_image: np.ndarray, dimension: int) -> list[AiResult]:
        """Do a classification to identify the leaves vs soil

        Args:
            source_image (np.ndarray): Source image
            dimension (int): Dimension of the fragments

        Returns:
            list[np.ndarray]: Matching image fragments of dimension x dimension
        """
        result_set = []

        grid_dimension = (
            source_image.shape[0] // dimension,
            source_image.shape[1] // dimension,
        )

        # 1. First perform segmentation to identify the leaves:
        self.logger.info("Performing segmentation")

        if self._device == "GPU":
            results = self._model.predict(source_image, verbose=False, device=0)
        else:
            results = self._model.predict(source_image, verbose=False)

        self._leave_mask = np.zeros(source_image.shape[:2], np.uint8)

        for result in results:
            for c in result:
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(
                    self._leave_mask, [contour], -1, (255, 255, 255), cv2.FILLED
                )
        write_image(self._leave_mask, "output_1.jpg")

        # 2. Extract the images containing enough green:
        for x_index in range(grid_dimension[0]):
            for y_index in range(grid_dimension[1]):
                x_coordinate = x_index * dimension
                y_coordinate = y_index * dimension
                relevant_pixels = (
                    (x_coordinate, y_coordinate),
                    (x_coordinate + dimension - 1, y_coordinate),
                    (x_coordinate, y_coordinate + dimension - 1),
                    (x_coordinate + dimension - 1, y_coordinate + dimension - 1),
                    (x_coordinate + dimension // 2, y_coordinate + dimension // 2),
                )
                green_hits = 0
                self.logger.info(len(relevant_pixels))
                for pixel in relevant_pixels:
                    if pixel[0] < 0 or pixel[0] >= source_image.shape[0]:
                        # Out of bounds in x direction
                        continue
                    if pixel[1] < 0 or pixel[1] >= source_image.shape[1]:
                        # Out of bounds in y direction
                        continue
                    if self._leave_mask[pixel[0]][pixel[1]] == 0:
                        # No green pixel in this fragment
                        continue
                    green_hits += 1
                if green_hits == 0:
                    continue

                ai_result = AiResult(
                    image=source_image[
                        x_coordinate : x_coordinate + dimension,
                        y_coordinate : y_coordinate + dimension,
                    ],
                    loc_x=y_coordinate,
                    loc_y=x_coordinate,
                    width=dimension,
                    height=dimension,
                    confidence=0,
                )

                result_set.append(ai_result)
        self.logger.info("Segmentation completed. Found %d fragments", len(result_set))

        return result_set


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        format="[%(levelname)s %(name)s @ %(asctime)s] %(message)s", level=logging.DEBUG
    )
    logger = logging.getLogger(__name__)
    logger.info("Testing segmentation")
    segmentation_filter = SegmentationFilter(
        device="CPU",
        do_warmup=False,
        config=OpenagriConfig(
            DETECTION_MODEL="trained_models/model_yolov8n_segmentation/3d_092025",
            LEAVE_SEGMENTATION_MODEL="NCNN",
        ),
    )
    input_image = cv2.imread("test_images/DSC00813.JPG")
    dimension = 640
    x_size = input_image.shape[0] // dimension
    y_size = input_image.shape[1] // dimension
    logger.info("Crop image to %dx%d", x_size * dimension, y_size * dimension)
    input_image = input_image[: x_size * dimension, : y_size * dimension]
    leave_images = segmentation_filter.filter(input_image, 640)
    logger.info(
        "Matched %d out of %d image pieces for leaves",
        len(leave_images),
        x_size * y_size,
    )
    write_image(segmentation_filter._leave_mask, "/tmp/leave_mask.jpg")
    masked_image = cv2.bitwise_and(
        input_image, input_image, mask=segmentation_filter._leave_mask
    )
    write_image(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB), "/tmp/masked_image.jpg")
