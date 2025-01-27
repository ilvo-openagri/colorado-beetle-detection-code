from ..model.result import AiResult
from ..tools.visualise import show_image
from ..settings import get_config
from ..config import OpenagriConfig
from ..constants import MODEL_CLASS_NCNN, MODEL_CLASS_PYTORCH

from ultralytics import YOLO
import cv2
import logging
import numpy as np


class SegmentationFilter:
    """Leave filter using a YoloV8 segmentation"""

    def __init__(self, do_warmup: bool = False, config: OpenagriConfig = None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if config is None:
            config = get_config()

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
        results = self._model.predict(source_image, verbose=False)
        self._leave_mask = np.zeros(source_image.shape[:2], np.uint8)
        for result in results:
            for c in result:
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(
                    self._leave_mask, [contour], -1, (255, 255, 255), cv2.FILLED
                )

        # 2. Extract the images containing enough green:
        for x_index in range(grid_dimension[0]):
            for y_index in range(grid_dimension[1]):
                x_coordinate = x_index * dimension
                y_coordinate = y_index * dimension
                relevant_pixels = (
                    self._leave_mask[x_coordinate][y_coordinate],
                    self._leave_mask[x_coordinate + dimension - 1][y_coordinate],
                    self._leave_mask[x_coordinate][y_coordinate + dimension - 1],
                    self._leave_mask[x_coordinate + dimension - 1][
                        y_coordinate + dimension - 1
                    ],
                    self._leave_mask[x_coordinate + dimension // 2][
                        y_coordinate + dimension // 2
                    ],
                )
                green_hits = 0
                for pxl in relevant_pixels:
                    if pxl == 0:
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
        self.logger.info("Segmentation completed")

        return result_set


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        format="[%(levelname)s %(name)s @ %(asctime)s] %(message)s", level=logging.DEBUG
    )
    logger = logging.getLogger(__name__)
    logger.info("Testing segmentation")
    segmentation_filter = SegmentationFilter(
        config=OpenagriConfig(DETECTION_MODEL=None, LEAVE_SEGMENTATION_MODEL="PYTORCH")
    )
    input_image = cv2.imread("test_images/sectie32.JPG")
    segmentation_filter.filter(input_image, 256)
    show_image(segmentation_filter._leave_mask)
    masked_image = cv2.bitwise_and(
        input_image, input_image, mask=segmentation_filter._leave_mask
    )
    show_image(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
