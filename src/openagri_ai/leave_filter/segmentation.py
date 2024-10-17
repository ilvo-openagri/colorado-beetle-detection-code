from ..model.result import AiResult
from ..tools.visualise import show_image

from ultralytics import YOLO
import cv2
import logging
import numpy as np


class SegmentationFilter:

    def __init__(
        self,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self._model = YOLO(
            "/home/jbauwens/Documents/Code/openagri_ai/runs/segment/yolov8n_segmentation_100e6/weights/best.pt"
        )
        self._model("test_images/sectie32.JPG")

    def filter(self, source_image: np.ndarray, dimension: int) -> list[np.ndarray]:
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
        results = self._model(source_image)
        self.logger.info("Create mask")
        leave_mask = np.zeros(source_image.shape[:2], np.uint8)
        for result in results:
            for c in result:
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(leave_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
        self.logger.info("Done")
        source_image = cv2.bitwise_and(source_image, source_image, mask=leave_mask)
        # show_image(source_image)

        # 2. Extract the images containing enough green:
        for x_index in range(grid_dimension[0]):
            for y_index in range(grid_dimension[1]):
                x_coordinate = x_index * dimension
                y_coordinate = y_index * dimension
                relevant_pixels = (
                    source_image[x_coordinate][y_coordinate],
                    source_image[x_coordinate + dimension - 1][y_coordinate],
                    source_image[x_coordinate][y_coordinate + dimension - 1],
                    source_image[x_coordinate + dimension - 1][
                        y_coordinate + dimension - 1
                    ],
                    source_image[x_coordinate + dimension // 2][
                        y_coordinate + dimension // 2
                    ],
                )
                green_hits = 0
                for pxl in relevant_pixels:
                    if all(i == 0 for i in pxl):
                        continue
                    green_hits += 1
                if green_hits == 0:
                    continue

                ai_result = AiResult(
                    image=source_image[
                        x_coordinate : x_coordinate + dimension,
                        y_coordinate : y_coordinate + dimension,
                    ],
                    center_x=y_coordinate + dimension // 2,
                    center_y=x_coordinate + dimension // 2,
                    confidence=0,
                )

                result_set.append(ai_result)

        return result_set
