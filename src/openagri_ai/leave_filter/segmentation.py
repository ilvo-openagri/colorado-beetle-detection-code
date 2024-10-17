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
        # Load a model
        model = YOLO(
            "/home/jbauwens/Documents/Code/openagri_ai/runs/segment/yolov8n_segmentation_100e6/weights/best.pt"
        )  # load a custom model

        # Predict with the model
        results = model(
            "/home/jbauwens/Documents/Code/openagri_ai/test_images/DCS05185.JPG",
        )
        leave_mask = np.zeros(source_image.shape[:2], np.uint8)
        for result in results:
            for ci, c in enumerate(result):
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(leave_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
        show_image(leave_mask)

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
                    try:
                        greeness = (float(pxl[0]) + float(pxl[2])) / float(pxl[1])
                        if greeness < self.green_cutoff_value:
                            green_hits += 1
                    except ZeroDivisionError:
                        continue
                if green_hits <= 1:
                    continue

                # ai_result = AiResult(
                #     image=source_image[
                #         x_coordinate : x_coordinate + dimension,
                #         y_coordinate : y_coordinate + dimension,
                #     ],
                #     center_x=y_coordinate + dimension // 2,
                #     center_y=x_coordinate + dimension // 2,
                #     confidence=0,
                # )

                # result_set.append(ai_result)

        return result_set
