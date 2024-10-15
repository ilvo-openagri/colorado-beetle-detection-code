from ..model.result import AiResult
from ..tools.visualise import show_image

import cv2
import logging
import numpy as np


class LeaveFilter:

    def __init__(
        self,
        green_cutoff_value: float = 2.0,
        do_hsv_green: bool = False,
        do_canny: bool = False,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.green_cutoff_value = green_cutoff_value
        self.do_hsv_green = do_hsv_green
        self.do_canny = do_canny

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

        # 1. First extract the "real" greens:
        if self.do_hsv_green:
            self.logger.info("Extract green")
            # Convert to HSV color-space
            hsv_image = cv2.cvtColor(source_image, cv2.COLOR_RGB2HSV)
            first_round = True

            # Perform color-segmentation to get the binary mask
            lower_bound = np.array([20, 40, 40])
            upper_bound = np.array([80, 255, 255])
            msk = cv2.inRange(hsv_image, lower_bound, upper_bound)
            source_image = cv2.bitwise_and(source_image, source_image, mask=msk)

        # 2. Canny
        if self.do_canny:
            self.logger.info("Perform Canny")
            canny = cv2.Canny(
                cv2.cvtColor(source_image, cv2.COLOR_RGB2GRAY),
                255,
                255,
                apertureSize=3,
            )

            canny_edges = []
            canny_patch = 128
            for x_index in range(source_image.shape[0] // canny_patch):
                for y_index in range(source_image.shape[1] // canny_patch):
                    x_coordinate = x_index * canny_patch
                    y_coordinate = y_index * canny_patch
                    # We only do the canny on green patches:
                    center_pxl = source_image[x_coordinate + canny_patch // 2][
                        y_coordinate + canny_patch // 2
                    ]
                    greeness = (float(center_pxl[0]) + float(center_pxl[2])) / float(
                        center_pxl[1]
                    )
                    if greeness > self.green_cutoff_value:
                        continue
                    # Update the resultset
                    canny_edges.append(
                        np.count_nonzero(
                            canny[
                                x_coordinate : x_coordinate + canny_patch,
                                y_coordinate : y_coordinate + canny_patch,
                            ],
                        )
                    )
            canny_cutoff = max(1750, np.mean(canny_edges) + 1 * np.std(canny_edges))
            self.logger.info(canny_cutoff)

            canny_mask = np.ones_like(canny)
            for x_index in range(source_image.shape[0] // canny_patch):
                for y_index in range(source_image.shape[1] // canny_patch):
                    x_coordinate = x_index * canny_patch
                    y_coordinate = y_index * canny_patch
                    # We only do the canny on green patches:
                    center_pxl = source_image[x_coordinate + canny_patch // 2][
                        y_coordinate + canny_patch // 2
                    ]
                    greeness = (float(center_pxl[0]) + float(center_pxl[2])) / float(
                        center_pxl[1]
                    )
                    if greeness > self.green_cutoff_value:
                        continue
                    # Update the mask:
                    if canny_edges.pop(0) > canny_cutoff:
                        canny_mask[
                            x_coordinate : x_coordinate + canny_patch,
                            y_coordinate : y_coordinate + canny_patch,
                        ] = 0
            source_image = cv2.bitwise_and(source_image, source_image, mask=canny_mask)
            show_image(source_image)

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
