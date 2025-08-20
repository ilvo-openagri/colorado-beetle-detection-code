from .result import AiResult, ModelResult
from ..leave_filter import SegmentationFilter
from abc import ABC, abstractmethod
import cv2
import logging
import numpy as np


class _BaseAIModel(ABC):
    """Base AI model."""

    def __init__(self, device: str):
        self.logger = logging.getLogger(f"OpenAgriAIModule_{__name__}")
        self.logger.setLevel(logging.INFO)
        self.leave_filter = SegmentationFilter(device=device)

    def _split_image(
        self,
        source_image: np.ndarray,
        piece_size: int,
        x_offset: int = 0,
        y_offset: int = 0,
        margin: int = 0,
    ) -> list:
        """Split image into equal sized pieces

        Args:
            source_image (np.ndarray): Source image
            piece_size (int): Dimension of each piece
            x_offset (int): X offset to the center of the piece
            y_offset (int): Y offset to the center of the piece
            margin (int): Some margin around the image

        Returns:
            list: The equal sized pieces in which the image was split.
        """

        pieces = []
        y = 0

        x_max_dim = len(source_image[0])
        y_max_dim = len(source_image)
        self.logger.debug("Dimensions: %d/%d", x_max_dim, y_max_dim)

        # Remove edges if deemed too small
        while y < y_max_dim:
            x = 0
            # Remove edges if deemed too small
            while x < x_max_dim:
                x_end = x + piece_size if x + piece_size < x_max_dim else x_max_dim
                y_end = y + piece_size if y + piece_size < y_max_dim else y_max_dim

                # Cut out a piece
                img_piece = source_image[
                    max(y - margin, 0) : y_end + margin,
                    max(x - margin, 0) : x_end + margin,
                ]
                pieces.append(
                    AiResult(
                        image=img_piece,
                        loc_x=x + x_offset,
                        loc_y=y + y_offset,
                        width=piece_size,
                        height=piece_size,
                        confidence=0,
                    )
                )
                x += piece_size
            y += piece_size
        if len(pieces) != 4:
            self.logger.error("Unexpected image split size of %d", len(pieces))

        return pieces

    def _collect_leaves(self, image_file: cv2.typing.MatLike, dimension: int):
        """Evaluate the image to extract regions with leaves

        Args:
            image_file (cv2.typing.MatLike): Source image
            dimension (int): Dimension of the regions
        """

        # # Crop the image:
        x_size = image_file.shape[0] // dimension
        y_size = image_file.shape[1] // dimension
        # self.logger.info("Crop image to %dx%d [%d]", x_size * dimension, y_size * dimension, dimension)
        # image_file = image_file[: x_size * dimension, : y_size * dimension]

        # Filter out the leaves:
        self.logger.info("Start filtering to extract the leaves")
        leave_images = self.leave_filter.filter(image_file, dimension)
        self.logger.info(
            "Matched %d out of %d image pieces for leaves",
            len(leave_images),
            x_size * y_size,
        )

        return leave_images

    @abstractmethod
    def evaluate(self, image_file: cv2.typing.MatLike) -> ModelResult:
        """Evaluate the images from the file path against the openagri AI model

        Args:
            image_file (cv2.typing.MatLike): Source image
        """
        return

    def close(self):
        """Release all resources on close"""
        return
