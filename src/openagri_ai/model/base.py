from .result import AiResult, ModelResult
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
import cv2
import logging
import numpy as np


class _BaseAIModel(ABC):
    """Base AI model."""

    def __init__(self):
        self.logger = logging.getLogger(f"OpenAgriAIModule_{__name__}")
        self.logger.setLevel(logging.INFO)

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
                        center_x=x + piece_size / 2 + x_offset,
                        center_y=y + piece_size / 2 + y_offset,
                        confidence=0,
                    )
                )
                x += piece_size
            y += piece_size
        if len(pieces) != 4:
            self.logger.error("Unexpected image split size of %d", len(pieces))

        return pieces

    def _filter_kmeans(
        self, source_image: np.ndarray, dimension: int
    ) -> list[np.ndarray]:
        """Do a kmeans classification to identify the leaves vs soil

        Args:
            source_image (np.ndarray): Source image
            dimension (int): Dimension of the fragments

        Returns:
            list[np.ndarray]: Matching image fragments of dimension x dimension
        """
        # Split the image into its RGB components
        b, g, r = cv2.split(source_image)

        # Convert channels to float to avoid integer division issues
        r = r.astype(np.float32)
        g = g.astype(np.float32)
        b = b.astype(np.float32)

        # Avoid division by zero by setting zero values in g to a small number
        g[g == 0] = 1
        # Apply the formula (r + b) / g
        one_d_image = (r + b) / g

        # Convert the result back to uint8

        resized_image = cv2.resize(
            one_d_image,
            (one_d_image.shape[1] // dimension, one_d_image.shape[0] // dimension),
            interpolation=cv2.INTER_AREA,
        )

        (h, w) = resized_image.shape
        flat_image = resized_image.reshape(h * w, 1)
        kmeans_model = KMeans(n_clusters=2)
        cluster_labels = kmeans_model.fit_predict(flat_image)
        clustered_image = np.reshape(cluster_labels, (h, w))

        # Determine the cluster with a green color:
        green_cluster = np.argmin(kmeans_model.cluster_centers_)

        # Create the resultset
        result_set = []
        for y_index, _ in enumerate(clustered_image):
            for x_index, classification in enumerate(clustered_image[y_index]):
                if classification != green_cluster:
                    continue
                x_offset = x_index * dimension
                y_offset = y_index * dimension
                result_set.append(
                    AiResult(
                        image=source_image[
                            y_offset : y_offset + dimension,
                            x_offset : x_offset + dimension,
                        ],
                        center_x=x_offset + dimension // 2,
                        center_y=y_offset + dimension // 2,
                        confidence=0,
                    )
                )

        return result_set

    def _collect_leaves(self, image_file: cv2.typing.MatLike, dimension: int):
        """Evaluate the image to extract regions with leaves

        Args:
            image_file (cv2.typing.MatLike): Source image
            dimension (int): Dimension of the regions
        """

        # Crop the image:
        x_size = image_file.shape[0] // dimension
        y_size = image_file.shape[1] // dimension
        self.logger.info("Crop image to %dx%d", x_size * dimension, y_size * dimension)
        image_file = image_file[: x_size * dimension, : y_size * dimension]

        # Filter out the leaves through kmeans clustering:
        self.logger.info("Start kmeans filtering to extract the leaves")
        leave_images = self._filter_kmeans(image_file, dimension)
        self.logger.info(
            "Kmeans matches %d out of %d image pieces",
            len(leave_images),
            x_size * y_size,
        )

        return leave_images

    @abstractmethod
    def evaluate(self, image_file: cv2.typing.MatLike) -> ModelResult:
        """Evaluate the images from the file path against the openagri YOLO AI model

        Args:
            image_file (cv2.typing.MatLike): Source image
        """
        return
