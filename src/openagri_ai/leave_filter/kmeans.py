from ..model.result import AiResult
from sklearn.cluster import KMeans
import cv2
import logging
import numpy as np


class KmeansFilter:
    """Leave filter using kmeans clustering"""

    def __init__(self, green_cutoff_value: float = 2.0):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.green_cutoff_value = green_cutoff_value

    def filter(self, source_image: np.ndarray, dimension: int) -> list[np.ndarray]:
        """Do a classification to identify the leaves vs soil

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
