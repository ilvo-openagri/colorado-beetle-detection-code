import numpy as np
import matplotlib.pyplot as plt


def show_image(image: np.ndarray):
    """Show image in a matplotlib graph

    Args:
        image (np.ndarray): Input image
    """
    _fig, ax = plt.subplots(1, 1)
    ax.imshow(
        image,
    )
    plt.show()
