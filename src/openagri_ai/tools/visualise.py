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
    
def write_image(image: np.ndarray, filename: str):
    """Save image to a file using matplotlib

    Args:
        image (np.ndarray): Input image
        filename (str): Output file path
    """
    plt.imsave(filename, image)
