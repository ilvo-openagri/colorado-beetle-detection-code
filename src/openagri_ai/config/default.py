"""Module containing the default openagri configuration options
"""

from . import OpenagriConfig

config = OpenagriConfig(
    DETECTION_MODEL="YOLO",
    TF_VERSION="default",
    LEAVE_SEGMENTATION_MODEL="NCNN",
    DETECTION_MODEL_DIM=640,
    DEVICE="CPU",
)
