"""Module containing the Raspberry Pi with Hailo8L AI shield openagri configuration options
"""

from . import OpenagriConfig

config = OpenagriConfig(
    DETECTION_MODEL="HAILO", DETECTION_MODEL_DIM=640, LEAVE_SEGMENTATION_MODEL="NCNN"
)
