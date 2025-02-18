"""Module containing the pipeline openagri configuration options
"""

from . import OpenagriConfig

config = OpenagriConfig(DETECTION_MODEL="PIPELINE", TF_VERSION="default", DEVICE="CPU")
