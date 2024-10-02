from enum import Enum


class ModelVersion(Enum):
    """Versions of the models to parse 256x256 image parts"""

    PIPELINE = "PIPELINE"
    YOLO = "YOLO"


class TensorFlowVersion(Enum):
    """Tensorflow versions that can be selected for the model"""

    DEFAULT = "default"
    LITE = "lite"
