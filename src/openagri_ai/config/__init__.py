from dataclasses import dataclass


@dataclass
class OpenagriConfig:
    """Dataclass for the configuration options for the OpenAgri AI project"""

    DETECTION_MODEL: str  # See ModelVersion (model.version.py)
    LEAVE_SEGMENTATION_MODEL: str = ""  # PYTORCH or NCNN
    DETECTION_MODEL_DIM: int = 640  # Output dimension of the model.
    TF_VERSION: str = "default"  # See TensorFlowVersion (model.version.py)
    DEVICE: str = "CPU"
