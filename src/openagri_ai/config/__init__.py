from dataclasses import dataclass


@dataclass
class OpenagriConfig:
    DETECTION_MODEL: str
    LEAVE_SEGMENTATION_MODEL: str
    DETECTION_MODEL_DIM: int = 640
    TF_VERSION: str = "default"
