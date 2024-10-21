from dataclasses import dataclass


@dataclass
class OpenagriConfig:
    DETECTION_MODEL: str
    TF_VERSION: str
    LEAVE_SEGMENTATION_MODEL: str
