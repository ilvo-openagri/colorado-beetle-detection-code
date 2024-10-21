from . import OpenagriConfig

config = OpenagriConfig(
    DETECTION_MODEL="YOLO",
    TF_VERSION="default",
    LEAVE_SEGMENTATION_MODEL="PYTORCH",
    DETECTION_MODEL_DIM=256,
)
