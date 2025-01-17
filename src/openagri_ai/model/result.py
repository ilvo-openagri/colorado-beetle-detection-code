import dataclasses
import numpy as np
from typing import Tuple


@dataclasses.dataclass
class AiResult:
    """Class for storing the result of an image inference."""

    image: np.ndarray
    center_x: int
    center_y: int
    confidence: float
    annotation: str = ""


@dataclasses.dataclass
class ModelResult:
    """Class for storing the result of the OpenAgri model."""

    leaves: list[AiResult]
    beetles: dict[
        int, list[AiResult]
    ]  # Maps detection dimension on the detected beetles
    execution_time: float = 0.0
