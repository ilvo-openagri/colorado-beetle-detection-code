import dataclasses
import numpy as np
from typing import Tuple


@dataclasses.dataclass
class AiResult:
    """Class for storing the result of an image inference."""

    image: np.ndarray
    loc_x: int
    loc_y: int
    width: int
    height: int
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
