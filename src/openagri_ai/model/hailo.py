from ..constants import MODEL_HAILO_256, MODEL_HAILO_640
from ..settings import get_config
from .base import _BaseAIModel
from .result import AiResult, ModelResult
import cv2
import numpy as np
import platform
import time

if platform.machine() == "aarch64":
    from hailo_platform import (
        HEF,
        VDevice,
        HailoStreamInterface,
        InferVStreams,
        ConfigureParams,
        InputVStreamParams,
        OutputVStreamParams,
        FormatType,
    )


class HailoModel(_BaseAIModel):
    """YoloV8 model which can be run on the Hailo8L AI shield"""

    def __init__(self, confidence: float, device: str = "cpu"):
        super().__init__(device=device)
        config = get_config()
        self._confidence = confidence
        self._output_dimension = config.DETECTION_MODEL_DIM

        # The target can be used as a context manager ("with" statement) to ensure it's released on time.
        # Here it's avoided for the sake of simplicity
        self.target = VDevice()

        # Loading compiled HEFs to device:
        if self._output_dimension == 256:
            hef_path = MODEL_HAILO_256
        else:
            hef_path = MODEL_HAILO_640
        self.hef = HEF(hef_path)

        # Configure network groups
        configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_groups = self.target.configure(self.hef, configure_params)
        self.network_group_params = self.network_groups[0].create_params()

        # Create input and output virtual streams params
        self.input_vstreams_params = InputVStreamParams.make(
            self.network_groups[0], format_type=FormatType.FLOAT32
        )
        self.output_vstreams_params = OutputVStreamParams.make(self.network_groups[0])

    def evaluate(self, image_file: cv2.typing.MatLike) -> ModelResult:
        """Evaluate the images from the file path against the openagri Hailo AI model

        Args:
            image_file (cv2.typing.MatLike): Source image
        """
        matching_image_set = []
        self.logger.info("Start evaluation")
        start_time = time.time()

        leave_images = self._collect_leaves(image_file, self._output_dimension)
        input_vstream_info = self.hef.get_input_vstream_infos()[0]
        output_vstream_info = self.hef.get_output_vstream_infos()[0]
        with InferVStreams(
            self.network_groups[0],
            self.input_vstreams_params,
            self.output_vstreams_params,
        ) as infer_pipeline:
            with self.network_groups[0].activate(self.network_group_params):
                for active_image in leave_images:
                    input_data = {
                        input_vstream_info.name: np.array(
                            [cv2.cvtColor(active_image.image, cv2.COLOR_BGR2RGB)],
                            dtype=np.float32,
                        )
                    }
                    infer_results = infer_pipeline.infer(input_data)
                    for hailo_result in infer_results[output_vstream_info.name][0][0]:
                        if hailo_result[4] < self._confidence:
                            continue
                        matching_image_set.append(
                            AiResult(
                                image=None,
                                loc_x=active_image.loc_x
                                + self._output_dimension * hailo_result[1],
                                loc_y=active_image.loc_y
                                + self._output_dimension * hailo_result[0],
                                confidence=hailo_result[4],
                                width=hailo_result[3],
                                height=hailo_result[2],
                            )
                        )

        end_time = time.time()

        self.logger.info("Completed. Matched %d tiles", len(matching_image_set))

        return ModelResult(
            execution_time=end_time - start_time,
            leaves=leave_images,
            beetles={64: matching_image_set},
        )

    def close(self):
        """Release all resources on close"""
        self.logger.info("Releasing all resources")
        self.target.release()
