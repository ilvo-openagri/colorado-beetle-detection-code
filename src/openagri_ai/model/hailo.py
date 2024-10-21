from .base import _BaseAIModel
from .result import AiResult, ModelResult
import platform

if platform.machine() == "aarch64":
    from hailo_platform import (
        HEF,
        VDevice,
        HailoStreamInterface,
        InferVStreams,
        ConfigureParams,
        InputVStreamParams,
        OutputVStreamParams,
        InputVStreams,
        OutputVStreams,
        FormatType,
    )
import cv2
from .version import TensorFlowVersion
import numpy as np


class HailoModel(_BaseAIModel):

    def __init__(self, confidence: float):
        super().__init__()
        self._confidence = confidence
        self._output_dimension = 640

        # The target can be used as a context manager ("with" statement) to ensure it's released on time.
        # Here it's avoided for the sake of simplicity
        self.target = VDevice()

        # Loading compiled HEFs to device:
        hef_path = "trained_models/model_yolov8n_640/model.hef"
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
        matching_image_set = []
        self.logger.info("Start evaluation")

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
                            [active_image.image], dtype=np.float32
                        )
                    }
                    infer_results = infer_pipeline.infer(input_data)
                    for hailo_result in infer_results[output_vstream_info.name][0][0]:
                        if hailo_result[4] < self._confidence:
                            continue
                        matching_image_set.append(
                            AiResult(
                                image=None,
                                center_x=active_image.center_x
                                - self._output_dimension / 2
                                + self._output_dimension * hailo_result[1],
                                center_y=active_image.center_y
                                - self._output_dimension / 2
                                + self._output_dimension * hailo_result[0],
                                confidence=hailo_result[4],
                            )
                        )

        self.logger.info("Completed")

        return ModelResult(leaves=leave_images, beetles={64: matching_image_set})

    def close(self):
        """Release all resources on close"""
        self.logger.info("Releasing all resources")
        self.target.release()
