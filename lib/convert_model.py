# General imports used throughout the tutorial
import tensorflow as tf
from IPython.display import SVG

# import the ClientRunner class from the hailo_sdk_client package
from hailo_sdk_client import ClientRunner

onnx_model_name = "beetledetection"
onnx_path = "../trained_models/yolo256.onnx"
chosen_hw_arch = "hailo8l"

runner = ClientRunner(hw_arch=chosen_hw_arch)
hn, npz = runner.translate_onnx_model(
    onnx_path,
    onnx_model_name,
    # start_node_names=["input.1"],
    # end_node_names=["192"],
    # net_input_shapes={"input.1": [1, 3, 224, 224]},
)
