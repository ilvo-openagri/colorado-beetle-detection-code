import numpy as np
import cv2
from multiprocessing import Process
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
import tensorflow as tf
import time

# The target can be used as a context manager ("with" statement) to ensure it's released on time.
# Here it's avoided for the sake of simplicity
target = VDevice()

# Loading compiled HEFs to device:
model_name = "cb_modelyolosegmentation"
hef_path = "trained_models/model_yolov8n_segmentation/model.hef"
hef = HEF(hef_path)

# Configure network groups
configure_params = ConfigureParams.create_from_hef(
    hef=hef, interface=HailoStreamInterface.PCIe
)
network_groups = target.configure(hef, configure_params)
network_group = network_groups[0]
network_group_params = network_group.create_params()

# Create input and output virtual streams params
input_vstreams_params = InputVStreamParams.make(
    network_group, format_type=FormatType.FLOAT32
)
output_vstreams_params = OutputVStreamParams.make(network_group)

# Define dataset params
input_vstream_info = hef.get_input_vstream_infos()[0]
output_vstream_info = hef.get_output_vstream_infos()[0]
image_height, image_width, channels = input_vstream_info.shape
print(image_height, image_width, channels)
num_of_images = 1
low, high = 2, 20

# Generate random dataset
if True:
    input_image = cv2.imread("test_images/DSC00191.JPG")
else:
    input_image = np.random.randint(
        low, high, (num_of_images, image_height, image_width, channels)
    ).astype(np.float32)
    print(input_image)


# HAILO inference:
with InferVStreams(
    network_group, input_vstreams_params, output_vstreams_params
) as infer_pipeline:
    resized_img = cv2.resize(
        input_image, (image_height, image_width), interpolation=cv2.INTER_AREA
    )
    input_data = {
        input_vstream_info.name: np.expand_dims(resized_img, axis=0).astype(np.float32)
    }
    with network_group.activate(network_group_params):
        print("Starting inference")
        hailo_start_time = time.time()
        infer_results = infer_pipeline.infer(input_data)
        hailo_end_time = time.time()
        hailo_result = infer_results[output_vstream_info.name]

# print(filter_result)
import pickle

with open("tmp/hailo_segmentation.pickle", "wb") as handle:
    pickle.dump(infer_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
