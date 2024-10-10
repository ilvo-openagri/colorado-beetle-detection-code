import json
import os

import numpy as np
import tensorflow as tf
from IPython.display import SVG
from matplotlib import patches
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.python.eager.context import eager_mode

# import the hailo sdk client relevant classes
from hailo_sdk_client import ClientRunner, InferenceContext, CalibrationDataType

IMAGES_TO_VISUALIZE = 5

chosen_hw_arch = "hailo8l"
model_name = "cb_model64"
model_path = "../trained_models/model64/"
model_tflite = "../trained_models/model64/saved_model.tflite"
model_har = "../trained_models/model64/saved_model.har"
model_hef = "../trained_models/model64/saved_model.hef"
quantization_images = "../../../Datasets/Colorado_complete_2019/Gelabeld/images_64"
image_size = 64


# First, we will prepare the calibration set. Resize the images to the correct size and crop them.
def preproc(image, output_height=224, output_width=224, resize_side=256):
    """imagenet-standard: aspect-preserving resize to 256px smaller-side, then central-crop to 224px"""
    with eager_mode():
        h, w = image.shape[0], image.shape[1]
        scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
        resized_image = tf.compat.v1.image.resize_bilinear(
            tf.expand_dims(image, 0), [int(h * scale), int(w * scale)]
        )
        cropped_image = tf.compat.v1.image.resize_with_crop_or_pad(
            resized_image, output_height, output_width
        )

        return tf.squeeze(cropped_image)


converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
]
tflite_model = (
    converter.convert()
)  # may cause warnings in jupyter notebook, don't worry.
with tf.io.gfile.GFile(model_tflite, "wb") as f:
    f.write(tflite_model)

# Parsing the model to Hailo format
runner = ClientRunner(hw_arch=chosen_hw_arch)
hn, npz = runner.translate_tf_model(model_tflite, model_name)
runner.save_har(model_har)

# Get the quantization image list:
quantization_images_list = [
    img_name
    for img_name in os.listdir(quantization_images)
    if os.path.splitext(img_name)[1] == ".JPG"
]

calib_dataset = np.zeros((len(quantization_images_list), image_size, image_size, 3))
for idx, img_name in enumerate(sorted(quantization_images_list)):
    img = np.array(Image.open(os.path.join(quantization_images, img_name)))
    img_preproc = preproc(img, 64, 64, 64)
    calib_dataset[idx, :, :, :] = img_preproc.numpy()

np.save("calib_set.npy", calib_dataset)

alls = "normalization1 = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])\n"
runner = ClientRunner(har=model_har)
runner.load_model_script(alls)
runner.optimize(calib_dataset)

hef = runner.compile()
with open(model_hef, "wb") as f:
    f.write(hef)
