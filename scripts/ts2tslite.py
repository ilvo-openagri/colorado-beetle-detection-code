"""Module to perform all AI related steps for the OpenAgri project.
"""

# Standard imports
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np

# Third party imports
import tensorflow as tf

DATASET = []


def representative_dataset():
    a = []
    for data in DATASET:
        img = cv2.imread(data)
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        img = img.astype(np.float32)
        a.append(img)
    a = np.array(a)
    print(a.shape)  # a is np array of 160 3D images
    for i in tf.data.Dataset.from_tensor_slices(a).batch(1).take(100):
        yield [i]


def tensorflow_to_tensorflowlite(
    tensorflow_model_path: Path, tensorflow_lite_model_path: Path, quantize: bool
):
    """Create a lite model out of the default tensorflow/keras models

    Args:
        tensorflow_model_path (Path): Path to the input tensorflow model
        tensorflow_lite_model_path (Path): Path to the output tensorflow lite model
        quantize (bool): Quantize the model to full int representation?
    """
    logger = logging.getLogger("TS2TSLITE")
    logger.info(
        "Starting conversion of model %s from tensorflow to tensorflow lite",
        tensorflow_model_path,
    )

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(
        tensorflow_model_path.as_posix()
    )
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_model = converter.convert()

    logger.info(type(tflite_model))
    # Save the model
    with open(tensorflow_lite_model_path, "wb") as tflite_fd:
        tflite_fd.write(tflite_model)

    logger.info("Conversion completed")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        format="[%(levelname)s %(name)s @ %(asctime)s] %(message)s", level=logging.DEBUG
    )
    # Parse the script arguments:
    parser = argparse.ArgumentParser(
        description="Convert a tensorflow model to the tensorflow lite variant"
    )
    parser.add_argument(
        "--input",
        dest="input",
        metavar="path",
        required=True,
        help="The tensorflow model that needs conversion.",
    )
    parser.add_argument(
        "--output",
        dest="output",
        metavar="path",
        required=True,
        help="The tensorflow lite model destination.",
    )
    parser.add_argument(
        "--quantize",
        dest="quantize",
        metavar="bool",
        required=False,
        help="Perform quantization of the model?",
        default=False,
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        metavar="path",
        required=False,
        help="Path to a representable dataset for quantization",
        default="trainingset/256by256/validate/images/",
    )

    script_arguments = parser.parse_args()
    if script_arguments.quantize:
        DATASET = [
            f.as_posix()
            for f in Path(script_arguments.dataset).iterdir()
            if f.is_file()
        ]

    # Perform the conversion:
    tensorflow_to_tensorflowlite(
        tensorflow_model_path=Path(script_arguments.input),
        tensorflow_lite_model_path=Path(script_arguments.output),
        quantize=script_arguments.quantize,
    )
