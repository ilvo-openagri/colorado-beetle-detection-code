"""Module to perform all AI related steps for the OpenAgri project.
"""

# Standard imports
import argparse
import logging
from pathlib import Path

# Third party imports
import tensorflow as tf


def tensorflow_to_tensorflowlite(tensorflow_model_path: Path, tensorflow_lite_model_path: Path):
    """Create a lite model out of the default tensorflow/keras models

    Args:
        tensorflow_model_path (Path): Path to the input tensorflow model
        tensorflow_lite_model_path (Path): Path to the output tensorflow lite model
    """
    logger = logging.getLogger("TS2TSLITE")
    logger.info(
        "Starting conversion of model %s from tensorflow to tensorflow lite",
        tensorflow_model_path)

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(
        tensorflow_model_path.as_posix())
    tflite_model = converter.convert()

    # Save the model
    with open(tensorflow_lite_model_path, 'wb') as tflite_fd:
        tflite_fd.write(tflite_model)

    logger.info("Conversion completed")


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(format='[%(levelname)s %(name)s @ %(asctime)s] %(message)s',
                        level=logging.DEBUG)
    # Parse the script arguments:
    parser = argparse.ArgumentParser(
        description='Convert a tensorflow model to the tensorflow lite variant')
    parser.add_argument('--input', dest="input", metavar='path', required=True,
                        help='The tensorflow model that needs conversion.')
    parser.add_argument('--output', dest="output", metavar='path', required=True,
                        help='The tensorflow lite model destination.')
    script_arguments = parser.parse_args()

    # Perform the conversion:
    tensorflow_to_tensorflowlite(
        tensorflow_model_path=Path(script_arguments.input),
        tensorflow_lite_model_path=Path(script_arguments.output),
    )
