"""A script to verify if a tflite model is quantized
"""

import tensorflow as tf


def check_quantization(tflite_model_path):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

    # Get details of input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check data types of input and output tensors
    print("Input Tensor Types:")
    for input_detail in input_details:
        print(f"  - {input_detail}")

    print("\nOutput Tensor Types:")
    for output_detail in output_details:
        print(f"  - {output_detail}")

    # Check if the model is quantized by inspecting tensor data types
    if input_details[0]["dtype"] in [tf.int8, tf.uint8] and output_details[0][
        "dtype"
    ] in [tf.int8, tf.uint8]:
        print("\nThe model is quantized.")
    else:
        print("\nThe model is not quantized.")


# Path to your TFLite model
tflite_model_path = "trained_models/model64/model.quant.tflite"

check_quantization(tflite_model_path)
