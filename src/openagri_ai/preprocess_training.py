import argparse
import glob
from joblib import Parallel, delayed
import json
from pathlib import Path
import logging
import math
import numpy as np
import os
import skimage.io
import skimage.transform
import sys

TRAINING_FORMATS = (
    "YOLO",
    "JSON",
)


def pad(array, target_shape):
    return np.pad(
        array,
        [(0, target_shape[i] - array.shape[i]) for i in range(len(array.shape))],
        "constant",
    )


def split_image(image_name: str, image_file: np.ndarray, labels: list, piece_size: int):
    resultset = []
    y = 0
    x_max_dim = len(image_file[0])
    y_max_dim = len(image_file)

    while y < y_max_dim:
        x = 0
        while x < x_max_dim:
            x_end = x + piece_size if x + piece_size < x_max_dim else x_max_dim
            y_end = y + piece_size if y + piece_size < y_max_dim else y_max_dim
            image_piece = pad(image_file[y:y_end, x:x_end], (piece_size, piece_size, 3))
            # Check if there are any relevant labels for this piece
            image_piece_labels = []
            for label in labels:
                x_center, y_center, width, height = [round(float(s)) for s in label[1:]]
                if (
                    x_center >= x
                    and x_center < x_end
                    and y_center >= y
                    and y_center < y_end
                ):
                    image_piece_labels.append(
                        [
                            label[0],
                            float(x_center - x) / piece_size,
                            float(y_center - y) / piece_size,
                            float(width) / piece_size,
                            float(height) / piece_size,
                        ]
                    )
            resultset.append(
                {
                    "image": image_piece,
                    "labels": image_piece_labels,
                    "name": f"{image_name}_{len(resultset)}",
                }
            )
            x += piece_size
        y += piece_size
    return resultset


def extract_labels(label_format: str, label_file: Path):
    logger = logging.getLogger("EXTRACT_LABELS")
    labels = []
    if label_format == "YOLO":
        with open(label_file, "r") as label_file_descriptor:
            labels = label_file_descriptor.read().splitlines()
            labels_cleaned = []
            for label in labels:
                labels_cleaned.append(label.split("\t"))
            labels = labels_cleaned
    elif label_format == "JSON":
        labels_cleaned = []
        with open(label_file, "r") as label_file_descriptor:
            labels = json.load(label_file_descriptor)
            for shape in labels["shapes"]:
                labels_cleaned.append(
                    [
                        0,
                        (shape["points"][1][0] + shape["points"][0][0]) / 2.0,
                        (shape["points"][1][1] + shape["points"][0][1]) / 2.0,
                        abs(shape["points"][1][0] - shape["points"][0][0]),
                        abs(shape["points"][1][1] - shape["points"][0][1]),
                    ]
                )
        labels = labels_cleaned

    return labels


def preprocess(
    training_input_path: Path,
    training_output_path: Path,
    format: str,
    image_size: int,
    store_empty: bool = True,
    split: list = [0.7, 0.15, 0.15],
    split_names: list = ["train", "validate", "test"],
):
    logger = logging.getLogger("PREPROCESS_TRAINING")

    if format not in TRAINING_FORMATS:
        logger.error("Unexpected input format. Possibilities: %s", TRAINING_FORMATS)
        sys.exit(1)

    # Detect the number of image files:
    files_in_dir = len(os.listdir(f"{training_input_path}/images"))

    # Determine the split count, on which the files need to be divided over the three directories:
    split_value = 0
    for split_index, split_entry in enumerate(split):
        split[split_index] = math.ceil(files_in_dir * split_entry + split_value)
        split_value = split[split_index]

    # Read the files one by one
    for image_file_index, image_file in enumerate(
        os.listdir(f"{training_input_path}/images")
    ):
        image_path = Path(f"{training_input_path}/images/{image_file}")
        if image_path.suffix.lower() not in (".jpg", ".png"):
            continue

        logger.info("%d/%d: %s", image_file_index + 1, files_in_dir, image_file)
        split_index = 0
        while image_file_index > split[split_index]:
            split_index += 1

        # Extract the labels belonging to this image
        labels = []
        matching_files = glob.glob(f"{training_input_path}/labels/{image_path.stem}.*")
        if matching_files:
            label_file = matching_files[0]
            labels = extract_labels(format, label_file)

        # Split the image in pieces of image_size x image_siwe
        result_images = split_image(
            image_path.stem, skimage.io.imread(image_path), labels, image_size
        )

        # Create the destination directories:
        os.makedirs(
            f"{training_output_path}/{split_names[split_index]}/images/", exist_ok=True
        )
        os.makedirs(
            f"{training_output_path}/{split_names[split_index]}/labels/", exist_ok=True
        )

        # Write the images to the output:
        for result_image in result_images:
            if not store_empty and len(result_image["labels"]) == 0:
                continue
            skimage.io.imsave(
                f"{training_output_path}/{split_names[split_index]}/images/{result_image['name']}.JPG",
                result_image["image"],
            )
            if result_image["image"].shape != (image_size, image_size, 3):
                logger.error(
                    "Exception on the image piece size. Got %s",
                    result_image["image"].shape,
                )
                sys.exit(1)
            with open(
                f"{training_output_path}/{split_names[split_index]}/labels/{result_image['name']}.txt",
                "w",
            ) as outfile:
                for label in result_image["labels"]:
                    outfile.write("\t".join(map(str, label)) + "\n")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        format="[%(levelname)s %(name)s @ %(asctime)s] %(message)s", level=logging.DEBUG
    )
    logging.getLogger("PIL").setLevel(logging.ERROR)

    # Parse the script arguments:
    parser = argparse.ArgumentParser(
        description="Convert a tensorflow model to the tensorflow lite variant"
    )
    parser.add_argument(
        "--input",
        dest="input",
        metavar="path",
        required=True,
        help="Input path of training data",
    )
    parser.add_argument(
        "--output",
        dest="output",
        metavar="path",
        required=True,
        help="Output path of training data",
    )
    parser.add_argument(
        "--format",
        dest="format",
        metavar="path",
        default="YOLO",
        help="Format of the training data",
    )
    parser.add_argument(
        "--size",
        dest="size",
        type=int,
        default=640,
        help="Dimension of the output images",
    )
    script_arguments = parser.parse_args()

    # Perform the conversion:
    preprocess(
        training_input_path=Path(script_arguments.input),
        training_output_path=Path(script_arguments.output),
        format=script_arguments.format,
        store_empty=False,
        image_size=script_arguments.size,
    )
