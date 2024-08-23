"""Module to perform all AI related steps for the OpenAgri project.
"""

# Standard imports
import argparse
import csv
import gc
import logging
import os
import time
from pathlib import Path
from random import randint

# Third party imports
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.color
import skimage.filters
import skimage.io
import skimage.transform
import torch
import torch.nn as nn
from torchvision import transforms

# Local imports
import constants


class Filter(nn.Module):
    def __init__(self):
        super(Filter, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class OpenAgriAI:
    """Class to perform all AI related steps for the OpenAgri project.
    """

    def __init__(self):
        self.logger = logging.getLogger("OpenAgriAIModule")

    @staticmethod
    def _split_image(source_image: np.ndarray, size: int) -> list[np.ndarray]:
        """Split the image in blocks of size * size

        Args:
            source_image (np.ndarray): Source image
            size (int): Dimension of the returned image parts

        Returns:
            list[np.ndarray]: The blocks in which the image was split
        """
        data = []
        y = 0
        while y < constants.Y_END - size:
            x = 0
            while x < constants.X_END - size:
                x_end = x + size if x + size < constants.X_END else constants.X_END
                y_end = y + size if y + size < constants.Y_END else constants.Y_END
                data.append(source_image[y:y_end, x:x_end])
                x += size - constants.OVERLAP
            y += size - constants.OVERLAP
        return data

    @staticmethod
    def _downsample(input_image: np.ndarray) -> np.ndarray:
        downscaled = skimage.transform.resize(
            input_image, (64, 64), anti_aliasing=True)
        return downscaled

    @staticmethod
    def _downsample_opencv(input_image: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(input_image, (23, 23), 0)
        downscaled = cv2.resize(blur, (64, 64), cv2.INTER_AREA)
        return downscaled

    def _inference(self, model_index: int, data, threshold):
        if len(data) == 0:
            return [], 0
        data_ = [self._downsample(np.array(model_index)) for i in data]
        start = time.time()
        model = Filter()
        model.cuda()
        model.load_state_dict(torch.load(f"./trained_models/{model_name}.pth"))
        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        transform = transforms.Compose([transforms.ToTensor()])
        input = torch.cat(
            [transform(i).float().unsqueeze(0).to(device) for i in data_], 0)
        output = model(input)
        output = output.to(device).data.cpu().numpy()
        filtered_data = []
        for i in range(len(input)):
            if output[i] > threshold:
                filtered_data.append(data[i])
        t = time.time() - start
        return filtered_data, t

    def _predict_data(self, data: list[np.ndarray], thresholds: list[float]):
        """ Function that does Colorado beetle detection on the input data: 
        data -> model1 -> filter -> model2 -> filter -> model3 -> filter -> model4

        Args:
            data (list[np.ndarray]): _description_
            thresholds (list[float]): _description_

        Returns:
            _type_: Patches with positive detection
        """
        torch.cuda.empty_cache()
        processing_time = 0
        all_data = []
        for model_index in range(1, 5):
            filtered_data, t_ = self._inference(
                model_index, data, thresholds[model_index-1])
            all_data.append(filtered_data)
            processing_time += t_

            if (model_index == 4):
                break

            start = time.time()
            data = cut_in_parts(filtered_data)
            processing_time += time.time() - start
        return filtered_data, processing_time, all_data

    def _evaluate_metrics(self, image_path: Path, label_path: Path):
        """ Evaluate performance and speed

        Args:
            image_path (Path): Path to the image
            label_path (Path): Path to the ground truth labels
        """
        with open('results.csv', mode='w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["p_TP", "p_TN", "p_FP", "p_FN",
                            "p_Accuracy", "p_Precision", "p_Recall", "p_Time"])

            for filename in image_path.iterdir():
                if filename.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'):
                    self.logger.error("Invalid input file: %s", filename)
                    continue

                self.logger.info("Evaluating %s", filename)
                input_image = skimage.io.imread(filename)
                data_512 = self._split_image(input_image, 512)
                data_64 = self._split_image(input_image, 64)

                # gt = get_ground_truth(
                #     f"{label_path}/{filename}.txt", 10168, 64, 7952)
                gc.collect()
                torch.cuda.empty_cache()

                result_pipeline, pt, _ = self._predict_data(
                    data_512, [0.5, 0.5, 0.5, 0.5])
                # pipeline = convert_result_to_indexed_array(
                #     result_pipeline, data_64)
                # ptp, ptn, pfp, pfn = calculate_stats(gt, pipeline)
                # pa, pp, pr = calculate_metrics(ptp, ptn, pfp, pfn)
                # print("------------------------------------------")
                # print("PIPELINE")
                # print(f"tp: {ptp}\ttn: {ptn}\tfp: {pfp}\tfn: {pfn}")
                # print("accuracy: "+str(pa))
                # print("precision: "+str(pp))
                # print("recall: "+str(pr))

                # writer.writerow([ptp, ptn, pfp, pfn, pa, pp, pr, pt])

                return

    def evaluate(self, image_path: Path):
        """Evaluate the images from the file path against the openagri AI model

        Args:
            image_path (Path): Path containing the evaluation images
        """
        self.logger.info("Start evaluation on path %s", image_path)
        self._evaluate_metrics(image_path, None)


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(format='[%(levelname)s %(name)s @ %(asctime)s] %(message)s',
                        level=logging.DEBUG)
    logging.getLogger("PIL").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    # Parse the script arguments:
    parser = argparse.ArgumentParser(
        description='Evaluate image(s) from Colorado Beetles')
    parser.add_argument('--input', dest="input", metavar='path', required=True,
                        help='The path to the images')
    script_arguments = parser.parse_args()

    # Initialise and start the OpenAgri AI module:
    ai_module = OpenAgriAI()
    ai_module.evaluate(Path(script_arguments.input))
