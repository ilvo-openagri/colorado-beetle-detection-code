# Overview
Here is an overview of the scripts within this project.

## Perform model training

### Pre-process the data
To pre-process the collected data, before training the model, use the script at scripts/training/preprocess_training.py.
The training/validation/testing distribution is by default 0,70/0,15/0,15.

```shell
usage: preprocess_training.py [-h] --input path --output path --labels path [--format path] [--size SIZE] [--split | --no-split]

Pre-process the collected images before training the model.

options:
  -h, --help     show this help message and exit
  --input path   Input path of training data
  --output path  Output path of training data
  --labels path  Path of the training labels
  --format path  Format of the training data
  --size SIZE    Dimension of the output images
  --split
  --no-split
```

### Train the model
To train a model, use the script at scripts/training/train.py.
In order to switch between the segmentation/object detection model, change the target variable.

## Transform the tensorflow model to tensorflow lite:
Tensorflow lite models are much more efficient on edge devices such as the raspberry PI.
To translate a model, use the ts2tslite.py script.

```shell
usage: ts2tslite.py [-h] --input path --output path

Convert a tensorflow model to the tensorflow lite variant

options:
  -h, --help     show this help message and exit
  --input path   The tensorflow model that needs conversion.
  --output path  The tensorflow lite model destination.
```
