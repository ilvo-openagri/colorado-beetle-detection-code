## Transform the tensorflow model to tensorflow lite:
Tensorflow lite models are much more efficient on edge devices such as the raspberry PI.
To translate a model, use the ts2tslite.py script.

```Powershell
usage: ts2tslite.py [-h] --input path --output path

Convert a tensorflow model to the tensorflow lite variant

options:
  -h, --help     show this help message and exit
  --input path   The tensorflow model that needs conversion.
  --output path  The tensorflow lite model destination.
```

## YOLO model on 256x256 Colorado Beetle images
Make sure that you put your trained YOLOv8 .pt file into this directory.
