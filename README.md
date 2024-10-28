# Coloradobeetles
This code detects Colorado Beetles in high resolution images.

## Install the dependencies
See the documentation [here](documentation/INSTALL.md)

## Evaluate an image:
To evaluate an image, use the evaluate.py script.

```
usage: evaluate.py [-h] --input path [--output path] --tensorflow path

Evaluate image(s) from Colorado Beetles

options:
  -h, --help         show this help message and exit
  --input path       The path to the image
  --output path      The path where an annotated output image can be stored.
  --tensorflow path  Select the tensorflow version (lite or default)
```

```Powershell
python.exe evaluate.py --input .\test_images\sectie32.JPG --output .\test_images\sectie32_annotated. JPG --tensorflow lite
```

Example output
```
...
[INFO OpenAgriAIModule @ 2024-08-28 15:00:03,538] Start evaluation on path test_images\sectie32.JPG
[INFO OpenAgriAIModule @ 2024-08-28 15:00:07,244] Kmeans matches 89 out of 150 image pieces
[INFO OpenAgriAIModule @ 2024-08-28 15:00:22,625] Pipeline matches 40 out of 356 image pieces (size 256)
[INFO OpenAgriAIModule @ 2024-08-28 15:00:29,367] Pipeline matches 91 out of 160 image pieces (size 128)
[INFO OpenAgriAIModule @ 2024-08-28 15:00:29,367] Completed
```

## Create a python package
To use the AI model in other projects, you can create a python package from the source code.
See the documentation [here](documentation/PACKAGING.md)