# Coloradobeetles
This code detects Colorado Beetles in high resolution images.

## Getting started
First make sure that the python environment has been created.
You can run this in the system python environment, but this is not required.
It is better to create a python virtual environment.

For reference, this project was tested with python3.11. If you use a newer python version, ensure that the packages are available for the python version.

To create a virtual environment, execute following command:
```Powershell
python.exe -m venv .venv
```

Active the virtual environment:
```Powershell
.\.venv\Scripts\activate
# This should activate the virtual environment. To verify, execute:
(Get-Command python).Path
```

To install the required python packages, the requirements.txt was created.
This file can be used to easily install the dependencies.
```Powershell
python.exe -m pip install -r requirements.txt
```

## Evaluate an image:
To evaluate an image, use the evaluate.py script.

```
usage: evaluate.py [-h] --input path

Evaluate image(s) from Colorado Beetles

options:
  -h, --help    show this help message and exit
  --input path  The path to the images
```

```Powershell
python.exe evaluate.py --input .\test_images\sectie32.JPG
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