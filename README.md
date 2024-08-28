# Coloradobeetles
This code detects Colorado Beetles in high resolution images.

## Getting started
First make sure that the python environment has been created.
You can run this in the system python environment, but this is not required.
It is better to create a python virtual environment.

For reference, this project was tested with python3.11. If you use a newer python version, ensure that the packages are available for the python version.

To create a virtual environment, execute following command:
```Powershell
python -m venv .venv
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
