# Installation of the Colorado Beetle model
This installation readme goes over the steps requried to run the Colorado Beetle detection model.

## Dependencies
Install and activate the required dependencies on your machine.

### Python virtual environment
First make sure that the python environment has been created.
You can run this in the system python environment, but this is not advisable.
It is better to create a python virtual environment.

For reference, this project was tested with python3.11. If you use a newer python version, ensure that the packages are available for the python version.

To create a virtual environment, execute following command:
```shell
# Windows:
python.exe -m venv .venv
# Linux:
python -m venv .venv
```

Active the virtual environment:
```shell
# Windows:
.\.venv\Scripts\activate
# Linux:
source .venv/bin/activate
```

To install the required python packages, the requirements.txt was created.
This file can be used to easily install the dependencies.
```shell
# Windows:
python.exe -m pip install -r requirements.txt
# Linux:
python -m pip install -r requirements.txt
```

### Hailo8L AI shield dependencies (Raspberry PI AI kit)
Install the necessary hailo dependencies.
```shell
sudo apt install hailo-full
```

Download the hailort python wheel from the hailo developer zone to the lib directory.

```shell
(.venv) jbauwens@rpiILVO:~/Code/openagri_ai $ ll lib/
total 8440
drwxr-xr-x  3 jbauwens jbauwens    4096 Oct  3 15:52 .
drwxr-xr-x 11 jbauwens jbauwens    4096 Oct  3 15:18 ..
-rwxr-xr-x  1 jbauwens jbauwens 8626214 Oct  3 15:51 hailort-4.18.0-cp311-cp311-linux_aarch64.whl
```

Additional packages:
```shell
apt-get install libgl1
```