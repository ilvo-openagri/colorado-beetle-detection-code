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


### Using the HAILO shield (Raspberry PI AI kit)
Install the necessary hailo dependencies.
```
sudo apt install hailo-full
```

Download the hailort python wheel from the hailo developer zone to the lib directory.

```
(.venv) jbauwens@rpiILVO:~/Code/openagri_ai $ ll lib/
total 8440
drwxr-xr-x  3 jbauwens jbauwens    4096 Oct  3 15:52 .
drwxr-xr-x 11 jbauwens jbauwens    4096 Oct  3 15:18 ..
-rwxr-xr-x  1 jbauwens jbauwens 8626214 Oct  3 15:51 hailort-4.18.0-cp311-cp311-linux_aarch64.whl
```

### Additions for Linux

Additional packages:
```shell
apt-get install libgl1
```

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

## Create a python package
To use the AI model in other projects, we create a python package from the source code.
The resulting package can be uploaded to the gitlab package manager, to be later used by pip (install).

### Generate the python package
The description of the python package is done through two files:
- pyproject.toml: Package descritpion
- MANIFEST.in: Manifest file which includes/excludes specific files and directories

To generate the python package, execute following commands:
```
pip install -q build # Ensures that you have the build dependency in python. Only needs to be run once.
python -m build # Create the python package
```

The resulting files can be found under the dist/ directory.

### Deploy the python package to the gitlab package registry
First create deploy tokens to read/write the package registry: https://gitlab.ilvo.be/jbauwens/openagri_ai/-/settings/repository
To deploy the package, the write_package_registry scope needs to be granted to the deploy key.
The deploy key needs to be stored in ~/.pypirc.

```
[distutils]
index-servers = 
    ilvo-gitlab-openagri-ai

[ilvo-gitlab-openagri-ai]
repository = https://gitlab.ilvo.be/api/v4/projects/719/packages/pypi
username = <deploy-key-name>
password = <deploy-key-password>
```

To upload the package to the package registry, execute following command/
```
python3 -m twine upload --repository ilvo-gitlab-openagri-ai dist/*

# Output:
Uploading distributions to https://gitlab.ilvo.be/api/v4/projects/719/packages/pypi
Uploading OpenAgriAiModule-1.0-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.4/16.4 kB • 00:00 • 591.0 kB/s
Uploading openagriaimodule-1.0.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14.8/14.8 kB • 00:00 • ?
```

You can only store the package with the same version once. To upload a new version, create a package with a different version.

## YOLO model on 256x256 Colorado Beetle images
Make sure that you put your trained YOLOv8 .pt file into this directory.

### Prepare your environment for the HAILO AI shield:
Activate the virtual environment, and install the requirements
```
cd openagri-ai
python3 -m venv .venv
python3 -m pip install -r requirements.txt
```

Clone the hailo model zoo repository
```
https://github.com/hailo-ai/hailo_model_zoo.git
```

Link the git repository into your virtual environment. Dirty, but the code did not seem to work otherwise (file not found exception). Hopefully this will be patched in a future release of the Hailo Model Zoo.
```
ln -sf /home/jbauwens/Documents/Code/hailo_model_zoo/hailo_model_zoo/cfg/ /home/jbauwens/Documents/Code/openagri_ai/.venv/lib/python3.10/site-packages/hailo_model_zoo/
```

#### Update the Hailo Model Zoo parameters
The basic Hailo Model Zoo parameters might not be ok if you trained the model yourself. Please check the output layers, for example through https://netron.app/.
The output layers for the newly created model were:
- yolo256/conv39
- yolo256/conv42
- yolo256/conv50
- yolo256/conv53
- yolo256/conv60
- yolo256/conv63

Following files need to be adapted:
```
(.venv) jbauwens@tv115ljan:~/Documents/Code/hailo_model_zoo$ git diff hailo_model_zoo/cfg/base/yolo.yaml
diff --git a/hailo_model_zoo/cfg/base/yolo.yaml b/hailo_model_zoo/cfg/base/yolo.yaml
index f7d458f..32a8f2a 100644
--- a/hailo_model_zoo/cfg/base/yolo.yaml
+++ b/hailo_model_zoo/cfg/base/yolo.yaml
@@ -3,8 +3,8 @@ base:
 preprocessing:
   network_type: detection
   input_shape:
-  - 640
-  - 640
+  - 256
+  - 256
   - 3
   meta_arch: yolo_v5
   padding_color: 114
(.venv) jbauwens@tv115ljan:~/Documents/Code/hailo_model_zoo$ git diff hailo_model_zoo/cfg/networks/yolov8s.yaml
diff --git a/hailo_model_zoo/cfg/networks/yolov8s.yaml b/hailo_model_zoo/cfg/networks/yolov8s.yaml
index e038020..9e619d5 100644
--- a/hailo_model_zoo/cfg/networks/yolov8s.yaml
+++ b/hailo_model_zoo/cfg/networks/yolov8s.yaml
@@ -5,7 +5,7 @@ postprocessing:
     nms: true
   hpp: true
 network:
-  network_name: yolov8s
+  network_name: yolo256
 paths:
   network_path:
   - models_files/ObjectDetection/Detection-COCO/yolo/yolov8s/2023-02-02/yolov8s.onnx
(.venv) jbauwens@tv115ljan:~/Documents/Code/hailo_model_zoo$ git diff hailo_model_zoo/cfg/postprocess_config/yolov8s_nms_config.json
diff --git a/hailo_model_zoo/cfg/postprocess_config/yolov8s_nms_config.json b/hailo_model_zoo/cfg/postprocess_config/yolov8s_nms_config.json
index 3d7747e..e8cbcf9 100644
--- a/hailo_model_zoo/cfg/postprocess_config/yolov8s_nms_config.json
+++ b/hailo_model_zoo/cfg/postprocess_config/yolov8s_nms_config.json
@@ -2,8 +2,8 @@
        "nms_scores_th": 0.2,
        "nms_iou_th": 0.7,
        "image_dims": [
-               640,
-               640
+               256,
+               256
        ],
        "max_proposals_per_class": 100,
        "classes": 80,
@@ -12,21 +12,21 @@
        "background_removal_index": 0,
        "bbox_decoders": [
                {
-                       "name": "bbox_decoder41",
+                       "name": "bbox_decoder39",
                        "stride": 8,
-                       "reg_layer": "conv41",
+                       "reg_layer": "conv39",
                        "cls_layer": "conv42"
                },
                {
-                       "name": "bbox_decoder52",
+                       "name": "bbox_decoder50",
                        "stride": 16,
-                       "reg_layer": "conv52",
+                       "reg_layer": "conv50",
                        "cls_layer": "conv53"
                },
                {
-                       "name": "bbox_decoder62",
+                       "name": "bbox_decoder60",
                        "stride": 32,
-                       "reg_layer": "conv62",
+                       "reg_layer": "conv60",
                        "cls_layer": "conv63"
```


#### Create a HEF file
To use the model on a HAILO AI shield, you need to create a HEF file.
Export your YOLO (yolov8s) model to ONNX format:
```
yolo export model=model.pt imgsz=256 format=onnx opset=11
```

Afterwards you can convert it to a HEF format:
```
hailomz compile --ckpt ~/Documents/Code/openagri_ai/trained_models/model_yolov8_256/model.onnx --calib-path ~/Documents/Code/openagri_ai/trainingset/256by256/test/images/ --yaml ./hailo_model_zoo/cfg/networks/yolov8s.yaml --classes 1 --hw-arch hailo8l
```

