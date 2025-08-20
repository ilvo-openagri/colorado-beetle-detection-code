

### Prepare your environment for the HAILO AI shield:
Activate the virtual environment, and install the requirements
```bash
cd openagri-ai
python3.10 -m venv .venv-hailo
source .venv-hailo/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install ultralytics
cd ./lib
sudo dpkg -i hailort_4.22.0_amd64.deb
python3 -m pip install hailo_dataflow_compiler-3.32.0-py3-none-linux_x86_64.whl
python3 -m pip install hailort-4.22.0-cp310-cp310-linux_x86_64.whl
cd hailo_model_zoo

# HailoRT 4.20 → MZ version 2.14
# HailoRT 4.21 → MZ version 2.15
# HailoRT 4.22 → MZ version 2.16
git checkout v2.16
python3 -m pip install -e .
```

Clone the hailo model zoo repository
```
https://github.com/hailo-ai/hailo_model_zoo.git
```

Link the git repository into your virtual environment. Dirty, but the code did not seem to work otherwise (file not found exception). Hopefully this will be patched in a future release of the Hailo Model Zoo.
```
ln -sf /home/jbauwens/openagri/openagri_ai/lib/hailo_model_zoo/hailo_model_zoo/cfg/ /home/jbauwens/openagri/openagri_ai/.venv-hailo/lib/python3.12/site-packages/hailo_model_zoo
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
(.venv-hailo) jbauwens@tv115ljan:~/Documents/Code/hailo_model_zoo$ git diff hailo_model_zoo/cfg/base/yolo.yaml
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
(.venv-hailo) jbauwens@tv115ljan:~/Documents/Code/hailo_model_zoo$ git diff hailo_model_zoo/cfg/networks/yolov8s.yaml
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
(.venv-hailo) jbauwens@tv115ljan:~/Documents/Code/hailo_model_zoo$ git diff hailo_model_zoo/cfg/postprocess_config/yolov8s_nms_config.json
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

