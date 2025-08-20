# Pipeline basic models
MODEL_64 = "./trained_models/model64/"
MODEL_128 = "./trained_models/model128/"
MODEL_256 = "./trained_models/model256/"

# Pipeline tflite models
MODEL_LITE_64 = "./trained_models/model64/model.tflite"
MODEL_LITE_128 = "./trained_models/model128/model.tflite"
MODEL_LITE_256 = "./trained_models/model256/model.tflite"

# Yolo models (default + tflite)
MODEL_YOLO = "./trained_models/model_yolov8n_640/3d_092025/model.pt"
MODEL_YOLO_TFLITE = "./trained_models/yolo256_saved_model/yolo256_float32.tflite"

# Leave detection YoloV8 model (NCNN + PYTORCH)
MODEL_CLASS_NCNN = "./trained_models/model_yolov8n_segmentation/3d_092025/model_ncnn_model/"
MODEL_CLASS_PYTORCH = "./trained_models/model_yolov8n_segmentation/3d_092025/model.pt"

# YoloV8 models for the Hailo8L AI shield
MODEL_HAILO_256 = "./trained_models/model_yolov8n_256/model.hef"
MODEL_HAILO_640 = "./trained_models/model_yolov8n_640/model.hef"
