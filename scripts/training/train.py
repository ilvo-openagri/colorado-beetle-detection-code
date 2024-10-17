from ultralytics import YOLO

target = "SEGMENTATION"

if target == "OBJECTS":
    model = YOLO("yolov8n.yaml")
    results = model.train(
        data="trainingset/640by640/yolo.yaml",
        # imgsz=640,
        epochs=100,
        batch=6,
        plots=False,
        workers=4,
        name="yolov8n_640x640_100e",
        amp=False,
        half=False,
    )
elif target == "SEGMENTATION":
    # model = YOLO("yolov8n-seg.yaml")
    model = YOLO("trained_models/model_yolov8n_segmentation/model.pt")
    results = model.train(
        data="trainingset/segmentation/yolo.yaml",
        # imgsz=640,
        epochs=100,
        batch=4,
        plots=False,
        workers=4,
        name="yolov8n_segmentation_100e",
        amp=False,
        half=False,
    )
