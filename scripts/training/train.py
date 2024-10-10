from ultralytics import YOLO

model = YOLO("yolov8s.yaml")
results = model.train(
    data="trainingset/640by640/yolo.yaml",
    # imgsz=640,
    epochs=100,
    batch=3,
    plots=False,
    workers=4,
    name="yolov8s_640x640_20e",
    amp=False,
    half=False,
)
