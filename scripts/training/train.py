from ultralytics import YOLO

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
