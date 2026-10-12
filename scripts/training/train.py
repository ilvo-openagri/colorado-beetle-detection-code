from ultralytics import YOLO

def main(target="OBJECTS"):
    if target == "OBJECTS":
        model = YOLO("yolov8l.yaml")
        results = model.train(
            data="trainingset/640by640/yolo.yaml",
            # imgsz=640,
            epochs=100,
            batch=6,
            plots=False,
            workers=1,
            name="yolov8l_640x640_100e",
            amp=False,
            half=False,
            # device=[0],
            resume=True,
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

if __name__ == "__main__":
    main()