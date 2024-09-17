from ultralytics import YOLO

# model = YOLO("yolov9t.yaml")
# results = model.train(
#     data="trainingset/256by256/yolo.yaml",
#     # imgsz=256,
#     epochs=20,
#     batch=4,
#     plots=False,
#     workers=4,
#     name="yolov9t_256x256_20e",
# )

# model = YOLO("runs/detect/yolov9t_256x256_20e7/weights/last.pt")
# model.train(resume=True)


# Run batched inference on a list of images
model = YOLO("runs/detect/yolov9t_256x256_20e7/weights/best.pt")
model.export(format="ncnn")
