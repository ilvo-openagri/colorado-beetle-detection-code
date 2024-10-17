from ultralytics.data.annotator import auto_annotate

auto_annotate(
    data="/tmp/images",
    det_model="yolo11n.pt",
    sam_model="mobile_sam.pt",
    device="cuda",
    output_dir="/tmp/labels",
)
