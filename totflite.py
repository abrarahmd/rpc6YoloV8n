from ultralytics import YOLO

model = YOLO("best.pt")

# Export to TFLite
model.export(
  format="tflite",
  nms=True,
  int8=True,
  data="data.yaml",
)
