from ultralytics import YOLO

model = YOLO('yolov8n.yaml') 
# model.train(data='data.yaml', epochs=200, patience=50, imgsz=256, augment=True,
#             hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.2, translate=0.1,
#             scale=0.5, shear=0.1, perspective=0.0005, flipud=0.5, fliplr=0.5, mosaic=0.7)
model.train(
  data='data.yaml',
  epochs=50,
  patience=5,
  imgsz=256,
  pretrained=False,
  project="/mnt/d/BRACU_Projects/KIBO_6th/rpc6YoloV8n/runs",
  overlap_mask=False,
  batch=32,
  mosaic=0.0,
  hsv_h=0.0,
  hsv_s=0.0,
  hsv_v=0.0,
  translate=0.0,
  scale=0.0,
  fliplr=0.0,
  erasing=0.0,
)
