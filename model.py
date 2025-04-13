from ultralytics import YOLO

model = YOLO('yolov8n.yaml') 
# model.train(data='data.yaml', epochs=200, patience=50, imgsz=256, augment=True,
#             hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.2, translate=0.1,
#             scale=0.5, shear=0.1, perspective=0.0005, flipud=0.5, fliplr=0.5, mosaic=0.7)
model.train(data='data.yaml', epochs=10, patience=3, imgsz=256)