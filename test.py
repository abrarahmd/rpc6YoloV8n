from ultralytics import YOLO
import pandas as pd

model = YOLO('best.pt') 
results = model(
  'datasets/yolo_data/images/test',
  data='data.yaml',
  conf=0.65,
  iou=0.7,
  save=True,
  project="/mnt/d/BRACU_Projects/KIBO_6th/rpc6YoloV8n/runs"
)
# print(results)
