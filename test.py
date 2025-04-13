from ultralytics import YOLO
import pandas as pd

model = YOLO('best.pt') 
results = model('datasets/custom_dataset/images/test', data='data.yaml', conf=0.3, iou=0.6, save=True)