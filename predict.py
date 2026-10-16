from ultralytics import YOLO

# Roboflow se trained best model load karein
model = YOLO('runs/detect/roboflow_car_detector/weights/best.pt')

# Test images par inference run karein
results = model.predict(
    source='tested',
    conf=0.70,  # Clear detection confidence
    iou=0.45,   # Duplicate bounding boxes remove karne ke liye
    save=True
)

print("Testing Complete! Results check karne ke liye 'runs/detect/predict' folder kholein.")