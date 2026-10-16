from ultralytics import YOLO

if __name__ == '__main__':
    # Roboflow dataset ke liye clean YOLOv8 Nano model
    model = YOLO('yolov8s.pt')

    # GPU par fast training
    model.train(
        data='data.yaml',       # Direct root mein rakhi data.yaml file use hogi
        epochs=50,              # Roboflow dataset ke liye 30-50 epochs perfect hain
        imgsz=640,
        batch=16,               # GPU par batch size 16 rakhen (fast chalega)
        workers=2,              # Multi-threading enable karein GPU ke liye
        device=0,               # NVIDIA GPU (device=0)
        name='roboflow_car_detector'
    )