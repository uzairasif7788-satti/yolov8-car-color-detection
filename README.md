# 🚗 YOLOv8 Car Color Detection

A robust deep learning project for detecting and classifying **black** and **white** cars in custom aerial and ground-level images using the state-of-the-art **YOLOv8** object detection framework.

---

## 🎯 Objectives
- **Object Detection:** Accurately locate and bound cars in complex aerial/street-view imagery.
- **Color Classification:** Differentiate and classify vehicles into specific categories: `black_car` and `white_car`.
- **Model Training & Resumption:** Train a custom YOLOv8 model from scratch with robust checkpoint recovery mechanisms (`resume=True`).
- **Performance Evaluation:** Analyze metrics including Box Loss, Classification Loss, DFL Loss, and mAP50.

---

## 🛠️ Technologies Used
- **Python** (Core Scripting)
- **Ultralytics YOLOv8** (Object Detection Framework)
- **PyTorch** (Deep Learning Backend)
- **OpenCV** (Image Processing & Computer Vision)
- **Roboflow** (Dataset Annotation & Management)

---

## 📂 Project Workflow
1. **Dataset Preparation:** Curated, cleaned, and annotated images of vehicles using Roboflow (1091 training images, 418 validation images).
2. **Dataset Configuration:** Configured `data.yaml` with correct relative paths (`train/images`, `valid/images`) and class mappings.
3. **Model Training:** Trained a YOLOv8 Small (`yolov8s.pt`) model for 50 epochs using GPU acceleration (`device=0`, `batch=16`, `workers=4`).
4. **Checkpoint Management:** Maintained automated tracking via `last.pt` to seamlessly resume training after interruptions.
5. **Inference & Testing:** Validated unseen test images using optimized weights (`best.pt`).

---

## 📊 Dataset Specifications
- **Training Images:** 1091
- **Validation Images:** 418
- **Total Instances:** ~8017 labeled instances
- **Classes:**
  - `0`: `black_car`
  - `1`: `white_car`

---

## ⚙️ Model Architecture & Training Details
- **Architecture:** YOLOv8 Small (`yolov8s.pt`)
- **Total Parameters:** ~11.1 million parameters (~22.5 MB weight file size)
- **Loss Functions:** 
  - CIoU Loss (Box regression)
  - BCE Loss (Class classification)
  - Distribution Focal Loss (DFL)
- **Validation Metrics:** Achieved ~80% mAP50 performance across target classes.

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/uzairasif7788-satti/yolov8-car-color-detection.git](https://github.com/uzairasif7788-satti/yolov8-car-color-detection.git)
   cd yolov8-car-color-detection
