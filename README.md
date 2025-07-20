# Steel Tube Detection using YOLOv8

A deep learning-based application for detecting **oriented steel tubes** in industrial environments using **YOLOv8**. The project combines custom dataset preparation, image augmentation, label handling, and a simple web app for real-time predictions.

---

## Features

- Custom YOLO-compatible dataset
- Data augmentation to improve model generalization
- Label fixing utilities for augmented data
- Web app interface using Flask
- Live object detection with bounding boxes
- Trained on YOLOv8n for lightweight performance

---

## Project Structure

```bash
SteelTubeDetection/
│
├── augment.py                   # Augments images & labels
├── fix_labels.py                # Aligns labels after augmentation
├── organize_dataset.py          # Organizes dataset into train/val
├── match_augmented_labels.py    # Matches original & augmented labels
├── dataset.yaml                 # YOLOv8 dataset config
├── dataset/                     # Organized dataset (images & labels)
├── yoloenv/                     # Virtual environment (not tracked)
├── runs/                        # YOLO training results
├── yolov8n.pt                   # Trained YOLOv8 model weights
├── tube_detection_roi_combined.py  # Inference on images
├── oriented_steel_tube_app/
│   ├── app.py                   # Flask app
│   └── index.html               # Simple upload & prediction UI
├── output_prediction.jpg        # Sample detection output
├── .gitignore
└── README.md
