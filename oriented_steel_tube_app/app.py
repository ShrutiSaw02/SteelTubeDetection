from flask import Flask, request, jsonify, render_template_string
from ultralytics import YOLO
import numpy as np
import cv2
import os
from PIL import Image
from io import BytesIO

app = Flask(__name__)

model_path = "runs/detect/steel_tube_final_v2/weights/best.pt"
model = YOLO(model_path)

with open("index.html", "r", encoding="utf-8") as f:
    html_ui = f.read()

@app.route("/")
def home():
    return render_template_string(html_ui)

@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files or "roi" not in request.form:
            return jsonify({"error": "Missing image or ROI data."}), 400

        image_file = request.files["image"]
        roi_data = request.form["roi"]

        # Convert image to OpenCV format
        img = Image.open(image_file.stream).convert("RGB")
        img_np = np.array(img)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Parse ROI points
        roi = np.array(eval(roi_data), dtype=np.int32)
        mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [roi], 255)

        # Apply mask
        masked_img = cv2.bitwise_and(img_cv, img_cv, mask=mask)

        # Run YOLO detection
        results = model(masked_img)[0]
        count = 0
        boxes = []

        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            # Check if center is inside ROI
            if cv2.pointPolygonTest(roi, (cx, cy), False) >= 0:
                count += 1
                boxes.append({"x": x1, "y": y1, "w": w, "h": h})

        return jsonify({"count": count, "boxes": boxes})

    except Exception as e:
        print("Detection error:", str(e))
        return jsonify({"error": "Detection failed. Check logs."}), 500

if __name__ == "__main__":
    app.run(debug=True)