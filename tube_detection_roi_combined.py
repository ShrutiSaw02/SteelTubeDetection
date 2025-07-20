import cv2
import matplotlib
matplotlib.use("TkAgg")  
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector, RectangleSelector
from ultralytics import YOLO
import numpy as np
import os

image_path = "dataset/images/val/IMG-20250618-WA0092.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

roi_type = input("Enter ROI type (rect/poly): ").strip().lower()
roi_coords = []

done = False  

def handle_rectangle(eclick, erelease):
    global roi_coords, done
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    roi_coords = [(x1, y1), (x2, y2)]
    done = True
    plt.close()

def handle_polygon(verts):
    global roi_coords, done
    if len(verts) < 3:
        print("Polygon must have at least 3 points.")
        return
    roi_coords = [(int(x), int(y)) for x, y in verts]
    done = True
    plt.close()

fig, ax = plt.subplots()
ax.imshow(image_rgb)
plt.title("Draw ROI and close the window when done")

if roi_type == "rect":
    rect_selector = RectangleSelector(ax, handle_rectangle, useblit=True, button=[1])
elif roi_type == "poly":
    poly_selector = PolygonSelector(ax, handle_polygon, useblit=True)
else:
    print("Invalid ROI type. Use 'rect' or 'poly'.")
    exit()

plt.show()

if not roi_coords:
    print("No ROI selected. Exiting.")
    exit()

if roi_type == "rect":
    (x1, y1), (x2, y2) = roi_coords
    cropped_img = image[y1:y2, x1:x2]
    roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    roi_mask[y1:y2, x1:x2] = 255
else:
    roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.array(roi_coords, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(roi_mask, [pts], 255)
    cropped_img = cv2.bitwise_and(image, image, mask=roi_mask)

model = YOLO("runs/detect/steel_tube_final_v2/weights/best.pt")
results = model(cropped_img)[0]

boxes = results.boxes.xyxy.cpu().numpy()
count = 0

for box in boxes:
    x1, y1, x2, y2 = map(int, box)
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

    if roi_type == "poly":
        if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
            count += 1
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        count += 1
        cv2.rectangle(image, (x1 + roi_coords[0][0], y1 + roi_coords[0][1]),
                      (x2 + roi_coords[0][0], y2 + roi_coords[0][1]), (0, 255, 0), 2)

cv2.putText(image, f"Tubes Detected: {count}", (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

output_path = "output_prediction.jpg"
cv2.imwrite(output_path, image)
print(f" Output saved to {output_path}")
print(f" Total tubes detected: {count}")