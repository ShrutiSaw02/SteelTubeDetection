import os
import shutil

original_img_dir = "images"
augmented_img_dir = "images/augmented"
original_lbl_dir = "annotations"
augmented_lbl_dir = "annotations/augmented"
img_out_dir = "dataset/images/train"
lbl_out_dir = "dataset/labels/train"

os.makedirs(img_out_dir, exist_ok=True)
os.makedirs(lbl_out_dir, exist_ok=True)

for file in os.listdir(original_img_dir):
    if file.endswith(".jpg"):
        shutil.move(os.path.join(original_img_dir, file), os.path.join(img_out_dir, file))

for file in os.listdir(augmented_img_dir):
    if file.endswith(".jpg"):
        shutil.move(os.path.join(augmented_img_dir, file), os.path.join(img_out_dir, file))

for file in os.listdir(original_lbl_dir):
    if file.endswith(".txt"):
        shutil.move(os.path.join(original_lbl_dir, file), os.path.join(lbl_out_dir, file))

for file in os.listdir(augmented_lbl_dir):
    if file.endswith(".txt"):
        shutil.move(os.path.join(augmented_lbl_dir, file), os.path.join(lbl_out_dir, file))

print("All images and labels moved into dataset/train folders.")