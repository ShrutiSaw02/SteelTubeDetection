import os
import shutil

original_ann_dir = 'annotations'
augmented_img_dir = 'images/augmented'
augmented_ann_dir = 'annotations/augmented'

os.makedirs(augmented_ann_dir, exist_ok=True)

aug_images = [f for f in os.listdir(augmented_img_dir) if f.endswith('.jpg')]

for img in aug_images:
    if "_aug" in img:
        base = img.split("_aug")[0] + '.txt'
        src_label_path = os.path.join(original_ann_dir, base)
        dst_label_path = os.path.join(augmented_ann_dir, img.replace(".jpg", ".txt"))

        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dst_label_path)
            print(f"✓ Created: {dst_label_path}")
        else:
            print(f"✗ Missing original label for: {img}")