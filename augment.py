import os
import cv2
import albumentations as A
import shutil
from glob import glob

INPUT_FOLDER = 'images'
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, 'augmented')

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=25, p=0.5),
    A.RandomScale(scale_limit=0.1, p=0.3),
    A.GaussNoise(p=0.2),
    A.MotionBlur(p=0.2),
])

image_paths = glob(os.path.join(INPUT_FOLDER, '*.jpg'))

print(f'Found {len(image_paths)} images to augment.')

for img_path in image_paths:
    filename = os.path.basename(img_path)
    image = cv2.imread(img_path)

    for i in range(2):  # create 2 augmented versions per image
        augmented = transform(image=image)['image']
        out_name = f"{os.path.splitext(filename)[0]}_aug{i}.jpg"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        cv2.imwrite(out_path, augmented)
        print(f'Saved: {out_path}')

print("Augmentation complete.")
