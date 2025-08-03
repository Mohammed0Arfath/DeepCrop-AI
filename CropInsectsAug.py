import os
import cv2
import albumentations as A
from tqdm import tqdm
import numpy as np

# 🗂️ Directories
IMG_DIR = "Crop_Insects/images/train"
LBL_DIR = "Crop_Insects/labels/train"
AUG_IMG_DIR = "Crop_Insects/images/aug"
AUG_LBL_DIR = "Crop_Insects/labels/aug"

# 🔧 Create output dirs
os.makedirs(AUG_IMG_DIR, exist_ok=True)
os.makedirs(AUG_LBL_DIR, exist_ok=True)

# 🌀 Augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.Rotate(limit=15, p=0.6),
    A.ColorJitter(p=0.4),
    A.Blur(p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def read_label_file(label_path):
    bboxes, class_labels = [], []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = parts
            bboxes.append([float(x), float(y), float(w), float(h)])
            class_labels.append(int(cls))
    return bboxes, class_labels

def save_label_file(path, class_ids, boxes):
    with open(path, 'w') as f:
        for cls, box in zip(class_ids, boxes):
            box_str = " ".join([f"{x:.6f}" for x in box])
            f.write(f"{cls} {box_str}\n")

# 🚀 Process files
for img_file in tqdm(os.listdir(IMG_DIR)):
    if not img_file.endswith(".jpg"):
        continue

    img_path = os.path.join(IMG_DIR, img_file)
    lbl_path = os.path.join(LBL_DIR, img_file.replace(".jpg", ".txt"))

    if not os.path.exists(lbl_path):
        continue

    image = cv2.imread(img_path)
    bboxes, class_labels = read_label_file(lbl_path)

    for i in range(3):  # Generate 3 augmented versions
        try:
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        except Exception as e:
            print(f"Augment error on {img_file}: {e}")
            continue

        aug_img = transformed["image"]
        aug_boxes = transformed["bboxes"]
        aug_labels = transformed["class_labels"]

        # Skip empty output (no valid boxes)
        if len(aug_boxes) == 0:
            continue

        aug_img_name = f"{img_file.replace('.jpg', '')}_aug{i}.jpg"
        aug_lbl_name = f"{img_file.replace('.jpg', '')}_aug{i}.txt"

        cv2.imwrite(os.path.join(AUG_IMG_DIR, aug_img_name), aug_img)
        save_label_file(os.path.join(AUG_LBL_DIR, aug_lbl_name), aug_labels, aug_boxes)
