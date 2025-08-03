import os
import cv2
import albumentations as A
from tqdm import tqdm
import numpy as np

# 🗂️ Define your paths
IMG_DIR = "Crop_Diseases/images/train"
LBL_DIR = "Crop_Diseases/labels/train"
AUG_IMG_DIR = "Crop_Diseases/images/aug"
AUG_LBL_DIR = "Crop_Diseases/labels/aug"

# ✅ Create output dirs
os.makedirs(AUG_IMG_DIR, exist_ok=True)
os.makedirs(AUG_LBL_DIR, exist_ok=True)

# 🔧 Albumentations pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.Rotate(limit=20, p=0.6),
    A.ColorJitter(p=0.4),
    A.ElasticTransform(p=0.2),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def parse_label_file(label_path):
    keypoints = []
    class_id = None
    with open(label_path, 'r') as file:
        for line in file:
            items = line.strip().split()
            if len(items) < 3: continue
            class_id = int(items[0])
            coords = list(map(float, items[1:]))
            for i in range(0, len(coords), 2):
                keypoints.append((coords[i], coords[i+1]))
    return class_id, keypoints

def save_augmented_label(path, class_id, keypoints):
    with open(path, 'w') as file:
        flat = " ".join([f"{x:.6f}" for kp in keypoints for x in kp])
        file.write(f"{class_id} {flat}")

# 🧠 Loop through all images
for img_file in tqdm(os.listdir(IMG_DIR)):
    if not img_file.endswith(".jpg"): continue

    img_path = os.path.join(IMG_DIR, img_file)
    lbl_path = os.path.join(LBL_DIR, img_file.replace(".jpg", ".txt"))

    image = cv2.imread(img_path)
    height, width = image.shape[:2]

    if not os.path.exists(lbl_path): continue

    class_id, keypoints = parse_label_file(lbl_path)

    # Convert normalized xy to pixel space
    keypoints_px = [(x * width, y * height) for x, y in keypoints]

    for i in range(3):  # 3 augmented copies
        transformed = transform(image=image, keypoints=keypoints_px)
        aug_img = transformed['image']
        aug_keypoints = transformed['keypoints']

        # Convert back to normalized format
        aug_keypoints_norm = [(x / width, y / height) for x, y in aug_keypoints]

        out_img_name = f"{img_file.replace('.jpg', '')}_aug{i}.jpg"
        out_lbl_name = f"{img_file.replace('.jpg', '')}_aug{i}.txt"

        cv2.imwrite(os.path.join(AUG_IMG_DIR, out_img_name), aug_img)
        save_augmented_label(os.path.join(AUG_LBL_DIR, out_lbl_name), class_id, aug_keypoints_norm)
