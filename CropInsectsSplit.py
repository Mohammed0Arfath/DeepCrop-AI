import os, random, shutil
from pathlib import Path

def split_aug_data(base_dir):
    image_dir = Path(f"Crop_Insects/images/aug")
    label_dir = Path(f"Crop_Insects/labels/aug")

    out_img_train = Path(f"Crop_Insects_Aug/images/train")
    out_img_val = Path(f"Crop_Insects_Aug/images/val")
    out_lbl_train = Path(f"Crop_Insects_Aug/labels/train")
    out_lbl_val = Path(f"Crop_Insects_Aug/labels/val")

    out_img_train.mkdir(parents=True, exist_ok=True)
    out_img_val.mkdir(parents=True, exist_ok=True)
    out_lbl_train.mkdir(parents=True, exist_ok=True)
    out_lbl_val.mkdir(parents=True, exist_ok=True)

    image_files = list(image_dir.glob("*.jpg"))
    random.shuffle(image_files)

    val_count = int(len(image_files) * 0.2)
    for idx, img_file in enumerate(image_files):
        lbl_file = label_dir / f"{img_file.stem}.txt"
        if idx < val_count:
            shutil.copy(img_file, out_img_val / img_file.name)
            shutil.copy(lbl_file, out_lbl_val / lbl_file.name)
        else:
            shutil.copy(img_file, out_img_train / img_file.name)
            shutil.copy(lbl_file, out_lbl_train / lbl_file.name)

    print(f"✅ Split complete: {val_count} images in val set")

# Run this:
split_aug_data("C:/Users/moham/Agrithon/Crop_Insects")
