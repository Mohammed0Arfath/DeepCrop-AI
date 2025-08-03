from ultralytics import YOLO

# 🐛 Insects
model_insects = YOLO("yolov8s.pt")
model_insects.train(
    data="C:/Users/moham/Agrithon/Crop_Insects_Aug/crop_insects_aug.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="crop_insects_aug_model",
    project="C:/Users/moham/Agrithon/outputs"
)

# 🍃 Diseases
model_diseases = YOLO("yolov8s-seg.pt")
model_diseases.train(
    data="C:/Users/moham/Agrithon/Crop_Diseases_Aug/crop_diseases_aug.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="crop_diseases_aug_model",
    project="C:/Users/moham/Agrithon/outputs"
)
