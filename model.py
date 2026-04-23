from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    plots=True,
)
print("Training completed")

model= YOLO("runs/detect/train/weights/best.pt")

metrics=model.val(
    data="dataset.yaml",
    split="test"
)

print("Evaluation is ",metrics)

model.predict(
    source="dataset/images/test",
    save=True
)
print("All images detected successfully")