from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data = 'custom_conveyor.yaml',
    epochs = 100,
    imgsz = 640,
    batch = 16,
    name = 'custom_model'
)