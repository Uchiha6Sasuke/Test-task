# from ultralytics import YOLO

# model = YOLO('yolov8n.pt')

# model.train(
#     data = 'custom.yaml',
#     epochs = 100,
#     imgsz = 640,
#     batch = 16,
#     name = 'custom_model'
# )

import os
import shutil
import random

img_path = 'C:/Users/Alyona/Downloads/custom_conveyor/images'
lbl_path = 'C:/Users/Alyona/Downloads/conveyor_passed/obj_train_data'
output_path = 'dataset'

os.makedirs(f'{output_path}/images/train', exist_ok=True)
os.makedirs(f'{output_path}/images/val', exist_ok=True)
os.makedirs(f'{output_path}/labels/train', exist_ok=True)
os.makedirs(f'{output_path}/labels/val', exist_ok=True)

images = sorted([f for f in os.listdir(img_path) if f.endswith('.PNG')])
labels = sorted([f for f in os.listdir(lbl_path) if f.endswith('.txt')])

data = list(zip(images, labels))
random.shuffle(data)

train_split = 0.8
val_split = 0.2

train_count = int(train_split*len(data))
val_count = int(val_split*len(data))

train_data = data[:train_count]
val_data = data[train_count:]

def move_files(data, subj):
    for img_file, lbl_file in data:
        shutil.move(os.path.join(img_path, img_file), os.path.join(output_path, f'images/{subj}', img_file))
        shutil.move(os.path.join(lbl_path, lbl_file), os.path.join(output_path, f'labels/{subj}', lbl_file))

move_files(train_data, 'train')
move_files(val_data, 'val')