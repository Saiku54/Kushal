# 1. Install YOLOv5 and dependencies (Uncomment to run)
git clone https://github.com/ultralytics/yolov5
pip install -r yolov5/requirements.txt

import torch
from yolov5 import YOLOv5

# 2. Set up paths
# Path to YOLOv5 directory (adjust if necessary)
YOLOv5_PATH = "./yolov5"
DATASET_PATH = "./ship_dataset"   # Path to your dataset
MODEL_SAVE_PATH = "./yolov5/runs/train/exp"  # Save path for the trained model

# 3. Load the YOLOv5 model
model = YOLOv5(YOLOv5_PATH)

# 4. Configure the dataset
# Create a YAML configuration file to specify the dataset paths
# Example (save this as ship.yaml in the `yolov5/data` folder):
# train: ./ship_dataset/train/images
# val: ./ship_dataset/val/images
# nc: 3  # number of classes
# names: ['cargo_ship', 'passenger_ship', 'fishing_boat']  # example class names

DATA_CONFIG_PATH = "yolov5/data/ship.yaml"  # Path to your data config

# 5. Train the model
model.train(data=DATA_CONFIG_PATH, epochs=100, batch_size=16, imgsz=640, device="0")

# 6. Run inference on new images
TEST_IMAGE_PATH = "./test_images/test1.jpg"  # Path to an image to test

results = model(TEST_IMAGE_PATH)
results.show()  # Displays the image with detected ships
results.save()  # Saves the output with detections

# 7. Access results
# Get the list of detections
detections = results.pandas().xyxy[0]  # Pandas dataframe of detections
print(detections)
