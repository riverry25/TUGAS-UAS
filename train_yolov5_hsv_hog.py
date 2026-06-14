import os

YOLOV5_DIR = "/content/yolov5"
DATA_YAML = "/content/dataset_yolov5_hsv_hog/data.yaml"

EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16
MODEL = "yolov5s.pt"
PROJECT = "/content/runs/train"
NAME = "yolov5_hsv_hog"


os.system(f"""
cd {YOLOV5_DIR} && \
python train.py \
--img {IMG_SIZE} \
--batch {BATCH_SIZE} \
--epochs {EPOCHS} \
--data {DATA_YAML} \
--weights {MODEL} \
--project {PROJECT} \
--name {NAME}
""")