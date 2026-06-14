import os

YOLOV5_DIR = "/content/yolov5"

MODEL_PATH = "/content/best.pt"
SOURCE_DIR = "/content/images_to_predict"

PROJECT = "/content/runs/predict"
NAME = "yolov5_prediction"


os.system(f"""
cd {YOLOV5_DIR} && \
python detect.py \
--weights {MODEL_PATH} \
--source {SOURCE_DIR} \
--conf 0.25 \
--save-txt \
--save-conf \
--project {PROJECT} \
--name {NAME}
""")