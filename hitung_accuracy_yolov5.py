import os
import cv2
import torch
from pathlib import Path


YOLOV5_DIR = "/content/yolov5"

MODEL_PATH = "/content/best.pt"
IMAGE_DIR = "/content/dataset_yolov5_rgb/valid/images"
LABEL_DIR = "/content/dataset_yolov5_rgb/valid/labels"

CONF_THRES = 0.25
IOU_THRES = 0.5


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


def yolo_to_xyxy(label, img_width, img_height):
    cls, x_center, y_center, w, h = label

    x_center *= img_width
    y_center *= img_height
    w *= img_width
    h *= img_height

    xmin = x_center - w / 2
    ymin = y_center - h / 2
    xmax = x_center + w / 2
    ymax = y_center + h / 2

    return int(cls), [xmin, ymin, xmax, ymax]


model = torch.hub.load(YOLOV5_DIR, "custom", path=MODEL_PATH, source="local")
model.conf = CONF_THRES

image_paths = sorted(
    list(Path(IMAGE_DIR).glob("*.jpg")) +
    list(Path(IMAGE_DIR).glob("*.jpeg")) +
    list(Path(IMAGE_DIR).glob("*.png"))
)

correct_detection = 0
total_ground_truth = 0

for image_path in image_paths:
    image = cv2.imread(str(image_path))

    if image is None:
        continue

    h, w = image.shape[:2]
    label_path = Path(LABEL_DIR) / (image_path.stem + ".txt")

    if not label_path.exists():
        continue

    gt_boxes = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        values = list(map(float, line.strip().split()))
        cls, box = yolo_to_xyxy(values, w, h)

        gt_boxes.append({
            "class": cls,
            "box": box
        })

    total_ground_truth += len(gt_boxes)

    results = model(str(image_path))
    predictions = results.xyxy[0].cpu().numpy()

    pred_boxes = []

    for pred in predictions:
        xmin, ymin, xmax, ymax, conf, cls = pred

        pred_boxes.append({
            "class": int(cls),
            "box": [xmin, ymin, xmax, ymax],
            "conf": float(conf),
            "used": False
        })

    for gt in gt_boxes:
        best_iou = 0
        best_pred_idx = -1

        for idx, pred in enumerate(pred_boxes):
            if pred["used"]:
                continue

            if pred["class"] != gt["class"]:
                continue

            iou = calculate_iou(gt["box"], pred["box"])

            if iou > best_iou:
                best_iou = iou
                best_pred_idx = idx

        if best_iou >= IOU_THRES and best_pred_idx != -1:
            correct_detection += 1
            pred_boxes[best_pred_idx]["used"] = True


accuracy = correct_detection / total_ground_truth if total_ground_truth > 0 else 0

print("===== HASIL DETECTION ACCURACY YOLOv5 =====")
print(f"Total Ground Truth : {total_ground_truth}")
print(f"Correct Detection  : {correct_detection}")
print(f"Accuracy           : {accuracy:.6f}")