import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt


MODEL_PATH = "/content/faster_rcnn_best.pth"
CLASSES_PATH = "/content/classes.txt"
IMAGE_DIR = "/content/images_to_predict"
OUTPUT_DIR = "/content/prediction_faster_rcnn"

CONF_THRES = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_classes(classes_path):
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


classes = load_classes(CLASSES_PATH)

num_classes = len(classes) + 1

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features,
    num_classes
)

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()


for filename in os.listdir(IMAGE_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(IMAGE_DIR, filename)
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)[0]

    image_cv = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    for box, label, score in zip(
        prediction["boxes"],
        prediction["labels"],
        prediction["scores"]
    ):
        if score < CONF_THRES:
            continue

        x1, y1, x2, y2 = box.int().tolist()
        class_name = classes[label.item() - 1]

        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            image_cv,
            f"{class_name} {score:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR))

print("Prediksi Faster R-CNN selesai.")