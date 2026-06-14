import os
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from torchvision.transforms import functional as F
from torchvision.ops import box_iou
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchmetrics.detection.mean_ap import MeanAveragePrecision


# =====================================================
# KONFIGURASI HSV + HOG
# =====================================================

DATASET_DIR = "/content/dataset_pascal_voc_hsv_hog"
OUTPUT_DIR = "/content/output_faster_rcnn_hsv_hog"

NUM_EPOCHS = 40
BATCH_SIZE = 4
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# DATASET PASCAL VOC
# =====================================================

def find_image_xml_pairs(split_dir):
    split_dir = Path(split_dir)

    image_files = []
    for ext in [".jpg", ".jpeg", ".png"]:
        image_files.extend(list(split_dir.rglob(f"*{ext}")))

    xml_files = list(split_dir.rglob("*.xml"))
    xml_map = {xml.stem: xml for xml in xml_files}

    pairs = []

    for img_path in image_files:
        xml_path = xml_map.get(img_path.stem)
        if xml_path is not None:
            pairs.append((img_path, xml_path))

    return pairs


def collect_class_names(dataset_dir):
    class_names = set()

    for split in ["train", "valid", "test"]:
        split_dir = Path(dataset_dir) / split

        if not split_dir.exists():
            continue

        for xml_file in split_dir.rglob("*.xml"):
            root = ET.parse(xml_file).getroot()

            for obj in root.findall("object"):
                name = obj.find("name").text.strip()
                class_names.add(name)

    return sorted(list(class_names))


class PascalVOCDataset(Dataset):
    def __init__(self, split_dir, class_to_idx):
        self.split_dir = Path(split_dir)
        self.class_to_idx = class_to_idx
        self.data = find_image_xml_pairs(self.split_dir)

        if len(self.data) == 0:
            raise ValueError(f"Tidak ada pasangan gambar dan XML di {split_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, xml_path = self.data[idx]

        image = Image.open(img_path).convert("RGB")
        image = F.to_tensor(image)

        boxes = []
        labels = []

        root = ET.parse(xml_path).getroot()

        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip()

            if class_name not in self.class_to_idx:
                continue

            label = self.class_to_idx[class_name]
            bndbox = obj.find("bndbox")

            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# =====================================================
# MODEL FASTER R-CNN RESNET-50
# =====================================================

def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    model = fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    return model


# =====================================================
# HITUNG ACCURACY, PRECISION, RECALL, F1
# =====================================================

def calculate_detection_metrics(predictions, targets):
    total_gt = 0
    tp = 0
    fp = 0
    fn = 0

    for pred, target in zip(predictions, targets):
        pred_boxes = pred["boxes"].cpu()
        pred_labels = pred["labels"].cpu()
        pred_scores = pred["scores"].cpu()

        gt_boxes = target["boxes"].cpu()
        gt_labels = target["labels"].cpu()

        keep = pred_scores >= SCORE_THRESHOLD

        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]

        total_gt += len(gt_boxes)

        used_pred = set()

        for gt_idx in range(len(gt_boxes)):
            best_iou = 0
            best_pred_idx = -1

            for pred_idx in range(len(pred_boxes)):
                if pred_idx in used_pred:
                    continue

                if pred_labels[pred_idx] != gt_labels[gt_idx]:
                    continue

                iou = box_iou(
                    gt_boxes[gt_idx].unsqueeze(0),
                    pred_boxes[pred_idx].unsqueeze(0)
                ).item()

                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx

            if best_iou >= IOU_THRESHOLD and best_pred_idx != -1:
                tp += 1
                used_pred.add(best_pred_idx)
            else:
                fn += 1

        fp += len(pred_boxes) - len(used_pred)

    accuracy = tp / total_gt if total_gt > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score


# =====================================================
# TRAINING 1 EPOCH
# =====================================================

def train_one_epoch(model, optimizer, data_loader):
    model.train()

    total_loss = 0

    for images, targets in data_loader:
        images = [img.to(DEVICE) for img in images]

        targets = [
            {
                "boxes": t["boxes"].to(DEVICE),
                "labels": t["labels"].to(DEVICE),
                "image_id": t["image_id"].to(DEVICE)
            }
            for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)


# =====================================================
# EVALUATION
# =====================================================

@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()

    metric_map = MeanAveragePrecision(iou_type="bbox")

    all_predictions = []
    all_targets = []

    for images, targets in data_loader:
        images = [img.to(DEVICE) for img in images]

        targets_device = [
            {
                "boxes": t["boxes"].to(DEVICE),
                "labels": t["labels"].to(DEVICE)
            }
            for t in targets
        ]

        predictions = model(images)

        predictions_cpu = []
        targets_cpu = []

        for pred in predictions:
            predictions_cpu.append({
                "boxes": pred["boxes"].detach().cpu(),
                "scores": pred["scores"].detach().cpu(),
                "labels": pred["labels"].detach().cpu()
            })

        for target in targets_device:
            targets_cpu.append({
                "boxes": target["boxes"].detach().cpu(),
                "labels": target["labels"].detach().cpu()
            })

        metric_map.update(predictions_cpu, targets_cpu)

        all_predictions.extend(predictions_cpu)
        all_targets.extend(targets_cpu)

    map_result = metric_map.compute()

    map_score = float(map_result["map"])
    map50 = float(map_result["map_50"])
    map75 = float(map_result["map_75"])

    accuracy, precision, recall, f1_score = calculate_detection_metrics(
        all_predictions,
        all_targets
    )

    return accuracy, precision, recall, f1_score, map_score, map50, map75


# =====================================================
# MAIN PROGRAM
# =====================================================

def main():
    print("Device:", DEVICE)

    class_names = collect_class_names(DATASET_DIR)

    print("Class names:")
    for idx, name in enumerate(class_names, start=1):
        print(idx, name)

    class_to_idx = {
        class_name: idx + 1
        for idx, class_name in enumerate(class_names)
    }

    num_classes = len(class_names) + 1

    with open(os.path.join(OUTPUT_DIR, "classes.txt"), "w") as f:
        for name in class_names:
            f.write(name + "\n")

    train_dataset = PascalVOCDataset(
        split_dir=os.path.join(DATASET_DIR, "train"),
        class_to_idx=class_to_idx
    )

    valid_dataset = PascalVOCDataset(
        split_dir=os.path.join(DATASET_DIR, "valid"),
        class_to_idx=class_to_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = get_model(num_classes)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    results_path = os.path.join(OUTPUT_DIR, "results.csv")

    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "map",
            "map50",
            "map75"
        ])

    best_map50 = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader)

        accuracy, precision, recall, f1_score, map_score, map50, map75 = evaluate(
            model,
            valid_loader
        )

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"Loss: {train_loss:.6f} | "
            f"Accuracy: {accuracy:.6f} | "
            f"Precision: {precision:.6f} | "
            f"Recall: {recall:.6f} | "
            f"F1: {f1_score:.6f} | "
            f"mAP: {map_score:.6f} | "
            f"mAP50: {map50:.6f} | "
            f"mAP75: {map75:.6f}"
        )

        with open(results_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                accuracy,
                precision,
                recall,
                f1_score,
                map_score,
                map50,
                map75
            ])

        if map50 > best_map50:
            best_map50 = map50

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "best_map50": best_map50
                },
                os.path.join(OUTPUT_DIR, "faster_rcnn_best.pth")
            )

            print("Model terbaik disimpan.")

    torch.save(
        {
            "epoch": NUM_EPOCHS,
            "model_state_dict": model.state_dict(),
            "class_names": class_names
        },
        os.path.join(OUTPUT_DIR, "last.pth")
    )

    print("Training selesai.")
    print("Results:", results_path)
    print("Best model:", os.path.join(OUTPUT_DIR, "faster_rcnn_best.pth"))


if __name__ == "__main__":
    main()