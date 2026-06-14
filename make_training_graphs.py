import pandas as pd
import matplotlib.pyplot as plt
import os


OUTPUT_DIR = "/content/Grafik_Training"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_graph(csv_path, model_name, output_prefix):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    print("Kolom:", df.columns.tolist())

    # Train loss
    if "train_loss" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
        plt.title(f"Train Loss Curve - {model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{output_prefix}_train_loss_curve.png", dpi=300)
        plt.show()

    # mAP curve
    plt.figure(figsize=(8, 5))

    if "map50" in df.columns:
        plt.plot(df["epoch"], df["map50"], marker="o", label="mAP 0.5")

    if "map75" in df.columns:
        plt.plot(df["epoch"], df["map75"], marker="o", label="mAP 0.75")

    if "map" in df.columns:
        plt.plot(df["epoch"], df["map"], marker="o", label="mAP")

    plt.title(f"mAP Curve - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("mAP Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{output_prefix}_map_curve.png", dpi=300)
    plt.show()

    # Evaluation metrics
    plt.figure(figsize=(8, 5))

    if "map50" in df.columns:
        plt.plot(df["epoch"], df["map50"], marker="o", label="mAP 0.5")

    if "map75" in df.columns:
        plt.plot(df["epoch"], df["map75"], marker="o", label="mAP 0.75")

    if "map" in df.columns:
        plt.plot(df["epoch"], df["map"], marker="o", label="mAP")

    plt.title(f"Evaluation Metrics - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{output_prefix}_evaluation_metrics_curve.png", dpi=300)
    plt.show()


make_graph(
    csv_path="/content/Hasil_R-cnn_rgb.csv",
    model_name="Faster R-CNN ResNet-50 RGB",
    output_prefix="faster_rcnn_rgb"
)

make_graph(
    csv_path="/content/Hasil_R-cnn_hsv_hog.csv",
    model_name="Faster R-CNN ResNet-50 HSV + HOG",
    output_prefix="faster_rcnn_hsv_hog"
)