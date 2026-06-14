import os
import cv2
import numpy as np
from pathlib import Path
from skimage.feature import hog


INPUT_DIR = "Dataset_RGB"
OUTPUT_DIR = "Dataset_HSV_HOG"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def convert_hsv_hog(image_path, output_path):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Gagal membaca gambar: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    hog_features, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        block_norm="L2-Hys"
    )

    hog_image = cv2.normalize(
        hog_image,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

    h, s, v = cv2.split(image_hsv)

    hsv_hog = cv2.merge([h, s, hog_image])
    hsv_hog_bgr = cv2.cvtColor(hsv_hog, cv2.COLOR_HSV2BGR)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(str(output_path), hsv_hog_bgr)


def process_dataset(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for image_path in input_dir.rglob("*"):
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            relative_path = image_path.relative_to(input_dir)
            output_path = output_dir / relative_path
            convert_hsv_hog(image_path, output_path)

    print("Preprocessing HSV + HOG selesai.")


if __name__ == "__main__":
    process_dataset(INPUT_DIR, OUTPUT_DIR)