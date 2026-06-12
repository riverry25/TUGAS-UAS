# Replikasi Jurnal AI: Deteksi Nominal Uang Kertas Rupiah Menggunakan YOLOv5 dan Faster R-CNN ResNet-50

Repository ini berisi dokumentasi, source code, dan ringkasan eksperimen untuk tugas replikasi jurnal Artificial Intelligence pada bidang computer vision dan object detection.

Topik proyek ini adalah deteksi nominal uang kertas rupiah menggunakan dua model object detection, yaitu YOLOv5 dan Faster R-CNN ResNet-50.

## Artikel Acuan

Artikel utama yang digunakan sebagai acuan replikasi:

**Rupiah Banknotes Detection: Comparison of The Faster R-CNN Algorithm and YOLOv5**
Jurnal INFOTEL, Vol. 16 No. 3, 2024.

Artikel tersebut membandingkan performa YOLOv5 dan Faster R-CNN ResNet-50 dalam mendeteksi nominal uang kertas rupiah menggunakan dua skema preprocessing, yaitu RGB dan HSV with HOG.

## Tujuan Proyek

Tujuan proyek ini adalah mereplikasi alur eksperimen dari artikel utama, mulai dari persiapan dataset, preprocessing, training model, evaluasi hasil, hingga perbandingan performa model.

Replikasi ini tidak menyalin isi artikel, tetapi mengikuti metode eksperimen dan alur penelitian yang digunakan.

## Model yang Digunakan

Model yang digunakan dalam proyek ini adalah:

1. **YOLOv5s**
   Digunakan karena ringan, cepat, dan cocok untuk real-time object detection.

2. **Faster R-CNN ResNet-50**
   Digunakan sebagai model pembanding karena termasuk metode two-stage object detection yang kuat untuk deteksi objek.

## Skema Eksperimen

Eksperimen dirancang menjadi empat skenario:

| No | Model                  | Skema     |
| -- | ---------------------- | --------- |
| 1  | YOLOv5s                | RGB       |
| 2  | YOLOv5s                | HSV + HOG |
| 3  | Faster R-CNN ResNet-50 | RGB       |
| 4  | Faster R-CNN ResNet-50 | HSV + HOG |

## Struktur Repository

```text
TUGAS-UAS/
├── README.md
├── src/
│   ├── preprocessing_hsv_hog.py
│   ├── train_faster_rcnn.py
│   ├── evaluate_faster_rcnn.py
│   └── visualize_prediction.py
├── results/
│   └── README.md
├── dataset/
│   └── README.md
├── requirements.txt
└── google_drive_link.txt
```

## Penjelasan Folder

| Folder/File             | Keterangan                                                                          |
| ----------------------- | ----------------------------------------------------------------------------------- |
| `README.md`             | Dokumentasi utama proyek                                                            |
| `src/`                  | Source code utama yang digunakan dalam eksperimen                                   |
| `dataset/README.md`     | Penjelasan dataset dan link dataset                                                 |
| `results/README.md`     | Penjelasan hasil pengujian dan link hasil lengkap                                   |
| `requirements.txt`      | Library Python yang digunakan                                                       |
| `google_drive_link.txt` | Link Google Drive yang berisi file besar seperti dataset, model, artikel, dan video |

## Dataset

Dataset yang digunakan adalah dataset uang kertas rupiah dari Roboflow.

Dataset digunakan dalam dua format:

| Model        | Format Dataset |
| ------------ | -------------- |
| YOLOv5       | YOLOv5 PyTorch |
| Faster R-CNN | Pascal VOC     |

Untuk skema HSV + HOG, dataset baru dibuat dari dataset RGB dengan proses:

1. Membaca gambar RGB.
2. Mengubah gambar ke format HSV.
3. Mengambil channel Hue, Saturation, dan Value.
4. Mengekstraksi fitur HOG dari channel Value.
5. Menggabungkan Hue, Saturation, dan HOG menjadi gambar 3 channel.
6. Menyalin label asli karena posisi bounding box tidak berubah.

## Source Code

Source code utama tersedia pada folder `src/`.

| File                       | Fungsi                                                                       |
| -------------------------- | ---------------------------------------------------------------------------- |
| `preprocessing_hsv_hog.py` | Mengubah dataset RGB menjadi dataset HSV + HOG                               |
| `train_faster_rcnn.py`     | Training Faster R-CNN ResNet-50 menggunakan dataset Pascal VOC               |
| `evaluate_faster_rcnn.py`  | Evaluasi model Faster R-CNN menggunakan mAP, precision, recall, dan F1-score |
| `visualize_prediction.py`  | Membuat visualisasi hasil prediksi bounding box                              |

## Langkah Eksperimen Singkat

### 1. Training YOLOv5 RGB

Training YOLOv5 RGB dilakukan menggunakan repository resmi YOLOv5.

Command training:

```bash
python train.py \
--img 640 \
--batch 16 \
--epochs 100 \
--data /content/dataset/data.yaml \
--weights yolov5s.pt \
--name rupiah_yolov5_rgb
```

### 2. Preprocessing Dataset HSV + HOG

Dataset HSV + HOG dibuat dari dataset RGB menggunakan file:

```bash
python src/preprocessing_hsv_hog.py \
--source /content/dataset \
--output /content/dataset_hsv_hog
```

### 3. Training YOLOv5 HSV + HOG

Training YOLOv5 HSV + HOG dilakukan dengan command:

```bash
python train.py \
--img 640 \
--batch 16 \
--epochs 100 \
--data /content/dataset_hsv_hog/data.yaml \
--weights yolov5s.pt \
--name rupiah_yolov5_hsv_hog
```

Jika training terhenti karena runtime Google Colab habis, training dapat dilanjutkan menggunakan checkpoint `last.pt`:

```bash
python train.py --resume /content/yolov5/runs/train/rupiah_yolov5_hsv_hog/weights/last.pt
```

### 4. Training Faster R-CNN RGB

Training Faster R-CNN ResNet-50 dilakukan menggunakan source code:

```bash
python src/train_faster_rcnn.py \
--data /content/rupiah_voc \
--epochs 40 \
--batch-size 8 \
--output /content/faster_rcnn_rgb
```

### 5. Evaluasi Faster R-CNN

Evaluasi Faster R-CNN dilakukan menggunakan source code:

```bash
python src/evaluate_faster_rcnn.py \
--data /content/rupiah_voc \
--weights /content/faster_rcnn_rgb/faster_rcnn_best.pth \
--output /content/faster_rcnn_rgb/evaluation
```

### 6. Visualisasi Prediksi Faster R-CNN

Visualisasi prediksi bounding box dilakukan menggunakan source code:

```bash
python src/visualize_prediction.py \
--data /content/rupiah_voc \
--weights /content/faster_rcnn_rgb/faster_rcnn_best.pth \
--output /content/faster_rcnn_rgb/predictions \
--threshold 0.5
```

## Hasil Pengujian Sementara

| Model                  | Scheme    | Precision | Recall |  mAP50 | mAP50-95 | Status   |
| ---------------------- | --------- | --------: | -----: | -----: | -------: | -------- |
| YOLOv5s                | RGB       |     0.968 |  0.974 |  0.978 |    0.893 | Selesai  |
| YOLOv5s                | HSV + HOG |         - |      - |      - |        - | Training |
| Faster R-CNN ResNet-50 | RGB       |    0.8839 | 0.9124 | 0.9906 |   0.8602 | Selesai  |
| Faster R-CNN ResNet-50 | HSV + HOG |         - |      - |      - |        - | Belum    |

## Hasil YOLOv5 RGB

Hasil evaluasi YOLOv5 RGB:

| Model   | Scheme | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| ------- | ------ | -----: | --------: | --------: | -----: | ----: | -------: |
| YOLOv5s | RGB    |    432 |       434 |     0.968 |  0.974 | 0.978 |    0.893 |

## Hasil Faster R-CNN RGB

Hasil terbaik Faster R-CNN ResNet-50 RGB diperoleh pada epoch 19.

| Epoch | Train Losses | Accuracy | mAP 0.5 | mAP 0.5:0.95 | Precision | Recall |
| ----: | -----------: | -------: | ------: | -----------: | --------: | -----: |
|    19 |       0.0343 |   0.8148 |  0.9906 |       0.8602 |    0.8839 | 0.9124 |

```

## Author

Ferdi Ahyana Yusri
Program Studi Informatika
UIN Sultan Maulana Hasanuddin Banten
