# Deteksi Nominal Uang Kertas Rupiah Menggunakan YOLOv5 dan Faster R-CNN ResNet-50

Repository ini berisi dokumentasi dan kode eksperimen untuk proyek replikasi jurnal Artificial Intelligence pada bidang computer vision dan object detection.

Topik utama proyek ini adalah deteksi nominal uang kertas rupiah menggunakan dua algoritma object detection, yaitu YOLOv5 dan Faster R-CNN ResNet-50.

## Artikel Acuan

Artikel utama yang digunakan sebagai acuan replikasi adalah:

**Rupiah Banknotes Detection: Comparison of The Faster R-CNN Algorithm and YOLOv5**
Jurnal INFOTEL, Vol. 16 No. 3, 2024.

Artikel tersebut membandingkan performa YOLOv5 dan Faster R-CNN ResNet-50 untuk mendeteksi nominal uang kertas rupiah menggunakan dataset 1120 gambar dari 8 kelas. Eksperimen dilakukan menggunakan dua skema preprocessing, yaitu RGB dan HSV with HOG.

## Tujuan Proyek

Tujuan proyek ini adalah mereplikasi alur eksperimen dari artikel utama, mulai dari pemahaman metode, persiapan dataset, training model, evaluasi hasil, hingga perbandingan hasil replikasi dengan artikel utama.

Replikasi ini tidak menyalin isi artikel, tetapi mengikuti metode eksperimen dan alur penelitian yang digunakan.

## Model yang Digunakan

Model yang digunakan dalam proyek ini adalah:

1. YOLOv5
   Digunakan karena memiliki performa cepat dan cocok untuk real-time object detection.

2. Faster R-CNN ResNet-50
   Digunakan sebagai model pembanding karena termasuk metode two-stage object detection yang kuat untuk deteksi objek.

## Skema Eksperimen

Eksperimen dirancang dalam empat skenario:

| No | Model                  | Skema     |
| -- | ---------------------- | --------- |
| 1  | YOLOv5                 | RGB       |
| 2  | YOLOv5                 | HSV + HOG |
| 3  | Faster R-CNN ResNet-50 | RGB       |
| 4  | Faster R-CNN ResNet-50 | HSV + HOG |

## Dataset

Dataset yang digunakan adalah dataset uang kertas rupiah dari Roboflow. Dataset digunakan dalam dua format:

| Model        | Format Dataset |
| ------------ | -------------- |
| YOLOv5       | YOLOv5 PyTorch |
| Faster R-CNN | Pascal VOC     |

Untuk skema HSV + HOG, dataset dibuat dari gambar RGB asli dengan proses preprocessing:

1. Membaca gambar RGB.
2. Mengubah gambar ke format HSV.
3. Mengambil fitur HOG dari channel Value.
4. Menggabungkan Hue, Saturation, dan HOG.
5. Menyimpan hasil preprocessing sebagai dataset baru.

Label bounding box tetap sama karena posisi objek uang tidak berubah.

## Struktur Dataset YOLOv5

```text
dataset/
├── train/images
├── train/labels
├── valid/images
├── valid/labels
├── test/images
├── test/labels
└── data.yaml
```

## Struktur Dataset Faster R-CNN

```text
dataset/
├── images/
└── annotations/
```

Dataset Faster R-CNN menggunakan format anotasi Pascal VOC dengan file `.xml`.

## Hasil Eksperimen Sementara

| Model                  | Scheme    | Precision | Recall |  mAP50 | mAP50-95 | Status   |
| ---------------------- | --------- | --------: | -----: | -----: | -------: | -------- |
| YOLOv5s                | RGB       |     0.968 |  0.974 |  0.978 |    0.893 | Selesai  |
| YOLOv5s                | HSV + HOG |         - |      - |      - |        - | Training |
| Faster R-CNN ResNet-50 | RGB       |    0.8839 | 0.9124 | 0.9906 |   0.8602 | Selesai  |
| Faster R-CNN ResNet-50 | HSV + HOG |         - |      - |      - |        - | Belum    |

## Hasil YOLOv5 RGB

Training YOLOv5 RGB dilakukan dengan command:

```bash
python train.py \
--img 640 \
--batch 16 \
--epochs 100 \
--data /content/dataset/data.yaml \
--weights yolov5s.pt \
--name rupiah_yolov5_rgb
```

Hasil evaluasi YOLOv5 RGB:

| Model   | Scheme | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| ------- | ------ | -----: | --------: | --------: | -----: | ----: | -------: |
| YOLOv5s | RGB    |    432 |       434 |     0.968 |  0.974 | 0.978 |    0.893 |

## Hasil Faster R-CNN RGB

Training Faster R-CNN ResNet-50 RGB telah selesai. Model terbaik diperoleh pada epoch 19 berdasarkan nilai mAP 0.5:0.95 tertinggi.

| Epoch | Train Losses | Accuracy | mAP 0.5 | mAP 0.5:0.95 | Precision | Recall |
| ----: | -----------: | -------: | ------: | -----------: | --------: | -----: |
|    19 |       0.0343 |   0.8148 |  0.9906 |       0.8602 |    0.8839 | 0.9124 |

File model terbaik:

```text
faster_rcnn_best.pth
```

## Training YOLOv5 HSV + HOG

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

Jika training berhenti karena runtime Google Colab habis, training dapat dilanjutkan menggunakan checkpoint `last.pt`:

```bash
python train.py --resume /content/yolov5/runs/train/rupiah_yolov5_hsv_hog/weights/last.pt
```

## Evaluasi

Metrik evaluasi yang digunakan dalam proyek ini adalah:

* Precision
* Recall
* mAP50
* mAP50-95
* Accuracy
* Train Losses
* Visualisasi bounding box prediksi

## Status Proyek

| Tahap                              | Status          |
| ---------------------------------- | --------------- |
| Pemilihan artikel utama            | Selesai         |
| Pemahaman metode                   | Selesai         |
| Persiapan dataset YOLOv5 RGB       | Selesai         |
| Training YOLOv5 RGB                | Selesai         |
| Evaluasi YOLOv5 RGB                | Selesai         |
| Persiapan dataset Faster R-CNN RGB | Selesai         |
| Training Faster R-CNN RGB          | Selesai         |
| Evaluasi Faster R-CNN RGB          | Selesai         |
| Dataset YOLOv5 HSV + HOG           | Selesai         |
| Training YOLOv5 HSV + HOG          | Sedang berjalan |
| Evaluasi YOLOv5 HSV + HOG          | Belum           |
| Training Faster R-CNN HSV + HOG    | Belum           |
| Draft artikel replikasi            | Belum final     |

## Catatan

Repository ini dibuat untuk dokumentasi tugas replikasi jurnal Artificial Intelligence. Hasil eksperimen dapat berbeda dari artikel utama karena perbedaan dataset, jumlah kelas, kualitas anotasi, pembagian data, preprocessing, konfigurasi training, dan perangkat komputasi yang digunakan.

## Author

Ferdi Ahyana Yusri
Program Studi Informatika
UIN Sultan Maulana Hasanuddin Banten
