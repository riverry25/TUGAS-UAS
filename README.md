# Replikasi Jurnal AI: Deteksi Nominal Uang Kertas Rupiah Menggunakan YOLOv5s dan Faster R-CNN ResNet-50

Repository ini berisi dokumentasi, source code, notebook eksperimen, dan ringkasan hasil replikasi jurnal Artificial Intelligence pada bidang **computer vision** dan **object detection**.

Topik proyek ini adalah **deteksi nominal uang kertas Rupiah Indonesia** menggunakan dua model object detection, yaitu **YOLOv5s** dan **Faster R-CNN ResNet-50**. Eksperimen dilakukan dengan dua skema citra, yaitu **RGB** dan **HSV + HOG**.

---

## Artikel Acuan

Artikel utama yang digunakan sebagai acuan replikasi:

**Rupiah Banknotes Detection: Comparison of The Faster R-CNN Algorithm and YOLOv5**

Jurnal: Jurnal INFOTEL
Volume: 16
Nomor: 3
Tahun: 2024
DOI: 10.20895/INFOTEL.V16I3.1189

Artikel tersebut membandingkan performa YOLOv5 dan Faster R-CNN ResNet-50 dalam mendeteksi nominal uang kertas Rupiah menggunakan dua skema preprocessing, yaitu **RGB** dan **HSV with HOG**.

---

## Tujuan Proyek

Tujuan proyek ini adalah mereplikasi alur eksperimen dari artikel utama, mulai dari persiapan dataset, preprocessing, training model, evaluasi hasil, visualisasi prediksi, hingga perbandingan performa model.

Replikasi ini tidak menyalin isi artikel, tetapi mengikuti alur metode eksperimen untuk melihat apakah hasil yang diperoleh dapat mendekati atau bahkan melebihi hasil pada artikel asli.

---

## Model yang Digunakan

Model yang digunakan dalam proyek ini adalah:

1. **YOLOv5s**
   Model one-stage object detection yang ringan, cepat, dan cocok untuk kebutuhan deteksi objek secara real-time.

2. **Faster R-CNN ResNet-50**
   Model two-stage object detection yang memiliki kemampuan kuat dalam lokalisasi dan klasifikasi objek.

---

## Skema Eksperimen

Eksperimen dilakukan dalam empat skenario:

| No | Model                  | Skema Citra |
| -: | ---------------------- | ----------- |
|  1 | YOLOv5s                | RGB         |
|  2 | YOLOv5s                | HSV + HOG   |
|  3 | Faster R-CNN ResNet-50 | RGB         |
|  4 | Faster R-CNN ResNet-50 | HSV + HOG   |

---

## Dataset

Dataset yang digunakan adalah dataset uang kertas Rupiah dari Roboflow. Dataset digunakan dalam dua format berbeda sesuai kebutuhan model:

| Model                  | Format Dataset |
| ---------------------- | -------------- |
| YOLOv5s                | YOLOv5 PyTorch |
| Faster R-CNN ResNet-50 | Pascal VOC     |

Dataset terdiri dari 8 kelas nominal uang kertas Rupiah:

1. Rp1.000
2. Rp2.000
3. Rp5.000
4. Rp10.000
5. Rp20.000
6. Rp50.000
7. Rp75.000
8. Rp100.000

Untuk skema **HSV + HOG**, dataset baru dibuat dari dataset RGB dengan tahapan:

1. Membaca gambar RGB.
2. Mengubah gambar RGB menjadi HSV.
3. Mengambil informasi warna dari ruang warna HSV.
4. Mengekstraksi fitur HOG dari gambar.
5. Menggabungkan fitur HSV dan HOG menjadi citra baru.
6. Menggunakan label asli karena posisi bounding box tidak berubah.

---

## Struktur Repository

```text
rupiah-banknotes-detection-replication/
├── README.md
├── requirements.txt
├── SourceCode/
│   ├── preprocessing_hsv_hog.py
│   ├── train_yolov5_rgb.py
│   ├── train_yolov5_hsv_hog.py
│   ├── train_faster_rcnn_rgb.py
│   ├── train_faster_rcnn_hsv_hog.py
│   ├── hitung_accuracy_yolov5.py
│   ├── predict_yolov5.py
│   ├── predict_faster_rcnn.py
│   └── make_training_graphs.py
│
├── Notebook/
│   ├── YOLOv5_RGB.ipynb
│   ├── YOLOv5_HSV_HOG.ipynb
│   ├── Faster_RCNN_RGB.ipynb
│   └── Faster_RCNN_HSV_HOG.ipynb
│
├── Results/
│   ├── hasil_yolov5_rgb.csv
│   ├── hasil_yolov5_hsv_hog.csv
│   ├── results_faster_rcnn_rgb.csv
│   └── results_faster_rcnn_hsv_hog.csv
│
├── Images/
│   ├── prediksi_faster_rcnn_rgb.png
│   ├── prediksi_yolov5_rgb.png
│   ├── prediksi_faster_rcnn_hsv_hog.png
│   └── prediksi_yolov5_hsv_hog.png
│
└── google_drive_link.txt
```

---

## Penjelasan Source Code

| File                           | Fungsi                                                                   |
| ------------------------------ | ------------------------------------------------------------------------ |
| `preprocessing_hsv_hog.py`     | Mengubah dataset RGB menjadi dataset HSV + HOG                           |
| `train_yolov5_rgb.py`          | Training YOLOv5s menggunakan dataset RGB                                 |
| `train_yolov5_hsv_hog.py`      | Training YOLOv5s menggunakan dataset HSV + HOG                           |
| `train_faster_rcnn_rgb.py`     | Training Faster R-CNN ResNet-50 menggunakan dataset Pascal VOC RGB       |
| `train_faster_rcnn_hsv_hog.py` | Training Faster R-CNN ResNet-50 menggunakan dataset Pascal VOC HSV + HOG |
| `hitung_accuracy_yolov5.py`    | Menghitung detection accuracy YOLOv5 berdasarkan IoU dan class match     |
| `predict_yolov5.py`            | Membuat visualisasi hasil prediksi YOLOv5                                |
| `predict_faster_rcnn.py`       | Membuat visualisasi hasil prediksi Faster R-CNN                          |
| `make_training_graphs.py`      | Membuat grafik training dari file hasil eksperimen                       |

---

## Instalasi Library

Install library yang dibutuhkan:

```bash
pip install torch torchvision torchmetrics pycocotools opencv-python pandas matplotlib scikit-image
```

Untuk YOLOv5, repository resmi YOLOv5 perlu di-clone terlebih dahulu:

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

---

## Cara Menjalankan Eksperimen

### 1. Preprocessing HSV + HOG

```bash
python SourceCode/preprocessing_hsv_hog.py
```

### 2. Training YOLOv5s RGB

```bash
python SourceCode/train_yolov5_rgb.py
```

### 3. Training YOLOv5s HSV + HOG

```bash
python SourceCode/train_yolov5_hsv_hog.py
```

### 4. Training Faster R-CNN ResNet-50 RGB

```bash
python SourceCode/train_faster_rcnn_rgb.py
```

### 5. Training Faster R-CNN ResNet-50 HSV + HOG

```bash
python SourceCode/train_faster_rcnn_hsv_hog.py
```

### 6. Menghitung Detection Accuracy YOLOv5

```bash
python SourceCode/hitung_accuracy_yolov5.py
```

### 7. Membuat Visualisasi Prediksi YOLOv5

```bash
python SourceCode/predict_yolov5.py
```

### 8. Membuat Visualisasi Prediksi Faster R-CNN

```bash
python SourceCode/predict_faster_rcnn.py
```

### 9. Membuat Grafik Training

```bash
python SourceCode/make_training_graphs.py
```

---

## Hasil Eksperimen Akhir

Berikut adalah hasil akhir replikasi berdasarkan empat skenario eksperimen:

| No | Model                  | Skema     | Epoch | Train Loss | Accuracy | Precision | Recall | F1-Score | mAP 0.5 | mAP 0.5:0.95 |
| -: | ---------------------- | --------- | ----: | ---------: | -------: | --------: | -----: | -------: | ------: | -----------: |
|  1 | YOLOv5s                | RGB       |   100 |     0.0299 |   0.9863 |    0.9680 | 0.9740 |   0.9710 |  0.9780 |       0.8930 |
|  2 | YOLOv5s                | HSV + HOG |   100 |     0.0347 |   0.9656 |    0.9500 | 0.9750 |   0.9623 |  0.9790 |       0.8750 |
|  3 | Faster R-CNN ResNet-50 | RGB       |    19 |     0.0343 |   0.8148 |    0.8839 | 0.9124 |   0.8979 |  0.9906 |       0.8602 |
|  4 | Faster R-CNN ResNet-50 | HSV + HOG |    40 |     0.0192 |   0.9691 |    0.9691 | 0.9931 |   0.9810 |  0.9864 |       0.8655 |

---

## Ringkasan Hasil

Berdasarkan hasil eksperimen, **YOLOv5s RGB** memperoleh nilai accuracy tertinggi sebesar **0.9863** dan mAP 0.5:0.95 tertinggi sebesar **0.8930**. Hal ini menunjukkan bahwa YOLOv5s pada skema RGB memiliki performa yang sangat baik dalam evaluasi lokalisasi objek secara ketat.

Sementara itu, **Faster R-CNN ResNet-50 HSV + HOG** memperoleh nilai recall dan F1-score tertinggi, yaitu recall sebesar **0.9931** dan F1-score sebesar **0.9810**. Hasil ini menunjukkan bahwa kombinasi HSV + HOG mampu meningkatkan kemampuan Faster R-CNN dalam mendeteksi objek secara menyeluruh.

Pada beberapa metrik, hasil replikasi Faster R-CNN ResNet-50 HSV + HOG juga menunjukkan peningkatan dibandingkan artikel acuan, khususnya pada accuracy, precision, recall, F1-score, dan mAP 0.5.

---

## Hasil Prediksi Visual

Proyek ini juga menyertakan visualisasi hasil prediksi dari setiap skenario model:

| Model                  | Skema     | Output Visual                      |
| ---------------------- | --------- | ---------------------------------- |
| Faster R-CNN ResNet-50 | RGB       | `prediksi_faster_rcnn_rgb.png`     |
| YOLOv5s                | RGB       | `prediksi_yolov5_rgb.png`          |
| Faster R-CNN ResNet-50 | HSV + HOG | `prediksi_faster_rcnn_hsv_hog.png` |
| YOLOv5s                | HSV + HOG | `prediksi_yolov5_hsv_hog.png`      |

Visualisasi ini digunakan untuk melihat kemampuan model dalam memberikan bounding box dan label nominal uang kertas Rupiah.

---

## Catatan File Besar

Beberapa file seperti dataset, model hasil training, video presentasi, dan dokumen lengkap disimpan di Google Drive karena ukurannya cukup besar.

File besar yang disimpan di Google Drive meliputi:

* Dataset YOLOv5 RGB
* Dataset YOLOv5 HSV + HOG
* Dataset Pascal VOC RGB
* Dataset Pascal VOC HSV + HOG
* Model YOLOv5 `.pt`
* Model Faster R-CNN `.pth`
* Draft jurnal `.docx` dan `.pdf`
* Video presentasi
* Grafik training dan hasil prediksi lengkap

Link Google Drive:

```text
https://drive.google.com/drive/folders/1kVgIqUf6ni5Fx4sSeqdwWsguDW8IXLcu?usp=sharing
```

---

## Bonus Kontribusi

Beberapa pengembangan tambahan yang dilakukan pada replikasi ini:

* Menggunakan dataset uang kertas Rupiah dari Roboflow.
* Membandingkan dua model object detection, yaitu YOLOv5s dan Faster R-CNN ResNet-50.
* Menggunakan dua skema citra, yaitu RGB dan HSV + HOG.
* Menambahkan perhitungan detection accuracy.
* Menambahkan visualisasi hasil prediksi bounding box.
* Menambahkan grafik training dan grafik evaluasi.
* Menyediakan source code dan dokumentasi melalui GitHub.

---

## Author

**Ferdi Ahyana Yusri**
Program Studi Informatika
UIN Sultan Maulana Hasanuddin Banten

---

## License

Repository ini dibuat untuk kebutuhan tugas akademik replikasi jurnal Artificial Intelligence.
