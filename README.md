# Proyek MLOps: Klasifikasi Performa Siswa  
_Submission Akhir Kelas **Membangun Sistem Machine Learning** – Dicoding_

---

## Deskripsi Proyek
Proyek ini merupakan **submission akhir** dari kelas **Membangun Sistem Machine Learning** pada platform **Dicoding**.  
Fokus utama proyek ini adalah **membangun end-to-end pipeline Machine Learning berbasis praktik MLOps**, mulai dari preprocessing data, training model, experiment tracking, CI/CD, hingga monitoring dan alerting.

Dataset yang digunakan adalah **Students Performance in Exams** yang diperoleh dari **Kaggle**, dengan total **1000 data siswa**.  
Pada proyek ini, dilakukan **klasifikasi performa siswa** berdasarkan nilai ujian dan faktor demografis untuk memahami serta memprediksi capaian akademik siswa.

---

## Tujuan Proyek
Tujuan utama dari proyek ini adalah:
- Membangun sistem Machine Learning yang **terotomatisasi, terukur, dan dapat dimonitor**
- Menerapkan praktik **MLOps end-to-end** sesuai standar industri
- Memastikan model tidak hanya akurat, tetapi juga **mudah direproduksi, dideploy, dan dipantau**
- Menjadi bahan evaluasi dan pembelajaran mengenai **faktor-faktor yang memengaruhi performa akademik siswa**

---

## Dataset
**Nama Dataset**: Students Performance in Exams  
**Sumber**: Kaggle  
**Jumlah Data**: 1000 baris  
**Tipe Masalah**: Klasifikasi

### Fitur dalam Dataset
Dataset memiliki **8 kolom utama**, yaitu:

| Fitur | Deskripsi |
|------|----------|
| gender | Jenis kelamin siswa (male / female) |
| race/ethnicity | Kelompok ras atau etnis siswa |
| parental level of education | Tingkat pendidikan orang tua |
| lunch | Jenis layanan makan siang (standard / reduced) |
| test preparation course | Keikutsertaan kursus persiapan ujian |
| math score | Nilai ujian matematika (0–100) |
| reading score | Nilai ujian membaca (0–100) |
| writing score | Nilai ujian menulis (0–100) |

Dataset ini digunakan untuk menganalisis hubungan antara **latar belakang siswa** dan **hasil ujian**, serta membangun model prediktif performa siswa.

---

## Alur Sistem MLOps
Secara umum, alur implementasi sistem pada proyek ini adalah sebagai berikut:

1. Pengolahan dan preprocessing dataset
2. Otomatisasi preprocessing
3. Training dan eksperimen model Machine Learning
4. Tracking eksperimen dengan MLflow (lokal & online)
5. CI/CD pipeline dengan GitHub Actions
6. Packaging model dan Dockerization
7. Model serving
8. Monitoring, logging, dan alerting

---

## Kriteria Submission & Implementasi

### Kriteria 1: Eksperimen terhadap Dataset Pelatihan
Pada tahap ini dilakukan:

- Preprocessing data:
  - Pembersihan data
  - Encoding fitur kategorikal
  - Normalisasi fitur numerik
- Pembuatan **script preprocessing otomatis**
  - Output berupa data siap latih
- Implementasi **workflow GitHub Actions**
  - Preprocessing berjalan otomatis ketika terjadi trigger tertentu
  - Menjamin konsistensi data untuk training

---

### Kriteria 2: Membangun Model Machine Learning
Tahapan ini berfokus pada **experiment tracking dan reproducibility** menggunakan **MLflow**:

1. **Training Model tanpa Hyperparameter Tuning**
   - Menggunakan Scikit-Learn
   - MLflow Tracking UI disimpan secara **lokal**
   - Menggunakan `mlflow.autolog()`
   - Implementasi terdapat pada file `modelling.py`

2. **Training Model dengan Hyperparameter Tuning**
   - Menggunakan MLflow Tracking UI secara lokal
   - Manual logging untuk metrik:
     - Accuracy
     - Precision
     - Recall
     - F1-score
   - Metrik yang dicatat disamakan dengan autolog

3. **Training Model dengan MLflow Online (DagsHub)**
   - Tracking eksperimen dilakukan secara **online**
   - Menggunakan:
     - Autolog
     - Manual logging
   - Menyimpan minimal **2 artefak tambahan**

---

### Kriteria 3: Workflow CI/CD
Pada tahap ini dibangun pipeline CI/CD yang mencakup:

- Workflow CI untuk:
  - Menjalankan training model (`modelling.py`)
  - Ter-trigger secara otomatis
- Penyimpanan artefak:
  - Model
  - Metadata eksperimen
- Dockerization:
  - Membangun Docker Image menggunakan `mlflow build-docker`
  - Push Docker Image ke **Docker Hub**
- Semua proses dilakukan secara **otomatis melalui GitHub Actions**

---

### Kriteria 4: Sistem Monitoring dan Logging
Tahap ini memastikan model yang sudah dideploy dapat **dipantau performanya**:

- Model Serving:
  - Model dijalankan melalui artefak yang telah dibuat
- Monitoring:
  - **Prometheus** untuk mengumpulkan metrik
  - **Grafana** untuk visualisasi monitoring
- Alerting:
  - Alerting dibuat di Grafana
  - Alert digunakan untuk memantau kondisi kritis (misalnya request error atau latency)

---

## Evaluasi dan Monitoring
Dengan adanya monitoring dan logging, sistem ini mampu:
- Memantau performa model secara real-time
- Mendeteksi potensi penurunan performa (model drift)
- Memberikan notifikasi dini melalui alerting

---

## 📝 Kesimpulan
Proyek ini berhasil mengimplementasikan **pipeline MLOps end-to-end** sesuai standar industri.  
Tidak hanya berfokus pada akurasi model, proyek ini menekankan pada:
- Automasi
- Reproducibility
- Monitoring
- Skalabilitas sistem Machine Learning

Implementasi ini diharapkan dapat menjadi **fondasi kuat** dalam membangun sistem Machine Learning yang siap digunakan di lingkungan produksi.

---
