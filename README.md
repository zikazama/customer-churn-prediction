# Prediksi Customer Churn dengan TFX Pipeline di Cloud dan Monitoring Prometheus

Nama: **Fauzi Fadhlurrohman**

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Telco Customer Churn (IBM)](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) |
| Masalah | Pelanggan yang berhenti berlangganan (churn) mengurangi pendapatan perusahaan. Perusahaan perlu mengetahui pelanggan mana yang berisiko churn agar tim retensi dapat bertindak lebih awal dan tepat sasaran. |
| Solusi machine learning | Model klasifikasi biner yang memprediksi probabilitas seorang pelanggan akan churn. Model dilatih dan dievaluasi melalui pipeline TFX end-to-end; hanya model yang mendapat *blessing* dari Evaluator yang dipublikasikan ke folder serving. Target: AUC di atas 0,80 dan binary accuracy di atas 0,75 pada data eval. |
| Metode pengolahan | `customerID` dihapus; `TotalCharges` dikonversi menjadi numerik dan nilai kosong diisi 0; data dibagi menjadi train dan eval oleh CsvExampleGen. Fitur numerik distandardisasi dengan z-score, fitur kategorikal dipetakan ke vocabulary, dan label `Churn` diubah menjadi biner (Yes = 1, No = 0). |
| Arsitektur model | Model Keras dua cabang: cabang numerik (Dense, Dropout, Dense) dan cabang kategorikal (Embedding 8 dimensi per fitur, Dense), digabung lalu Dense, Dropout, dan Dense 1 sigmoid. Ukuran lapisan, dropout rate, dan learning rate dipilih oleh Keras Tuner: units **64**, dropout rate **0,2**, learning rate **0,001**. |
| Metrik evaluasi | AUC, Binary Accuracy, Precision, Recall, dan Example Count dari TensorFlow Model Analysis. Ambang blessing: Binary Accuracy minimal 0,70. |
| Performa model | Evaluator pada 2.337 contoh eval: AUC **0,8512**, Binary Accuracy **0,8023**, Precision **0,6455**, Recall **0,5599**. Model mendapat *blessing* dan dipublikasikan oleh Pusher. |
| Deployment | Online serving lewat REST API: TensorFlow Serving + gateway Flask pada **Google Cloud Compute Engine** (VM e2-micro, systemd). |
| Web app | http://34.122.18.16:8080/ |
| Monitoring | **Prometheus** (`monitoring/`) men-scrape TensorFlow Serving (5 detik), gateway API, dan dirinya sendiri; tercatat 364 request `/predict` sukses, 31 ditolak 422 dan 17 ditolak 400, 1.552 prediksi (churn 366, tidak churn 1.186), latensi p50 6,4 ms dan p95 123,1 ms, ketiga target UP (ketersediaan 100,0%), dan model berstatus `AVAILABLE` (versi 1793092663). Detail pada bagian 7. |

> **Catatan:** layanan cloud yang disebut di dokumen ini (web app, API, dan Prometheus pada VM Google Cloud) sudah dimatikan setelah proyek selesai, sehingga tautan `34.122.18.16:8080` tidak lagi aktif. Seluruh kode, konfigurasi, dan tangkapan layar tetap tersedia di repositori ini; untuk menjalankannya kembali ikuti `deployment/setup.sh`.

## 1. Ringkasan Proyek

Proyek ini membangun machine learning pipeline dengan **TensorFlow Extended (TFX)** untuk memprediksi customer churn. Seluruh komponen dijalankan oleh **Apache Beam sebagai pipeline orchestrator** (`BeamDagRunner`, `from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner`) pada notebook `fauzi19-pipeline.ipynb`, dan artefaknya disimpan di folder **fauzi19-pipeline**.

Komponen pipeline: `CsvExampleGen`, `StatisticsGen`, `SchemaGen`, `ExampleValidator`, `Transform`, `Tuner`, `Trainer`, `Resolver`, `Evaluator`, dan `Pusher`. Komponen didefinisikan pada sel notebook, dirangkai menjadi satu objek `Pipeline` (metadata SQLite di dalam pipeline root), lalu dijalankan sekaligus dengan `BeamDagRunner().run(...)`. Statistik, schema, anomali, hasil Transform, hyperparameter Tuner, dan metrik Evaluator ditampilkan pada bagian *Hasil eksekusi komponen* notebook.

Sistem kemudian dijalankan di **cloud** (Google Cloud Compute Engine) sebagai REST API dan dipantau dengan **Prometheus**. Saran peningkatan yang diterapkan:

1. **Tuner** untuk hyperparameter tuning otomatis (Keras Tuner `RandomSearch`, 6 percobaan).
2. **Deployment dengan TensorFlow Serving** di VM cloud, lengkap dengan gateway, web app, dan `Dockerfile`.
3. **Notebook pengujian** `fauzi19-testing.ipynb` yang mengirim prediction request ke endpoint cloud.

## 2. Dataset

- Sumber: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv (salinan bersih: `data/telco_churn.csv`).
- Jumlah baris: **7.043**; kolom awal 21, menjadi **20** setelah `customerID` dihapus (19 fitur dan 1 label).
- Label: **Churn** (Yes/No); distribusi No = 73,46% dan Yes = 26,54%.
- Nilai kosong: 0 setelah `TotalCharges` dibersihkan.

Fitur numerik: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`.

Fitur kategorikal: `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`.

## 3. Masalah dan Tujuan

Churn berdampak langsung pada pendapatan dan biaya akuisisi pelanggan baru. Jika pelanggan berisiko churn dapat dikenali lebih awal, tim retensi dapat menawarkan insentif atau layanan yang tepat. Kelas positif adalah **Churn = Yes**; model menghasilkan probabilitas dan pada threshold 0,5 probabilitas di atasnya diprediksi sebagai churn.

## 4. Arsitektur Pipeline TFX

| Komponen | Fungsi | Artefak |
| ----------- | ----------- | ----------- |
| CsvExampleGen | Membaca CSV bersih, mengubahnya menjadi TFRecord, dan membagi data menjadi split train dan eval | `fauzi19-pipeline/CsvExampleGen/examples/2` |
| StatisticsGen | Menghitung statistik deskriptif setiap fitur | `fauzi19-pipeline/StatisticsGen/statistics/3` |
| SchemaGen | Membuat schema (tipe data, domain nilai, presensi) dari statistik | `fauzi19-pipeline/SchemaGen/schema/4` |
| ExampleValidator | Mendeteksi anomali dengan membandingkan statistik dan schema | `fauzi19-pipeline/ExampleValidator/anomalies/5` |
| Transform | Menjalankan `preprocessing_fn` pada `fauzi19_transform.py` | `fauzi19-pipeline/Transform/transformed_examples/6`, `fauzi19-pipeline/Transform/transform_graph/6` |
| Tuner | Mencari hyperparameter terbaik dengan Keras Tuner (`fauzi19_tuner.py`) | `fauzi19-pipeline/Tuner/best_hyperparameters/7` |
| Trainer | Melatih model dengan hyperparameter terbaik (`fauzi19_trainer.py`) | `fauzi19-pipeline/Trainer/model/8` |
| Resolver | Mencari model *blessed* terbaru sebagai baseline | (kosong pada eksekusi pertama) |
| Evaluator | Mengevaluasi model dengan TFMA dan memberi *blessing* | `fauzi19-pipeline/Evaluator/evaluation/9`, `fauzi19-pipeline/Evaluator/blessing/9` |
| Pusher | Menyalin model *blessed* ke folder serving | `fauzi19-pipeline/Pusher/pushed_model/10`, `serving_model/1793092663` |

### Transform

Fitur numerik distandardisasi dengan `tft.scale_to_z_score`; fitur kategorikal dipetakan ke indeks dengan `tft.compute_and_apply_vocabulary` (satu *out-of-vocabulary bucket*); label `Churn` menjadi `Churn_xf`. Transform graph dipakai ulang oleh Evaluator dan oleh signature serving sehingga preprocessing konsisten antara training dan inference.

### Tuner

Ruang pencarian: units (32, 64, 128), dropout rate (0,1; 0,2; 0,3), dan learning rate (0,01; 0,001; 0,0005). Objective: `val_auc` (maksimum). Enam percobaan menghasilkan `val_auc` terbaik **0,8510** dengan units 64, dropout rate 0,2, dan learning rate 0,001.

### Trainer

Model Keras menerima 4 input numerik (sudah distandardisasi) dan 15 input kategorikal (indeks vocabulary). Optimizer Adam, loss *binary crossentropy*, 500 langkah per epoch, evaluasi 100 langkah, batch 64, maksimum 3 epoch dengan *early stopping*. Model diekspor dengan `keras.export.ExportArchive` sebagai SavedModel yang kompatibel dengan TensorFlow Serving, dengan signature `serving_default` (menerima `tf.Example` mentah lalu menerapkan Transform) dan `transform_features` (dipakai Evaluator).

### Evaluator dan Pusher

Evaluator memakai signature `serving_default`, label `Churn_xf`, dan fungsi preprocessing `transform_features`. Ambang blessing: Binary Accuracy minimal 0,70. Pusher hanya menyalin model yang mendapat blessing.

## 5. Hasil Evaluasi

Evaluasi pada split eval (2.337 contoh):

| Metrik | Nilai |
| ----------- | ----------- |
| AUC | 0,8512 |
| Binary Accuracy | 0,8023 |
| Precision | 0,6455 |
| Recall | 0,5599 |
| Example Count | 2.337 |

Trainer berjalan 3 epoch. `val_auc` tertinggi dicapai pada epoch 1; *early stopping* dengan `restore_best_weights` memulihkan bobot epoch tersebut, sehingga model yang dipublikasikan sesuai dengan nilai berikut:

| Metrik | Train | Validasi |
| ----------- | ----------- | ----------- |
| AUC | 0,8285 | 0,8515 |
| Binary Accuracy | 0,7963 | 0,8027 |
| Precision | 0,6503 | 0,6428 |
| Recall | 0,5049 | 0,5638 |
| Loss | 0,4349 | 0,4121 |

Model memiliki AUC sekitar 0,85 dan accuracy sekitar 80%, jauh lebih baik daripada classifier acak. Recall sekitar 56% menunjukkan sebagian pelanggan churn belum terdeteksi. Untuk kebutuhan retensi, threshold probabilitas dapat diturunkan agar recall naik, dengan konsekuensi precision turun; pemilihan threshold sebaiknya mempertimbangkan biaya bisnis false positive dan false negative.

## 6. Deployment ke Cloud

### Opsi deployment dan platform

| Opsi | Pertimbangan |
| ----------- | ----------- |
| Batch prediction | Cocok bila churn dihitung berkala (mingguan), tetapi tim retensi tidak dapat memeriksa pelanggan secara langsung. |
| **Online serving lewat REST API (dipilih)** | Prediksi real-time per pelanggan dan mudah diintegrasikan ke aplikasi CRM. |
| Platform | **Google Cloud Platform, Compute Engine (VM e2-micro, region us-central1-a)**. VM berada pada *free tier* sehingga tanpa biaya, dan dikelola dengan `systemd` agar layanan hidup kembali otomatis. |

### Arsitektur

```text
Klien --HTTP :8080--> Gateway (Flask + gunicorn) --REST :8501--> TensorFlow Serving --> model hasil Pusher
                           |  /metrics                              |  /monitoring/prometheus/metrics
                           +----------------> Prometheus (:9090, scrape 15 detik) <-----+
```

- **TensorFlow Serving** (`tensorflow_model_server` 2.21.0 dari paket apt) memuat `serving_model/1793092663` hasil komponen Pusher (`deployment/fauzi19-tfserving.service`, memakai `monitoring/prometheus.config`). Image Docker setara tersedia pada `Dockerfile`.
- **Gateway** (`deployment/gateway.py`) menerima fitur pelanggan dalam JSON, memvalidasi nilai, membentuk `tf.Example`, memanggil TensorFlow Serving, lalu mengembalikan probabilitas dan label. Gateway juga menyediakan halaman web untuk mencoba prediksi dan mengekspor metrik Prometheus.
- **Prometheus** 3.5.0 memantau gateway dan TensorFlow Serving (lihat bagian Monitoring).
- Firewall hanya membuka port 8080. Endpoint metrik TensorFlow Serving diteruskan gateway pada `http://34.122.18.16:8080/monitoring/prometheus/metrics`, dan Prometheus mengambilnya dari URL cloud tersebut (bukan localhost). Port 9090 Prometheus dan 8501 TensorFlow Serving hanya dapat diakses dari VM.
- Instalasi: `deployment/setup.sh` memasang TensorFlow Serving, virtualenv gateway, Prometheus, dan mendaftarkan ketiga layanan `systemd`.

### Tautan web app

- **Web app**: http://34.122.18.16:8080/ (formulir prediksi churn)
- **Endpoint API**: `POST http://34.122.18.16:8080/predict`
- Kesehatan layanan: http://34.122.18.16:8080/health
- Metrik Prometheus gateway: http://34.122.18.16:8080/metrics
- Metrik Prometheus TensorFlow Serving (endpoint yang di-scrape Prometheus): http://34.122.18.16:8080/monitoring/prometheus/metrics
- Skema fitur: http://34.122.18.16:8080/schema
- **Status model TensorFlow Serving (cloud)**: http://34.122.18.16:8080/v1/models/fauzi19-churn-model
- **Metadata model TensorFlow Serving (cloud)**: http://34.122.18.16:8080/v1/models/fauzi19-churn-model/metadata

Bukti serving berjalan di cloud: `fauzi19-deployment.png` adalah tangkapan layar browser (address bar terlihat) yang mengakses `http://34.122.18.16:8080/v1/models/fauzi19-churn-model/metadata` pada VM Google Cloud.

Contoh request:

```bash
curl -X POST http://34.122.18.16:8080/predict -H "Content-Type: application/json" -d '{"gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":2,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":80.35,"TotalCharges":160.7}'
# {"label":"churn","probability":0.736}
```

Mengirim *list* JSON berisi beberapa pelanggan akan mengembalikan `{"predictions": [...]}`.

### Pengujian

`fauzi19-testing.ipynb` menguji endpoint cloud di atas: satu pelanggan berisiko (probabilitas churn 0,736) dan satu pelanggan aman (0,0121), batch 200 pelanggan (accuracy 0,7900, precision 0,6596, recall 0,5439; sejalan dengan metrik Evaluator pipeline), serta validasi input (422).

## 7. Monitoring dengan Prometheus

Prometheus berjalan di VM yang sama dan dikonfigurasi lewat direktori `monitoring/` (`Dockerfile`, `prometheus.yml`, dan `prometheus.config`; port 9090 hanya untuk localhost). Label global `monitor="tf-serving-monitor"`; TensorFlow Serving di-*scrape* setiap 5 detik, target lain setiap 15 detik:

| Job | Target (interval) | Isi |
| ----------- | ----------- | ----------- |
| `churn-gateway` | `34.122.18.16:8080/metrics` | Metrik aplikasi gateway: jumlah dan latensi request, prediksi per kelas, sebaran probabilitas, error, status model |
| `tf-serving` | `34.122.18.16:8080/monitoring/prometheus/metrics` (5 detik) | Metrik bawaan TensorFlow Serving yang diambil dari endpoint **cloud** (`:tensorflow:serving:request_count`, latensi, status load model) |
| `prometheus` | `127.0.0.1:9090` | Metrik Prometheus sendiri |

Metrik kustom pada gateway:

| Metrik | Jenis | Kegunaan |
| ----------- | ----------- | ----------- |
| `churn_http_requests_total{endpoint,method,status}` | Counter | Laju request dan kode status |
| `churn_request_latency_seconds` | Histogram | Latensi request, dipakai untuk p50/p95 |
| `churn_tfserving_latency_seconds` | Histogram | Latensi pemanggilan TensorFlow Serving |
| `churn_predictions_total{label}` | Counter | Jumlah prediksi churn dan tidak churn |
| `churn_prediction_probability` | Histogram | Sebaran probabilitas churn (untuk mendeteksi pergeseran) |
| `churn_errors_total{type}` | Counter | Error validasi, bad request, dan kegagalan TensorFlow Serving |
| `churn_model_available`, `churn_model_version` | Gauge | Status dan versi model aktif |
| `churn_requests_in_progress` | Gauge | Request yang sedang diproses |

### Hasil monitoring

Sistem diuji dengan lalu lintas campuran (prediksi tunggal, prediksi batch 5-40 pelanggan, serta data tidak valid dan bukan JSON untuk memicu error). Jumlah request dan prediksi adalah total sejak layanan gateway dijalankan; laju dan latensi dihitung pada jendela 15 menit terakhir. Hasil yang terbaca di Prometheus:

| Indikator | Nilai |
| ----------- | ----------- |
| Target `churn-gateway`, `tf-serving`, `prometheus` | Seluruhnya **UP** (ketersediaan gateway 100,0%) |
| Request `/predict` sukses (200) | 364 |
| Request ditolak: data tidak valid (422) | 31 |
| Request ditolak: bukan JSON atau jumlah salah (400) | 17 |
| Total prediksi | 1.552 (churn 366, tidak churn 1.186) |
| Rata-rata probabilitas churn | 0,284 |
| Laju request `/predict` (rata-rata jendela 15 menit) | 0,46 request/detik |
| Latensi `/predict` p50 / p95 | 6,4 ms / 123,1 ms |
| Latensi rata-rata TensorFlow Serving | 8,2 ms |
| `churn_model_available` | 1 (model `AVAILABLE`, versi 1793092663) |

Interpretasi: seluruh komponen sehat dan model selalu tersedia. Error yang tercatat berasal dari request uji yang sengaja tidak valid (kode 422 dan 400), bukan dari kegagalan model; tidak ada error pada TensorFlow Serving. Latensi p95 pada jendela 15 menit terakhir hanya 123,1 ms; pada grafik per 1 menit (`screenshots/3_prometheus_latency_p95.png`) p95 mulai sekitar 0,8 detik pada menit pertama (pemanasan), naik ke sekitar 0,1-0,33 detik saat request batch (hingga 40 pelanggan) masuk, dan berada di bawah 0,03 detik di luar itu. VM e2-micro memadai untuk model sekecil ini, tetapi batch besar akan menjadi pemicu pertama bila beban naik. Proporsi prediksi churn (23,6%) sedikit di bawah proporsi churn pada data latih (26,54%), wajar karena recall model sekitar 56% dan lalu lintas uji diambil dari dataset yang sama. Bila sebaran probabilitas atau proporsi prediksi bergeser jauh dari nilai ini pada lalu lintas nyata, itu menjadi sinyal awal *data drift* yang perlu diselidiki; metrik `churn_prediction_probability` disediakan untuk keperluan itu.

Screenshot dashboard monitoring: `fauzi19-monitoring.png` (halaman Prometheus pada browser). Tangkapan layar pendukung tersimpan di folder `screenshots/`:

- `screenshots/1_prometheus_targets.png`
- `screenshots/2_prometheus_request_rate.png`
- `screenshots/3_prometheus_latency_p95.png`
- `screenshots/4_prometheus_predictions_per_class.png`
- `screenshots/5_prometheus_errors.png`
- `screenshots/6_prometheus_model_available.png`
- `screenshots/7_prometheus_tfserving_requests.png`
- `screenshots/8_webapp_prediksi_churn.png`

## 8. Struktur Berkas

```text
.
├── fauzi19-pipeline/            # Artefak seluruh komponen TFX (pipeline root) dan metadata.sqlite
├── serving_model/               # Model hasil Pusher untuk TensorFlow Serving
├── data/telco_churn.csv         # Dataset bersih
├── deployment/                  # Gateway API, unit systemd, dan setup.sh untuk VM cloud
├── monitoring/                  # Kebutuhan Prometheus: Dockerfile, prometheus.yml, prometheus.config
├── screenshots/                 # Tangkapan layar pendukung (Prometheus, web app)
├── fauzi19-deployment.png       # Screenshot bukti sistem berjalan di cloud (URL metadata model pada address bar)
├── fauzi19-monitoring.png       # Screenshot dashboard monitoring Prometheus
├── fauzi19-pipeline.ipynb       # Notebook pipeline (sudah dijalankan)
├── fauzi19-testing.ipynb        # Notebook pengujian endpoint cloud (sudah dijalankan)
├── fauzi19_transform.py         # Modul Transform
├── fauzi19_trainer.py           # Modul Trainer
├── fauzi19_tuner.py             # Modul Tuner
├── Dockerfile                   # Image TensorFlow Serving
├── requirements.txt
└── README.md
```

## 9. Cara Menjalankan

1. Buka `fauzi19-pipeline.ipynb` di Google Colab dan pilih **Runtime > Run all**.
2. Sel pertama memasang TFX 1.21.0 dan pyarrow 18.1.0 bila belum tersedia. Bila impor pada sel berikutnya gagal setelah instalasi, restart runtime lalu jalankan ulang notebook.
3. Notebook mengunduh dataset, menulis ketiga modul `.py`, dan menjalankan seluruh komponen. Artefak tersimpan di `/content/fauzi19-pipeline` dan model serving di `/content/serving_model`.
4. Bagian terakhir notebook menjalankan TensorFlow Serving; setelah itu `fauzi19-testing.ipynb` dapat dijalankan pada runtime yang sama.

Lingkungan saat pembuatan: Python 3.13, TensorFlow 2.21.0, Keras 3, TFX 1.21.0, pyarrow 18.1.0 (versi lengkap pada `requirements.txt`).
5. Deployment ke cloud: ikuti `deployment/setup.sh` pada VM Ubuntu 22.04 (menyalin `serving_model`, `gateway.py`, direktori `monitoring/`, dan unit `systemd`).

## 10. Catatan dan Pengembangan Lanjutan

- Dataset memiliki *class imbalance* (26,54% churn); recall dapat ditingkatkan dengan *class weighting*, penyesuaian threshold, atau resampling.
- Monitoring saat ini mencakup kesehatan layanan, latensi, dan sebaran prediksi. Pengembangan lanjutan: alerting (Alertmanager), dasbor Grafana, pemantauan data drift terhadap statistik pipeline, dan retraining terjadwal.
- Gateway berjalan dengan satu worker pada VM kecil; untuk beban lebih besar dapat dipindah ke Cloud Run atau GKE dengan beberapa replika.

## 11. Referensi

- TensorFlow Extended: https://www.tensorflow.org/tfx
- TensorFlow Transform: https://www.tensorflow.org/tfx/transform
- TensorFlow Model Analysis: https://www.tensorflow.org/tfx/model_analysis
- TensorFlow Serving: https://www.tensorflow.org/tfx/serving/docker
- Telco Customer Churn (IBM): https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- Prometheus: https://prometheus.io/docs/introduction/overview/
- Google Compute Engine: https://cloud.google.com/compute/docs

