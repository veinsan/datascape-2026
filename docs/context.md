# DATASCAPE 2026 - Data Mining Competition

DATASCAPE 2026 - Data Mining Competition: E-Commerce Package Quality Control


Overview
--------

# DATASCAPE 2026 - Data Mining Competition: E-Commerce Package Quality Control

### Bridge the Gap: From Raw Data to Real Impact

Dalam pesatnya pertumbuhan industri e-commerce dan logistik nasional, kepuasan pelanggan dan efisiensi operasional ditentukan oleh integritas barang yang dikirimkan. Setiap harinya, jutaan paket melintasi rantai pasok. Kerusakan fisik pada paket bukan hanya berdampak pada kerugian finansial, melainkan juga menggerus kepercayaan konsumen.

Namun, di lapangan, data visual yang diterima sering kali jauh dari ideal. Foto-foto Quality Control (QC) dari kurir atau pusat pemrosesan (hub) bersifat heterogen, berderau (noisy), memiliki variasi pencahayaan yang ekstrem, serta ketidakseimbangan distribusi kelas (class imbalance).

DATASCAPE 2026 yang diselenggarakan oleh Himpunan Mahasiswa Data Science (HIMA DS) Telkom University menantang para talenta data muda Indonesia untuk menjembatani kesenjangan ini ("Bridge the Gap"). Peserta ditantang untuk mengolah data visual mentah (raw data) dari lapangan menjadi solusi otomatisasi prediksi (real impact) yang presisi, tangguh, dan siap pakai.

### Description

### Problem Statement

Peserta diminta untuk membangun model Computer Vision / Deep Learning yang mampu mengklasifikasikan kondisi fisik paket ke dalam 3 kategori utama:

1. Aman: Paket/dus dalam kondisi utuh, mulus, dan layak kirim.
2. Rusak: Paket/dus mengalami kerusakan fisik (penyok, sobek, basah, atau hancur).
3. Lainnya: Kemasan non-standar / out-of-distribution (seperti food packaging, kantong plastik belanjaan, atau paper bag).

---

### Evaluation

## Evaluation Metric

Evaluasi kompetisi ini menggunakan metrik **Macro-Averaged F1-Score (Macro F1-Score)**. Metrik ini dihitung dengan mencari nilai

| F1\text{-Score} |
| --------------- |

untuk masing-masing kelas, kemudian menghitung rata-ratanya secara tidak terbobot (*unweighted mean*):



| \text{Macro } F1 = \frac{F1\_{\text{Aman}} + F1\_{\text{Rusak}} + F1\_{\text{Lainnya}}}{3} |
| ------------------------------------------------------------------------------------------ |

Di mana untuk setiap kelas:



| F1_c = 2 \times \frac{\text{Precision}\_c \times \text{Recall}\_c}{\text{Precision}\_c + \text{Recall}\_c}<br>Alasan Penggunaan Metrik Ini |
| ------------------------------------------------------------------------------------------------------------------------------------------ |

Metrik Macro F1-Score memberikan bobot yang setara untuk ketiga kelas tanpa memedulikan proporsi jumlah datanya. Ini memaksa model untuk tidak hanya unggul pada kelas mayoritas (**Aman**), melainkan juga harus presisi dalam mendeteksi kelas minoritas (**Rusak** dan **Lainnya**).

## Submission File

Submission file harus mengikuti format file submission.
```stylus
filename,label
test_00001.jpg,Aman
test_00002.jpg,Aman
test_00003.jpg,Aman
...

```

### Citation

Zaky Muhammad Fauzi. DATASCAPE 2026 - Data Mining Competition. [https://www.kaggle.com/competitions/datascape-2026-data-mining-competition](https://www.kaggle.com/competitions/datascape-2026-data-mining-competition), 2026. Kaggle.

## Dataset Description

## Dataset Description

Dataset terdiri dari 15.281 gambar fisik paket yang terbagi menjadi Data Train dan Data Test.

### File Structure

- `Train_Set`: Folder berisi seluruh gambar yang telah dikelompokkan berdasarkan labelnya untuk proses training model.
- `Test_Set`: Folder berisi seluruh gambar data uji tanpa label.
- `sample_submission.csv`: Format standar pengunggahan hasil prediksi peserta dengan kolom:
  - `filename`: Nama file data uji (contoh: test_00001.jpg)
  - `label`: Prediksi kelas oleh model Anda (Aman, Rusak, atau Lainnya)
