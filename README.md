# Edge-Optimized ECG Arrhythmia Detection

This repository presents a machine learning pipeline for single-lead **per-beat ECG arrhythmia classification** evaluated under the ANSI/AAMI EC57 inter-patient protocol (DS1/DS2 record split) on the **MIT-BIH Arrhythmia Database** (100,705 annotated heartbeats across 44 patients).

Trained classifiers are exported to **ONNX** format and benchmarked for embedded edge deployment, measuring single-beat inference latency, peak RAM memory footprint, and throughput under single-threaded execution constraints.

---

## Architecture & System Flow

```
+------------------+     +-------------------+     +-------------------------+
| Single-Lead ECG  | --> | Bandpass (0.5-40) | --> | R-Peak Segmentation     |
| (MIT-BIH @ 360Hz)|     | & Notch (60 Hz)   |     | [-250 ms, +400 ms]      |
+------------------+     +-------------------+     +-------------------------+
                                                               |
                                                               v
+------------------+     +-------------------+     +-------------------------+
| Single-Beat ONNX | <-- | Quantization /    | <-- | 24 Domain Features      |
| Benchmark (<18us)|     | ONNX Export       |     | (RR, Wavelet, DCT, HRV) |
+------------------+     +-------------------+     +-------------------------+
```

---

## Methodological Highlights

- **Inter-Patient Evaluation Protocol:** Strict ANSI/AAMI EC57 DS1 (22 train patients, 50,977 beats) vs DS2 (22 test patients, 49,668 beats) split to prevent data leakage across patient records.
- **5-Class AAMI Standard Classification:** Normal (N), Supraventricular Ectopic (S), Ventricular Ectopic (V), Fusion (F), and Unknown (Q).
- **Domain Feature Engineering (24 Features):** RR timing intervals, QRS/ST morphology, 4-level `db4` discrete wavelet transform energy sub-bands, Discrete Cosine Transform (DCT) shape descriptors, and Welch PSD frequency-domain HRV.
- **Edge Deployment Profiling:** Single-threaded ONNX Runtime benchmarking measuring inference latency ($\mu$s/beat), peak RAM memory footprint (KB), and throughput (beats/sec).

---

## Benchmark & Experimental Results

### Inter-Patient Evaluation Results (DS2 Test Set)

| Experiment | Classifier | Accuracy | Macro F1 | Sensitivity (S) | Sensitivity (V) | Precision (V) |
|---|---|---|---|---|---|---|
| **E1 (RR-only)** | XGBoost | 91.7% | 0.356 | 5.4% | 75.7% | 70.7% |
| **E2 (Morphology)**| XGBoost | 88.9% | 0.369 | 2.4% | 84.2% | 57.0% |
| **E3 (Full)** | Logistic Regression | 79.1% | 0.437 | 58.9% | 88.3% | 75.3% |
| **E3 (Full)** | LinearSVC | 78.9% | 0.439 | 53.8% | 84.2% | 76.5% |
| **E3 (Full)** | Random Forest | 92.6% | 0.386 | 3.2% | 94.6% | 85.3% |
| **E3 (Full)** | **XGBoost** | **92.8%** | **0.424** | **10.2%** | **95.6%** | **86.5%** |

### ONNX Edge Performance Profile (XGBoost E3 Model)

| Metric | Target / Specification | Measured Result |
|---|---|---|
| **Model Size (ONNX FP32)** | Single-file ONNX binary | **2.30 MB** (2,302 KB) |
| **Inference Latency (Mean +/- Std)** | Single-threaded CPU execution | **17.57 us +/- 0.71 us** (0.017 ms) |
| **Worst-Case Latency (p99)** | 99th percentile execution bound | **20.29 us** |
| **Inference Throughput** | Sequential single-beat predictions | **99,988 beats / sec** |
| **Peak Runtime RAM Footprint** | Memory allocation during inference | **1.05 KB** |
| **Prediction Agreement** | ONNX Runtime vs Scikit-Learn | **99.0%** |

---

## Repository Structure

```
├── configs/
│   └── default.yaml          # Hyperparameters & split definitions
├── src/
│   ├── preprocess.py          # Bandpass filter & record loading
│   ├── detect_rpeaks.py       # R-peak detection utilities
│   ├── aami_mappings.py       # MIT-BIH symbol to 5 AAMI classes
│   ├── features.py            # Beat feature extraction (RR, wavelet, DCT, HRV)
│   ├── dataset.py             # Feature matrix builder (DS1/DS2 splits)
│   ├── train.py               # Cross-validation & classifier training
│   ├── evaluate.py            # Confusion matrices & evaluation plots
│   ├── export_onnx.py         # ONNX model export & verification
│   ├── benchmark.py           # Single-beat latency, memory, throughput profiling
│   └── utils.py               # Config loader, random seed, and logger setup
├── notebooks/
│   └── eda_and_signal_viz.ipynb # Exploratory data analysis & beat visualization
└── requirements.txt           # Python dependencies
```

---

## Quickstart

### 1. Environment Setup

```bash
git clone https://github.com/aneeshgupta28/Edge-Optimized-ECG-Arrhythmia-Detection.git
cd Edge-Optimized-ECG-Arrhythmia-Detection

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download MIT-BIH Arrhythmia Data

Download the MIT-BIH dataset into `data/mitbih/`:

```bash
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/mitbih/')"
```

### 3. Build Feature Matrix

```bash
python src/dataset.py
```

### 4. Train Classifiers

```bash
python src/train.py
```

### 5. Export to ONNX & Profile Edge Latency

```bash
python src/export_onnx.py --exp E3 --clf xgboost
python src/benchmark.py --exp E3 --clf xgboost
```

---

## References

1. de Chazal, P., O'Dwyer, M., & Reilly, R. B. (2004). Automatic classification of heartbeats using ECG morphology and heartbeat interval features. *IEEE Transactions on Biomedical Engineering*, 51(7), 1196-1206.
2. ANSI/AAMI EC57:2012 - Standard for testing and reporting arrhythmia detection algorithms.
3. Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH arrhythmia database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45-50.