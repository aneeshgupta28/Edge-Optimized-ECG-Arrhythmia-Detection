# Edge-Optimized ECG Arrhythmia Detection (MIT-BIH)

Personal project showcasing signal processing + ML + edge optimization on the MIT-BIH Arrhythmia dataset.

## Highlights
This project applies machine learning (XGBoost) to detect cardiac arrhythmias (atrial fibrillation, tachycardia, bradycardia) from ECG signals with 94% accuracy.

Preprocessed ECG data with filtering and feature extraction.

Trained and evaluated an XGBoost classifier.

Converted the model to ONNX and applied dynamic INT8 quantization to optimize it for edge deployment with minimal accuracy loss.

Demonstrated feasibility of real-time inference on resource-constrained devices for potential use in portable medical applications.

## Quickstart
```bash
# 1) create env
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 2) place MIT-BIH files under ./data/mitbih/
#    e.g., data/mitbih/100.dat, 100.hea, 100.atr

# 3) sanity check: filter demo
python src/plot_filter_demo.py --data_dir data/mitbih --record 100