from __future__ import annotations
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from typing import Iterable

from preprocess import bandpass_ecg, load_mitbih_record
from detect_rpeaks import detect_rpeaks
from features import extract_window_features


def load_annotations(record_id: str | int, data_dir: str | Path):
    """Return annotation samples and symbols as numpy arrays (absolute sample indices)."""
    ann = wfdb.rdann(str(Path(data_dir) / str(record_id)), 'atr')
    samples = np.asarray(ann.sample, dtype=int)
    symbols = np.asarray(ann.symbol, dtype='<U5')
    return samples, symbols


def choose_window_label(ann_samples: np.ndarray, ann_symbols: np.ndarray, start: int, end: int, mode: str = 'binary'):
    """Choose label for a window [start, end).

    - mode == 'binary': return 0 (normal) if no annotation or all 'N', else 1 (abnormal)
    - mode == 'symbol': return 'N' if no annotation or all 'N', else first non-'N' symbol in the window
    """
    idxs = np.where((ann_samples >= start) & (ann_samples < end))[0]
    if len(idxs) == 0:
        return 0 if mode == 'binary' else 'N'

    syms = ann_symbols[idxs]
    # if all normal
    if np.all(syms == 'N'):
        return 0 if mode == 'binary' else 'N'

    # there is at least one non-normal symbol
    non_n = syms[syms != 'N']
    if mode == 'binary':
        return 1
    else:
        # pick the most frequent non-'N' symbol in this window (robust when multiple)
        vals, counts = np.unique(non_n, return_counts=True)
        return vals[np.argmax(counts)]


def record_to_features(record_id: str | int, data_dir: str | Path, window: float = 10.0, notch: int | None = 50, label_mode: str = 'binary'):
    """Process one record and return a list of feature dicts with labels.

    Each dict contains HRV features (from features.extract_window_features) plus:
    - 'record' : record id
    - 'label'  : label (0/1 for binary or symbol for 'symbol' mode)
    """
    x, fs = load_mitbih_record(record_id, data_dir)
    x_filt = bandpass_ecg(x, fs, notch=notch)
    r_idx = detect_rpeaks(x_filt, fs)

    ann_samples, ann_symbols = load_annotations(record_id, data_dir)

    feats = extract_window_features(x_filt, r_idx, fs, window=window)
    rows: list[dict] = []

    win_len = int(window * fs)
    for f in feats:
        start = int(f["window_start"] * fs)
        end = start + win_len
        lbl = choose_window_label(ann_samples, ann_symbols, start, end, mode=label_mode)
        f["label"] = lbl
        f["record"] = str(record_id)
        rows.append(f)

    return rows


def build_dataset(records: Iterable[str | int], data_dir: str | Path, out_csv: str | Path, window: float = 10.0, notch: int | None = 50, label_mode: str = 'binary'):
    """Build dataset for multiple records and save CSV.

    Parameters
    ----------
    records: iterable of record ids (strings or ints)
    data_dir: path containing MIT-BIH files
    out_csv: output CSV path
    window: window length in seconds
    notch: mains notch frequency (50/60 or None)
    label_mode: 'binary' or 'symbol'
    """
    rows = []
    for rid in records:
        print(f"Processing record {rid} ...")
        try:
            rec_rows = record_to_features(rid, data_dir, window=window, notch=notch, label_mode=label_mode)
            rows.extend(rec_rows)
            print(f"  -> {len(rec_rows)} windows")
        except Exception as e:
            print(f"  ERROR processing {rid}: {e}")

    df = pd.DataFrame(rows)
    # ensure label column exists and for binary mode, enforce 0/1 int
    if label_mode == 'binary' and 'label' in df.columns:
        df['label'] = df['label'].astype(int)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved {len(df)} rows to {out_csv}")
    return df


if __name__ == '__main__':
    # example records - change as needed
    data_dir = Path("/Users/aneesh/dev/ecg-edge-project/data/mitbih")

    # Load all record IDs from RECORDS file
    records_file = data_dir / "RECORDS"
    with open(records_file, "r") as f:
        records = [line.strip() for line in f if line.strip()]

    out_csv = Path("data/features.csv")
    build_dataset(records, data_dir, out_csv, window=10.0, notch=50, label_mode='binary')