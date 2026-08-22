"""
Dataset Builder — Per-Beat, Inter-Patient (AAMI DS1/DS2)
=========================================================
Builds the full feature matrix for training and evaluation.

Key design choices:
  - Per-beat extraction (not windowed): each row = one annotated beat
  - AAMI DS1/DS2 inter-patient split — NO patient appears in both sets
  - Labels are 5-class AAMI integers: N=0, S=1, V=2, F=3, Q=4
  - Records 102, 104, 107, 217 are excluded (pacemaker patients)

Usage
-----
    python src/dataset.py                      # uses configs/default.yaml
    python src/dataset.py --config <path>      # custom config
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).parent))

from aami_mappings import SYMBOL_TO_AAMI, AAMI_TO_INT, AAMI_CLASSES
from features import extract_beat_features
from preprocess import bandpass_ecg, load_mitbih_record
from detect_rpeaks import detect_rpeaks
from utils import load_config, set_seed, get_logger, ensure_dirs, project_root

log = get_logger("dataset")


# Record loading

def load_annotations(record_id: str | int, data_dir: Path):
    """Return (samples, symbols) numpy arrays for a record's annotations."""
    ann = wfdb.rdann(str(data_dir / str(record_id)), "atr")
    samples = np.asarray(ann.sample, dtype=int)
    symbols = np.asarray(ann.symbol, dtype="<U5")
    return samples, symbols


def record_to_beats(
    record_id: str | int,
    data_dir: Path,
    prep_cfg: dict,
    seg_cfg: dict,
    feat_cfg: dict,
) -> list[dict]:
    """Process one record and return a list of per-beat feature dicts.

    Each dict contains:
        - All extracted features (RR, morphological, frequency-domain)
        - 'label_aami'  : str AAMI class (N/S/V/F/Q)
        - 'label'       : int 0-4
        - 'record'      : record id (str)
        - 'beat_index'  : index of beat within record
    """
    record_id = str(record_id)
    fs = float(prep_cfg.get("fs", 360))
    pre  = int(seg_cfg.get("pre_samples",  90))
    post = int(seg_cfg.get("post_samples", 110))
    notch = prep_cfg.get("notch_freq", 60)

    # Load and filter signal
    x, fs_actual = load_mitbih_record(record_id, data_dir)
    fs = fs_actual  # use actual FS from record header
    x_filt = bandpass_ecg(
        x, fs,
        low=float(prep_cfg.get("bandpass_low", 0.5)),
        high=float(prep_cfg.get("bandpass_high", 40.0)),
        order=int(prep_cfg.get("filter_order", 4)),
        notch=int(notch) if notch else None,
    )

    # Load annotations
    ann_samples, ann_symbols = load_annotations(record_id, data_dir)

    # Detect R-peaks (for RR feature context)
    # We use annotated R-peak positions as ground truth for beat segmentation
    # (more reliable than algorithmic detection for feature extraction)
    # Filter to only recognised beat annotation symbols
    beat_mask = np.array([s in SYMBOL_TO_AAMI for s in ann_symbols])
    beat_samples = ann_samples[beat_mask]
    beat_symbols = ann_symbols[beat_mask]

    n_sig = len(x_filt)
    rows: list[dict] = []

    for beat_idx, (r_pos, symbol) in enumerate(zip(beat_samples, beat_symbols)):
        aami_class = SYMBOL_TO_AAMI.get(symbol)
        if aami_class is None:
            continue

        # Segment beat window
        s_start = r_pos - pre
        s_end   = r_pos + post
        if s_start < 0 or s_end > n_sig:
            continue  # skip edge beats

        beat = x_filt[s_start:s_end]

        # Extract features
        feats = extract_beat_features(
            beat_idx=beat_idx,
            beat=beat,
            r_peaks=beat_samples,
            fs=fs,
            feat_cfg=feat_cfg,
        )

        feats["label_aami"]  = aami_class
        feats["label"]       = AAMI_TO_INT[aami_class]
        feats["record"]      = record_id
        feats["beat_index"]  = beat_idx

        rows.append(feats)

    return rows


# Dataset builder

def build_dataset(
    records: list[str | int],
    data_dir: Path,
    prep_cfg: dict,
    seg_cfg: dict,
    feat_cfg: dict,
    split_name: str = "",
) -> pd.DataFrame:
    """Build a DataFrame of per-beat features for a list of records."""
    all_rows: list[dict] = []
    for rid in records:
        log.info(f"[{split_name}] Processing record {rid} ...")
        try:
            rows = record_to_beats(rid, data_dir, prep_cfg, seg_cfg, feat_cfg)
            all_rows.extend(rows)
            log.info(f"  → {len(rows)} beats extracted")
        except Exception as e:
            log.warning(f"  ERROR on record {rid}: {e}")

    df = pd.DataFrame(all_rows)
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    log.info(
        f"[{split_name}] Total beats: {len(df)}  |  "
        f"Class dist: {df['label_aami'].value_counts().to_dict() if 'label_aami' in df.columns else 'n/a'}"
    )
    return df


# CLI entry point

def main(config_path: str = "configs/default.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    root = project_root()
    data_dir = root / cfg["data"]["mitbih_dir"]
    out_csv  = root / cfg["data"]["features_csv"]

    ensure_dirs(out_csv.parent, root / cfg["results"]["dir"], root / cfg["results"]["models_dir"])

    prep_cfg = cfg.get("preprocessing", {})
    seg_cfg  = cfg.get("segmentation", {})
    feat_cfg = cfg.get("features", {})

    ds1_records = [str(r) for r in cfg["splits"]["ds1"]]
    ds2_records = [str(r) for r in cfg["splits"]["ds2"]]

    log.info("=== Building DS1 (training set) ===")
    df_ds1 = build_dataset(ds1_records, data_dir, prep_cfg, seg_cfg, feat_cfg, split_name="DS1")
    df_ds1["split"] = "ds1"

    log.info("=== Building DS2 (test set) ===")
    df_ds2 = build_dataset(ds2_records, data_dir, prep_cfg, seg_cfg, feat_cfg, split_name="DS2")
    df_ds2["split"] = "ds2"

    df_all = pd.concat([df_ds1, df_ds2], ignore_index=True)
    df_all.to_csv(out_csv, index=False)
    log.info(f"Saved {len(df_all)} beats to {out_csv}")
    log.info(f"   DS1: {len(df_ds1)} beats | DS2: {len(df_ds2)} beats")

    return df_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build per-beat ECG feature dataset")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
