"""
Raw Beat Windows
================
Extracts the filtered signal window around every annotated beat, keyed by
(record, beat_index) so it lines up row-for-row with data/features_beat.csv.

The CNN in cnn_model.py consumes these windows. Segmentation and filtering
are identical to dataset.py, so both models see exactly the same beats.

Note: data/features_beat.csv was built with a 90/110-sample window (200
samples), before post_samples was raised to 144 in the config. Pass
--post_samples 110 so the CNN sees exactly the same window as the features.

Usage
-----
    python src/beat_windows.py --post_samples 110   # writes data/beat_windows.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).parent))

from aami_mappings import SYMBOL_TO_AAMI, AAMI_TO_INT
from preprocess import bandpass_ecg, load_mitbih_record
from utils import load_config, get_logger, project_root

log = get_logger("beat_windows")


def record_windows(record_id: str, data_dir: Path, prep_cfg: dict, seg_cfg: dict):
    """Return (windows, beat_index, labels) for one record."""
    pre = int(seg_cfg.get("pre_samples", 90))
    post = int(seg_cfg.get("post_samples", 110))
    notch = prep_cfg.get("notch_freq", 60)

    x, fs = load_mitbih_record(record_id, data_dir)
    x_filt = bandpass_ecg(
        x, fs,
        low=float(prep_cfg.get("bandpass_low", 0.5)),
        high=float(prep_cfg.get("bandpass_high", 40.0)),
        order=int(prep_cfg.get("filter_order", 4)),
        notch=int(notch) if notch else None,
    )

    ann = wfdb.rdann(str(data_dir / record_id), "atr")
    samples = np.asarray(ann.sample, dtype=int)
    symbols = np.asarray(ann.symbol, dtype="<U5")
    mask = np.array([s in SYMBOL_TO_AAMI for s in symbols])
    samples, symbols = samples[mask], symbols[mask]

    wins, idxs, labels = [], [], []
    for beat_idx, (r_pos, sym) in enumerate(zip(samples, symbols)):
        s0, s1 = r_pos - pre, r_pos + post
        if s0 < 0 or s1 > len(x_filt):
            continue
        wins.append(x_filt[s0:s1].astype(np.float32))
        idxs.append(beat_idx)
        labels.append(AAMI_TO_INT[SYMBOL_TO_AAMI[sym]])

    return np.stack(wins), np.array(idxs), np.array(labels)


def main(config_path: str = "configs/default.yaml", out: str = "data/beat_windows.npz",
         post_samples: int | None = None):
    cfg = load_config(config_path)
    root = project_root()
    data_dir = root / cfg["data"]["mitbih_dir"]
    prep_cfg = cfg.get("preprocessing", {})
    seg_cfg = dict(cfg.get("segmentation", {}))
    if post_samples is not None:
        seg_cfg["post_samples"] = post_samples
    log.info(f"Window: {seg_cfg.get('pre_samples')} before / {seg_cfg.get('post_samples')} after R-peak")

    all_w, all_rec, all_idx, all_y = [], [], [], []
    for split in ("ds1", "ds2"):
        for rid in cfg["splits"][split]:
            rid = str(rid)
            w, idx, y = record_windows(rid, data_dir, prep_cfg, seg_cfg)
            all_w.append(w)
            all_idx.append(idx)
            all_y.append(y)
            all_rec.append(np.full(len(idx), rid))
            log.info(f"[{split}] {rid}: {len(idx)} beats")

    np.savez_compressed(
        root / out,
        windows=np.concatenate(all_w),
        record=np.concatenate(all_rec),
        beat_index=np.concatenate(all_idx),
        label=np.concatenate(all_y),
    )
    log.info(f"Saved {sum(len(i) for i in all_idx)} windows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract raw beat windows for the CNN")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="data/beat_windows.npz")
    parser.add_argument("--post_samples", type=int, default=None,
                        help="override config; must match the window used to build features_beat.csv")
    args = parser.parse_args()
    main(args.config, args.out, args.post_samples)
