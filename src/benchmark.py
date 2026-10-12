"""
Edge Benchmarking
==================
Measures single-threaded inference performance of an ONNX model:
  - Mean ± std and p99 latency per beat (µs)
  - Sequential throughput (one call per beat) and batched throughput
  - Process memory (RSS) increase during inference
  - Model file size
  - Accuracy and per-class sensitivity on DS2

Latency covers classifier inference only, not filtering or feature extraction.

Usage
-----
    python src/benchmark.py                       # uses configs/default.yaml
    python src/benchmark.py --exp E3 --clf xgboost
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from aami_mappings import EVAL_CLASSES, EXCLUDED_CLASSES, AAMI_TO_INT
from utils import get_logger, load_config, ensure_dirs, project_root

log = get_logger("benchmark")

EVAL_LABELS = [AAMI_TO_INT[c] for c in EVAL_CLASSES]
EXCLUDED_LABELS = [AAMI_TO_INT[c] for c in EXCLUDED_CLASSES]


def load_test_data(
    exp: str, clf_name: str, results_dir: Path, features_csv: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Load DS2 test features for the given experiment."""
    report_path = results_dir / f"{exp}_{clf_name}_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}. Run train.py first.")

    with open(report_path) as f:
        report = json.load(f)

    feature_cols = report["features"]
    df = pd.read_csv(features_csv)
    df_test = df[df["split"] == "ds2"]
    cols_present = [c for c in feature_cols if c in df_test.columns]
    sub = df_test[cols_present + ["label"]].dropna()
    sub = sub[~sub["label"].isin(EXCLUDED_LABELS)]
    X = sub[cols_present].values.astype(np.float32)
    y = sub["label"].values.astype(int)
    return X, y


def _rss_kb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1024


def benchmark_onnx(
    onnx_path: Path,
    X: np.ndarray,
    y_true: np.ndarray,
    n_repeats: int = 10000,
    label: str = "",
) -> dict:
    """Benchmark an ONNX model: latency, throughput, memory, accuracy."""
    import onnxruntime as ort
    from sklearn.metrics import accuracy_score, recall_score

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}. Run export_onnx.py first.")

    gc.collect()
    rss_before = _rss_kb()

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1   # single thread
    sess_options.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_options,
                                providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Warmup
    sample = X[:1]   # one beat at a time
    for _ in range(100):
        sess.run(None, {input_name: sample})

    # Single-beat latency
    latencies_us = np.empty(n_repeats)
    t_loop = perf_counter()
    for i in range(n_repeats):
        t0 = perf_counter()
        sess.run(None, {input_name: sample})
        latencies_us[i] = (perf_counter() - t0) * 1e6
    loop_time = perf_counter() - t_loop

    rss_after = _rss_kb()

    mean_lat = float(np.mean(latencies_us))
    std_lat  = float(np.std(latencies_us))
    p99_lat  = float(np.percentile(latencies_us, 99))
    seq_throughput = n_repeats / loop_time   # beats/s, one call per beat

    # Batched throughput (all DS2 beats in one call)
    t_batch_start = perf_counter()
    out = sess.run(None, {input_name: X})
    t_batch = perf_counter() - t_batch_start
    batch_throughput = len(X) / t_batch

    # Accuracy on DS2
    y_pred = np.asarray(out[0]).flatten().astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    sens = recall_score(y_true, y_pred, labels=EVAL_LABELS, average=None, zero_division=0)

    file_size_kb = onnx_path.stat().st_size / 1024

    result = {
        "model":                 label or onnx_path.name,
        "cpu":                   platform.processor() or platform.machine(),
        "file_size_kb":          file_size_kb,
        "latency_mean_us":       mean_lat,
        "latency_std_us":        std_lat,
        "latency_p99_us":        p99_lat,
        "seq_throughput_bps":    seq_throughput,
        "batch_throughput_bps":  batch_throughput,
        "rss_increase_kb":       rss_after - rss_before,
        "accuracy_ds2":          acc,
        **{f"sensitivity_{c}": float(s) for c, s in zip(EVAL_CLASSES, sens)},
    }

    log.info(
        f"  [{result['model']}] size={file_size_kb:.1f} KB  "
        f"latency={mean_lat:.2f}±{std_lat:.2f} µs (p99={p99_lat:.2f} µs)  "
        f"sequential={seq_throughput:.0f} beats/s  batched={batch_throughput:.0f} beats/s  "
        f"RSS +{rss_after - rss_before:.0f} KB  acc={acc:.3f}"
    )
    return result


def main(
    config_path: str = "configs/default.yaml",
    exp: str = "E3",
    clf_name: str = "xgboost",
) -> None:
    cfg = load_config(config_path)
    root = project_root()

    features_csv = root / cfg["data"]["features_csv"]
    results_dir  = root / cfg["results"]["dir"]
    models_dir   = root / cfg["results"]["models_dir"]
    n_repeats    = int(cfg.get("benchmark", {}).get("n_repeats", 10000))

    ensure_dirs(results_dir)

    onnx_path = models_dir / f"{exp}_{clf_name}.onnx"

    log.info(f"Loading test data for {exp}/{clf_name} ...")
    X, y_true = load_test_data(exp, clf_name, results_dir, features_csv)
    log.info(f"Test set: {X.shape[0]} beats, {X.shape[1]} features")

    log.info(f"Benchmarking {n_repeats} single-beat inferences (single thread) ...")
    result = benchmark_onnx(onnx_path, X, y_true, n_repeats, label=f"{exp}_{clf_name}_onnx")

    log.info("\n" + "\n".join(
        f"  {k:22s} {v:.2f}" if isinstance(v, float) else f"  {k:22s} {v}"
        for k, v in result.items()
    ))

    bench_path = results_dir / f"benchmark_{exp}_{clf_name}.json"
    with open(bench_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Benchmark results saved to {bench_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark an ONNX model on the host CPU")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--exp",    default="E3")
    parser.add_argument("--clf",    default="xgboost")
    args = parser.parse_args()
    main(args.config, args.exp, args.clf)
