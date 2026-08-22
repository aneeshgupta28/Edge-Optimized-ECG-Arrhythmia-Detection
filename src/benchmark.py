"""
Edge Benchmarking
==================
Measures inference performance of saved ONNX models:
  - Mean ± std latency per beat (µs)
  - Throughput (beats/second)
  - Peak memory usage during inference (tracemalloc)
  - Model file size
  - Accuracy delta: quantized vs original

Usage
-----
    python src/benchmark.py                       # uses configs/default.yaml
    python src/benchmark.py --exp E3 --clf xgboost
"""

from __future__ import annotations

import argparse
import json
import sys
import tracemalloc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, load_config, ensure_dirs, project_root

log = get_logger("benchmark")


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
    X = sub[cols_present].values.astype(np.float32)
    y = sub["label"].values.astype(int)
    return X, y


def benchmark_onnx(
    onnx_path: Path,
    X: np.ndarray,
    y_true: np.ndarray,
    n_repeats: int = 10000,
    label: str = "",
) -> dict:
    """Benchmark an ONNX model: latency, throughput, memory, accuracy."""
    import onnxruntime as ort
    from sklearn.metrics import accuracy_score

    if not onnx_path.exists():
        log.warning(f"ONNX file not found: {onnx_path} — skipping")
        return {}

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1   # single-threaded → edge simulation
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_options)
    input_name = sess.get_inputs()[0].name

    # Warmup
    for _ in range(10):
        sess.run(None, {input_name: X[:1]})

    # Latency: single-beat inference, n_repeats times
    sample = X[:1]   # one beat at a time
    latencies_us = []
    for _ in range(n_repeats):
        t0 = perf_counter()
        sess.run(None, {input_name: sample})
        latencies_us.append((perf_counter() - t0) * 1e6)

    latencies_us = np.array(latencies_us)
    mean_lat = float(np.mean(latencies_us))
    std_lat  = float(np.std(latencies_us))
    p99_lat  = float(np.percentile(latencies_us, 99))

    # Throughput: batch inference
    t_batch_start = perf_counter()
    out = sess.run(None, {input_name: X})
    t_batch = perf_counter() - t_batch_start
    throughput = len(X) / t_batch  # beats/second

    # Peak memory
    tracemalloc.start()
    sess.run(None, {input_name: X[:1]})
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_kb = peak_mem / 1024

    # Accuracy on DS2
    y_pred_raw = out[0]
    if hasattr(y_pred_raw, "flatten"):
        y_pred = y_pred_raw.flatten().astype(int)
    else:
        y_pred = np.array(y_pred_raw).flatten().astype(int)

    if len(y_pred) == len(y_true):
        acc = accuracy_score(y_true, y_pred)
    else:
        acc = float("nan")

    file_size_kb = onnx_path.stat().st_size / 1024

    result = {
        "model":           label or onnx_path.name,
        "file_size_kb":    file_size_kb,
        "latency_mean_us": mean_lat,
        "latency_std_us":  std_lat,
        "latency_p99_us":  p99_lat,
        "throughput_bps":  throughput,
        "peak_memory_kb":  peak_mem_kb,
        "accuracy_ds2":    acc,
    }

    log.info(
        f"  [{label}] "
        f"size={file_size_kb:.1f} KB  "
        f"latency={mean_lat:.1f}±{std_lat:.1f} µs (p99={p99_lat:.1f} µs)  "
        f"throughput={throughput:.0f} beats/s  "
        f"mem={peak_mem_kb:.1f} KB  "
        f"acc={acc:.3f}"
    )
    return result


def print_summary_table(results: list[dict]) -> None:
    """Pretty-print a benchmark summary table."""
    if not results:
        return
    df = pd.DataFrame(results)
    log.info("\n" + "="*80)
    log.info("EDGE BENCHMARK SUMMARY")
    log.info("="*80)
    cols = ["model", "file_size_kb", "latency_mean_us", "latency_p99_us",
            "throughput_bps", "peak_memory_kb", "accuracy_ds2"]
    cols_present = [c for c in cols if c in df.columns]
    log.info("\n" + df[cols_present].to_string(index=False, float_format="%.2f"))


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

    onnx_path  = models_dir / f"{exp}_{clf_name}.onnx"
    quant_path = models_dir / f"{exp}_{clf_name}_int8.onnx"

    log.info(f"Loading test data for {exp}/{clf_name} ...")
    X, y_true = load_test_data(exp, clf_name, results_dir, features_csv)
    log.info(f"Test set: {X.shape[0]} beats, {X.shape[1]} features")

    log.info(f"\nBenchmarking {n_repeats} single-beat inferences (single-threaded) ...")
    results = []

    r1 = benchmark_onnx(onnx_path,  X, y_true, n_repeats, label=f"{exp}_{clf_name}_fp32")
    if r1:
        results.append(r1)

    r2 = benchmark_onnx(quant_path, X, y_true, n_repeats, label=f"{exp}_{clf_name}_int8")
    if r2:
        results.append(r2)

    if len(results) == 2:
        delta_acc = results[1]["accuracy_ds2"] - results[0]["accuracy_ds2"]
        size_ratio = results[1]["file_size_kb"] / results[0]["file_size_kb"]
        speedup    = results[0]["latency_mean_us"] / results[1]["latency_mean_us"]
        log.info(f"\n  Quantization delta: acc={delta_acc:+.4f}  size={size_ratio:.2f}x  speedup={speedup:.2f}x")

    print_summary_table(results)

    # Save benchmark results
    bench_path = results_dir / f"benchmark_{exp}_{clf_name}.json"
    with open(bench_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Benchmark results saved to {bench_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ONNX models for edge deployment")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--exp",    default="E3")
    parser.add_argument("--clf",    default="xgboost")
    args = parser.parse_args()
    main(args.config, args.exp, args.clf)
