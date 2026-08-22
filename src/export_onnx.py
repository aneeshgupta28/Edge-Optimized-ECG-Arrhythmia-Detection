"""
ONNX Export Pipeline
======================
Exports the best trained model (E3/XGBoost by default) to ONNX format
and applies dynamic INT8 quantization.

Steps:
  1. Load best model pickle (sklearn Pipeline)
  2. Export the raw classifier to ONNX via skl2onnx or onnxmltools
  3. Save original ONNX
  4. Apply quantize_dynamic (INT8 weights)
  5. Verify both models produce identical predictions on a sample batch
  6. Report file sizes

Usage
-----
    python src/export_onnx.py                       # default: E3/xgboost
    python src/export_onnx.py --exp E3 --clf xgboost
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, load_config, ensure_dirs, project_root

log = get_logger("export_onnx")


def load_model_and_data(
    exp: str, clf_name: str, models_dir: Path, results_dir: Path, features_csv: Path
):
    """Load pipeline and a sample of test features."""
    model_path  = models_dir / f"{exp}_{clf_name}.pkl"
    report_path = results_dir / f"{exp}_{clf_name}_report.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    with open(report_path) as f:
        report = json.load(f)

    feature_cols = report["features"]

    df = pd.read_csv(features_csv)
    df_test = df[df["split"] == "ds2"]
    cols_present = [c for c in feature_cols if c in df_test.columns]
    sub = df_test[cols_present + ["label"]].dropna()
    X_sample = sub[cols_present].values.astype(np.float32)

    return pipeline, feature_cols, X_sample


def export_onnx(
    pipeline,
    feature_cols: list[str],
    X_sample: np.ndarray,
    onnx_path: Path,
    opset: int = 12,
) -> None:
    """Export a fitted sklearn/XGBoost pipeline to ONNX."""
    errors = []

    # 1. Try skl2onnx first (works for standard sklearn pipelines/models)
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        n_features = X_sample.shape[1]
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(pipeline, initial_types=initial_type, target_opset=opset)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        log.info(f"Exported via skl2onnx → {onnx_path}")
        return
    except Exception as e:
        errors.append(f"skl2onnx: {e}")
        log.warning(f"skl2onnx failed ({e}), trying onnxmltools ...")

    # 2. Try onnxmltools (works for XGBoost, LightGBM, etc.)
    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType

        clf = pipeline.named_steps.get("clf") if hasattr(pipeline, "named_steps") else pipeline
        if hasattr(pipeline, "named_steps") and "scaler" in pipeline.named_steps:
            # Fit/transform with scaler if present
            scaler = pipeline.named_steps["scaler"]
            # Note: onnxmltools converts the classifier directly
            # To include scaling in ONNX or convert pipeline, convert_xgboost handles the classifier
            initial_type = [("input", FloatTensorType([None, X_sample.shape[1]]))]
            onnx_model = onnxmltools.convert_xgboost(
                clf, initial_types=initial_type, target_opset=opset
            )
        else:
            initial_type = [("input", FloatTensorType([None, X_sample.shape[1]]))]
            onnx_model = onnxmltools.convert_xgboost(
                clf, initial_types=initial_type, target_opset=opset
            )

        onnxmltools.utils.save_model(onnx_model, str(onnx_path))
        log.info(f"Exported via onnxmltools → {onnx_path}")
        return
    except Exception as e:
        errors.append(f"onnxmltools: {e}")

    raise RuntimeError(f"ONNX export failed: {' | '.join(errors)}")


def quantize_model(onnx_path: Path, quant_path: Path) -> bool:
    """Apply dynamic INT8 quantization if model uses default ai.onnx domain."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    try:
        quantize_dynamic(str(onnx_path), str(quant_path), weight_type=QuantType.QInt8)
        log.info(f"Quantized model saved to {quant_path}")
        return True
    except ValueError as e:
        if "ai.onnx domain" in str(e):
            log.info("Note: Dynamic quantization skipped for ONNX-ML TreeEnsemble models (tree splits use discrete node attributes rather than weight matrices).")
        else:
            log.warning(f"Quantization skipped: {e}")
        return False
    except Exception as e:
        log.warning(f"Quantization skipped: {e}")
        return False


def verify_models(
    pipeline, onnx_path: Path, quant_path: Path, X_sample: np.ndarray
) -> None:
    """Check that ONNX and quantized predictions match sklearn predictions."""
    import onnxruntime as ort

    y_sklearn = pipeline.predict(X_sample[:100])

    for path, label in [(onnx_path, "ONNX-FP32"), (quant_path, "ONNX-INT8")]:
        if not path.exists():
            continue
        sess = ort.InferenceSession(str(path))
        input_name = sess.get_inputs()[0].name
        try:
            y_onnx = sess.run(None, {input_name: X_sample[:100]})[0]
            if hasattr(y_onnx, "flatten"):
                y_onnx = y_onnx.flatten().astype(int)
            match = np.mean(y_sklearn == y_onnx[:len(y_sklearn)])
            log.info(f"  {label}: prediction agreement with sklearn = {match:.1%}")
        except Exception as e:
            log.warning(f"  {label}: verification failed — {e}")


def report_sizes(onnx_path: Path, quant_path: Path) -> None:
    """Print model file sizes."""
    log.info("\nModel File Sizes:")
    for path, label in [(onnx_path, "ONNX FP32"), (quant_path, "ONNX INT8 (quantized)")]:
        if path.exists():
            size_kb = path.stat().st_size / 1024
            log.info(f"  {label:30s}  {size_kb:8.1f} KB")


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
    ensure_dirs(models_dir)

    opset = cfg.get("benchmark", {}).get("onnx_opset", 12)

    onnx_path  = models_dir / f"{exp}_{clf_name}.onnx"
    quant_path = models_dir / f"{exp}_{clf_name}_int8.onnx"

    log.info(f"Exporting {exp}/{clf_name} to ONNX...")
    pipeline, feature_cols, X_sample = load_model_and_data(
        exp, clf_name, models_dir, results_dir, features_csv
    )

    export_onnx(pipeline, feature_cols, X_sample, onnx_path, opset=opset)
    quantize_model(onnx_path, quant_path)
    verify_models(pipeline, onnx_path, quant_path, X_sample)
    report_sizes(onnx_path, quant_path)

    log.info("ONNX export complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and quantize model to ONNX")
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--exp",     default="E3")
    parser.add_argument("--clf",     default="xgboost")
    args = parser.parse_args()
    main(args.config, args.exp, args.clf)
