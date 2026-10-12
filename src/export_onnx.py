"""
ONNX Export
============
Exports a trained model (E3/XGBoost by default) to ONNX.

Steps:
  1. Load the trained pipeline (pickle)
  2. Convert to ONNX (onnxmltools for XGBoost, skl2onnx for sklearn models)
  3. Verify ONNX predictions against the trained model on all DS2 beats
  4. Report file size

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
from sklearn.metrics import recall_score

sys.path.insert(0, str(Path(__file__).parent))
from aami_mappings import EVAL_CLASSES, EXCLUDED_CLASSES, AAMI_TO_INT
from utils import get_logger, load_config, ensure_dirs, project_root

log = get_logger("export_onnx")

EVAL_LABELS = [AAMI_TO_INT[c] for c in EVAL_CLASSES]
EXCLUDED_LABELS = [AAMI_TO_INT[c] for c in EXCLUDED_CLASSES]


def load_model_and_data(
    exp: str, clf_name: str, models_dir: Path, results_dir: Path, features_csv: Path
):
    """Load the trained pipeline and the full DS2 test set."""
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
    sub = sub[~sub["label"].isin(EXCLUDED_LABELS)]
    X_test = sub[cols_present].values.astype(np.float32)
    y_test = sub["label"].values.astype(int)

    return pipeline, feature_cols, X_test, y_test


def export_onnx(pipeline, n_features: int, onnx_path: Path, opset: int = 12) -> None:
    """Export a fitted pipeline to ONNX without dropping any step."""
    steps = dict(pipeline.named_steps) if hasattr(pipeline, "named_steps") else {"clf": pipeline}
    clf = steps.get("clf", pipeline)

    is_xgb = type(clf).__name__ == "XGBClassifier"

    if is_xgb:
        # onnxmltools converts only the classifier
        extra = [name for name in steps if name != "clf"]
        if extra:
            raise RuntimeError(
                f"Pipeline has steps {extra} before XGBoost; onnxmltools would drop them. "
                "Retrain with train.py."
            )
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType

        initial_type = [("input", FloatTensorType([None, n_features]))]
        onnx_model = onnxmltools.convert_xgboost(clf, initial_types=initial_type, target_opset=opset)
        onnxmltools.utils.save_model(onnx_model, str(onnx_path))
        log.info(f"Exported XGBoost via onnxmltools → {onnx_path}")
        return

    # SMOTE is only used during training
    if hasattr(pipeline, "steps"):
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([(n, st) for n, st in pipeline.steps if not hasattr(st, "fit_resample")])

    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("input", FloatTensorType([None, n_features]))]
    if hasattr(clf, "predict_proba"):
        options = {id(clf): {"zipmap": False}}
    else:
        options = {type(clf): {}}
    onnx_model = convert_sklearn(
        pipeline, initial_types=initial_type,
        target_opset={"": opset, "ai.onnx.ml": 1}, options=options,
    )
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    log.info(f"Exported via skl2onnx → {onnx_path}")


def verify_model(pipeline, onnx_path: Path, X: np.ndarray, y: np.ndarray) -> dict:
    """Compare ONNX and trained-model predictions on every test beat."""
    import onnxruntime as ort

    y_ref = pipeline.predict(X)

    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    y_onnx = np.asarray(sess.run(None, {input_name: X})[0]).flatten().astype(int)

    agreement = float(np.mean(y_ref == y_onnx))
    n_disagree = int(np.sum(y_ref != y_onnx))

    sens_ref  = recall_score(y, y_ref,  labels=EVAL_LABELS, average=None, zero_division=0)
    sens_onnx = recall_score(y, y_onnx, labels=EVAL_LABELS, average=None, zero_division=0)

    log.info(f"Prediction agreement (ONNX vs trained model), all {len(X)} DS2 beats: "
             f"{agreement:.4%} ({n_disagree} beats differ)")
    for cls, a, b in zip(EVAL_CLASSES, sens_ref, sens_onnx):
        log.info(f"  Sensitivity {cls}: trained={a:.4f}  ONNX={b:.4f}")

    if agreement < 0.999:
        log.warning("ONNX predictions differ from the trained model on more than 0.1% of beats")

    return {
        "n_beats": int(len(X)),
        "agreement": agreement,
        "n_disagree": n_disagree,
        "sensitivity_trained": dict(zip(EVAL_CLASSES, map(float, sens_ref))),
        "sensitivity_onnx": dict(zip(EVAL_CLASSES, map(float, sens_onnx))),
    }


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
    ensure_dirs(models_dir, results_dir)

    opset = cfg.get("benchmark", {}).get("onnx_opset", 12)
    onnx_path = models_dir / f"{exp}_{clf_name}.onnx"

    log.info(f"Exporting {exp}/{clf_name} to ONNX...")
    pipeline, feature_cols, X_test, y_test = load_model_and_data(
        exp, clf_name, models_dir, results_dir, features_csv
    )

    export_onnx(pipeline, X_test.shape[1], onnx_path, opset=opset)
    verification = verify_model(pipeline, onnx_path, X_test, y_test)

    size_kb = onnx_path.stat().st_size / 1024
    log.info(f"ONNX file size: {size_kb:.1f} KB")
    verification["onnx_file_size_kb"] = size_kb

    out = results_dir / f"onnx_verification_{exp}_{clf_name}.json"
    with open(out, "w") as f:
        json.dump(verification, f, indent=2)
    log.info(f"Verification saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a trained model to ONNX and verify it")
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--exp",     default="E3")
    parser.add_argument("--clf",     default="xgboost")
    args = parser.parse_args()
    main(args.config, args.exp, args.clf)
