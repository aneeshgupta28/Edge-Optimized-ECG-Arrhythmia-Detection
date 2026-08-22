"""
Training Pipeline
==================
Trains 4 classifiers (LR, LinearSVM, RF, XGBoost) on DS1, evaluated on DS2.

Experiment matrix:
  E1 — RR-only features
  E2 — Morphological features only
  E3 — All features (full model)
  E4 — Top-K features by mutual information (feature selection)

For each experiment × classifier:
  - 5-fold cross-validation on DS1 (hyperparameter selection)
  - Final training on all of DS1 with best params
  - Evaluation on DS2 (inter-patient test set)
  - Save model and results to results/

Usage
-----
    python src/train.py                   # uses configs/default.yaml
    python src/train.py --config <path>   # custom config
    python src/train.py --experiment E3   # run one experiment only
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from aami_mappings import AAMI_CLASSES, INT_TO_AAMI
from utils import get_logger, load_config, set_seed, ensure_dirs, project_root

warnings.filterwarnings("ignore", category=FutureWarning)
log = get_logger("train")

# Optional SMOTE
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    _HAS_SMOTE = True
except ImportError:
    _HAS_SMOTE = False
    log.warning("imbalanced-learn not installed — SMOTE disabled. `pip install imbalanced-learn`")


# Feature groups

RR_FEATURES = ["pre_rr", "post_rr", "local_avg_rr", "pre_rr_ratio", "post_rr_ratio"]

MORPH_FEATURES = [
    "qrs_duration", "qrs_amplitude", "r_amplitude",
    "t_wave_amplitude", "st_level",
    "wavelet_energy_l1", "wavelet_energy_l2",
    "wavelet_energy_l3", "wavelet_energy_l4",
    "dct_0", "dct_1", "dct_2", "dct_3", "dct_4", "dct_5",
]

FREQ_FEATURES = ["vlf_power", "lf_power", "hf_power", "lf_hf_ratio"]

ALL_FEATURES = RR_FEATURES + MORPH_FEATURES + FREQ_FEATURES

EXPERIMENT_FEATURE_MAP = {
    "E1": RR_FEATURES,
    "E2": MORPH_FEATURES,
    "E3": ALL_FEATURES,
    # E4 is handled dynamically with mutual information
}


# Helpers

def load_splits(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load features CSV and return (ds1, ds2) DataFrames."""
    df = pd.read_csv(csv_path)
    ds1 = df[df["split"] == "ds1"].copy()
    ds2 = df[df["split"] == "ds2"].copy()
    log.info(f"Loaded {len(ds1)} DS1 beats, {len(ds2)} DS2 beats")
    return ds1, ds2


def get_Xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and label array, dropping NaN rows."""
    cols_present = [c for c in feature_cols if c in df.columns]
    sub = df[cols_present + ["label"]].dropna()
    X = sub[cols_present].values.astype(np.float32)
    y = sub["label"].values.astype(int)
    return X, y


def top_k_features_by_mi(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    k: int = 15,
    seed: int = 42,
) -> list[str]:
    """Return top-k feature names ranked by mutual information."""
    mi = mutual_info_classif(X_train, y_train, random_state=seed)
    ranked = sorted(zip(feature_names, mi), key=lambda x: -x[1])
    selected = [name for name, _ in ranked[:k]]
    log.info(f"Top-{k} features (MI): {selected}")
    return selected


def build_classifier(name: str, cfg: dict, seed: int) -> object:
    """Instantiate classifier with default params from config."""
    if name == "logistic_regression":
        lr_cfg = cfg.get("logistic_regression", {})
        return LogisticRegression(
            C=1.0,
            solver=lr_cfg.get("solver", "lbfgs"),
            max_iter=int(lr_cfg.get("max_iter", 1000)),
            class_weight="balanced",
            random_state=seed,
        )
    elif name == "svm":
        # LinearSVC: O(n) — orders of magnitude faster than RBF-SVM on 50k+ samples.
        # Closest to de Chazal 2004's linear classifier baseline.
        svm_cfg = cfg.get("svm", {})
        return LinearSVC(
            C=1.0,
            max_iter=int(svm_cfg.get("max_iter", 2000)),
            class_weight="balanced",
            dual="auto",
            random_state=seed,
        )
    elif name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    elif name == "xgboost":
        xgb_cfg = cfg.get("xgboost", {})
        return XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")


def get_param_grid(name: str, cfg: dict) -> dict:
    """Return hyperparameter grid for GridSearchCV."""
    if name == "logistic_regression":
        lr_cfg = cfg.get("logistic_regression", {})
        return {"clf__C": lr_cfg.get("C", [0.01, 0.1, 1.0, 10.0])}
    elif name == "svm":
        svm_cfg = cfg.get("svm", {})
        return {
            "clf__C": svm_cfg.get("C", [0.1, 1.0, 10.0]),
        }
    elif name == "random_forest":
        rf_cfg = cfg.get("random_forest", {})
        return {
            "clf__n_estimators": rf_cfg.get("n_estimators", [100, 300]),
            "clf__max_depth": [d if d != "null" else None
                               for d in (rf_cfg.get("max_depth", [10, 20, None]))],
        }
    elif name == "xgboost":
        xgb_cfg = cfg.get("xgboost", {})
        return {
            "clf__n_estimators": xgb_cfg.get("n_estimators", [100, 300]),
            "clf__max_depth": xgb_cfg.get("max_depth", [4, 6]),
            "clf__learning_rate": xgb_cfg.get("learning_rate", [0.05, 0.1]),
        }
    return {}


# Training logic

def train_one(
    clf_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    train_cfg: dict,
    seed: int,
) -> dict:
    """Train one classifier with cross-val on DS1, evaluate on DS2.

    Returns a results dict with metrics and the fitted model.
    """
    log.info(f"  Training {clf_name} ...")
    n_folds = int(train_cfg.get("cv_folds", 5))
    # Only apply SMOTE to linear models; tree models handle imbalance natively and SMOTE slows them down drastically
    use_smote = train_cfg.get("use_smote", True) and _HAS_SMOTE and (clf_name in ["logistic_regression", "svm"])

    clf = build_classifier(clf_name, train_cfg, seed)
    param_grid = get_param_grid(clf_name, train_cfg)

    # Build pipeline: scale → (SMOTE →) classify
    scaler = StandardScaler()

    if use_smote:
        smote = SMOTE(random_state=seed, k_neighbors=3)
        pipeline = ImbPipeline([
            ("scaler", scaler),
            ("smote",  smote),
            ("clf",    clf),
        ])
    else:
        pipeline = Pipeline([
            ("scaler", scaler),
            ("clf",    clf),
        ])

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    t0 = perf_counter()
    if param_grid:
        grid = GridSearchCV(
            pipeline, param_grid,
            cv=cv, scoring="f1_macro",
            n_jobs=-1, refit=True, verbose=0,
        )
        grid.fit(X_train, y_train)
        best_pipeline = grid.best_estimator_
        best_params = grid.best_params_
        cv_score = grid.best_score_
    else:
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline
        best_params = {}
        cv_score = 0.0

    train_time = perf_counter() - t0

    # Evaluate on DS2 (inter-patient test)
    y_pred = best_pipeline.predict(X_test)

    acc      = accuracy_score(y_test, y_pred)
    bal_acc  = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report   = classification_report(
        y_test, y_pred,
        target_names=AAMI_CLASSES,
        zero_division=0,
        output_dict=True,
    )

    log.info(
        f"    {clf_name}: acc={acc:.3f}  bal_acc={bal_acc:.3f}  "
        f"macro_f1={macro_f1:.3f}  cv_f1={cv_score:.3f}  "
        f"train_time={train_time:.1f}s"
    )

    return {
        "clf_name":    clf_name,
        "accuracy":    acc,
        "bal_accuracy": bal_acc,
        "macro_f1":    macro_f1,
        "cv_f1":       cv_score,
        "best_params": best_params,
        "train_time":  train_time,
        "report":      report,
        "pipeline":    best_pipeline,
    }


# Experiment runner

def run_experiment(
    exp_name: str,
    feature_cols: list[str],
    ds1: pd.DataFrame,
    ds2: pd.DataFrame,
    clf_names: list[str],
    train_cfg: dict,
    results_dir: Path,
    models_dir: Path,
    seed: int,
) -> pd.DataFrame:
    """Run one experiment (feature set) across all classifiers."""
    log.info(f"\n{'='*60}")
    log.info(f"EXPERIMENT {exp_name} - {len(feature_cols)} features: {feature_cols}")
    log.info(f"{'='*60}")

    X_train, y_train = get_Xy(ds1, feature_cols)
    X_test,  y_test  = get_Xy(ds2, feature_cols)

    log.info(f"DS1: {X_train.shape}   DS2: {X_test.shape}")
    log.info(f"Class dist (train): { {INT_TO_AAMI[i]: int(np.sum(y_train==i)) for i in range(5)} }")
    log.info(f"Class dist (test):  { {INT_TO_AAMI[i]: int(np.sum(y_test ==i)) for i in range(5)} }")

    summary_rows = []

    for clf_name in clf_names:
        result = train_one(
            clf_name, X_train, y_train, X_test, y_test,
            feature_cols, train_cfg, seed,
        )

        # Save model
        model_path = models_dir / f"{exp_name}_{clf_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(result["pipeline"], f)

        # Save per-class report
        report_path = results_dir / f"{exp_name}_{clf_name}_report.json"
        with open(report_path, "w") as f:
            json.dump(
                {
                    "experiment": exp_name,
                    "classifier": clf_name,
                    "features": feature_cols,
                    "best_params": result["best_params"],
                    "accuracy": result["accuracy"],
                    "bal_accuracy": result["bal_accuracy"],
                    "macro_f1": result["macro_f1"],
                    "cv_f1": result["cv_f1"],
                    "classification_report": result["report"],
                },
                f, indent=2,
            )

        summary_rows.append({
            "experiment":  exp_name,
            "classifier":  clf_name,
            "accuracy":    result["accuracy"],
            "bal_accuracy": result["bal_accuracy"],
            "macro_f1":    result["macro_f1"],
            "cv_f1":       result["cv_f1"],
            "n_features":  len(feature_cols),
            "train_time_s": result["train_time"],
            # Per-class sensitivity (recall) for N, S, V, F, Q
            **{f"recall_{cls}": result["report"].get(cls, {}).get("recall", float("nan"))
               for cls in AAMI_CLASSES},
            **{f"precision_{cls}": result["report"].get(cls, {}).get("precision", float("nan"))
               for cls in AAMI_CLASSES},
        })

    return pd.DataFrame(summary_rows)


# Main

def main(config_path: str = "configs/default.yaml", experiment: str | None = None):
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    root = project_root()
    features_csv = root / cfg["data"]["features_csv"]
    results_dir  = root / cfg["results"]["dir"]
    models_dir   = root / cfg["results"]["models_dir"]
    ensure_dirs(results_dir, models_dir)

    if not features_csv.exists():
        log.error(f"Features CSV not found: {features_csv}")
        log.error("Run: python src/dataset.py  first")
        sys.exit(1)

    ds1, ds2 = load_splits(features_csv)
    train_cfg = cfg.get("training", {})

    clf_names = ["logistic_regression", "svm", "random_forest", "xgboost"]

    experiments_to_run = list(EXPERIMENT_FEATURE_MAP.keys())
    if experiment:
        experiments_to_run = [experiment.upper()]

    all_summaries = []

    for exp_name in experiments_to_run:
        if exp_name == "E4":
            # Top-K features by mutual information
            X_tr, y_tr = get_Xy(ds1, ALL_FEATURES)
            cols_present = [c for c in ALL_FEATURES if c in ds1.columns]
            top_k = top_k_features_by_mi(X_tr, y_tr, cols_present, k=15, seed=seed)
            feature_cols = top_k
        else:
            feature_cols = EXPERIMENT_FEATURE_MAP[exp_name]

        summary = run_experiment(
            exp_name, feature_cols, ds1, ds2,
            clf_names, train_cfg, results_dir, models_dir, seed,
        )
        all_summaries.append(summary)

    # Run E4 if in full mode
    if experiment is None or experiment.upper() == "E4":
        X_tr, y_tr = get_Xy(ds1, ALL_FEATURES)
        cols_present = [c for c in ALL_FEATURES if c in ds1.columns]
        top_k = top_k_features_by_mi(X_tr, y_tr, cols_present, k=15, seed=seed)
        summary_e4 = run_experiment(
            "E4", top_k, ds1, ds2,
            clf_names, train_cfg, results_dir, models_dir, seed,
        )
        all_summaries.append(summary_e4)

    # Save combined summary
    df_summary = pd.concat(all_summaries, ignore_index=True)
    summary_path = results_dir / "experiment_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    log.info(f"\nAll experiments done. Summary saved to {summary_path}")
    log.info("\n" + df_summary[["experiment", "classifier", "macro_f1", "recall_S", "recall_V"]].to_string(index=False))

    return df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ECG arrhythmia classifiers")
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--experiment", default=None, help="Run one experiment: E1/E2/E3/E4")
    args = parser.parse_args()
    main(args.config, args.experiment)
