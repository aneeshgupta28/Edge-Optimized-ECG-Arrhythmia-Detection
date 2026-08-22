"""
Evaluation & Visualisation
===========================
Loads experiment results and produces:
  1. Confusion matrices (5×5 heatmap per best model)
  2. Per-class sensitivity / PPV bar chart (ablation)
  3. Feature importance chart (for tree models)
  4. Ablation comparison (E1 vs E2 vs E3 macro-F1)
  5. Model size vs accuracy scatter plot

Usage
-----
    python src/evaluate.py                   # uses configs/default.yaml
    python src/evaluate.py --config <path>
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from aami_mappings import AAMI_CLASSES
from utils import get_logger, load_config, ensure_dirs, project_root

log = get_logger("evaluate")

# Plot style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
PALETTE = sns.color_palette("coolwarm", n_colors=5)


# Confusion matrix

def plot_confusion_matrix(
    report_path: Path,
    features_csv: Path,
    model_path: Path,
    out_dir: Path,
    split: str = "ds2",
) -> None:
    """Load a saved model, run predictions on the test split, plot confusion matrix."""
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    df = pd.read_csv(features_csv)
    df_test = df[df["split"] == split]

    # load feature names from report
    with open(report_path) as f:
        report = json.load(f)
    feature_cols = report["features"]

    cols_present = [c for c in feature_cols if c in df_test.columns]
    sub = df_test[cols_present + ["label"]].dropna()
    X = sub[cols_present].values.astype(np.float32)
    y_true = sub["label"].values.astype(int)

    y_pred = pipeline.predict(X)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=AAMI_CLASSES,
        yticklabels=AAMI_CLASSES,
        ax=ax, linewidths=0.5,
    )
    exp  = report["experiment"]
    clf  = report["classifier"]
    mf1  = report["macro_f1"]
    ax.set_title(f"Confusion Matrix — {exp} / {clf}\nMacro-F1={mf1:.3f}", fontsize=12, pad=12)
    ax.set_xlabel("Predicted AAMI Class")
    ax.set_ylabel("True AAMI Class")
    plt.tight_layout()

    out_path = out_dir / f"cm_{exp}_{clf}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved plot to {out_path}")


# Per-class sensitivity chart

def plot_per_class_metrics(summary_csv: Path, out_dir: Path) -> None:
    """Bar chart of per-class recall for each experiment × classifier."""
    df = pd.read_csv(summary_csv)

    for exp in df["experiment"].unique():
        sub = df[df["experiment"] == exp]

        recall_cols = [f"recall_{c}" for c in AAMI_CLASSES]
        melted = sub.melt(
            id_vars=["classifier"],
            value_vars=recall_cols,
            var_name="metric",
            value_name="recall",
        )
        melted["class"] = melted["metric"].str.replace("recall_", "")

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(AAMI_CLASSES))
        clf_names = sub["classifier"].tolist()
        n = len(clf_names)
        width = 0.8 / n

        for i, clf in enumerate(clf_names):
            vals = [sub[sub["classifier"] == clf][f"recall_{c}"].values[0] for c in AAMI_CLASSES]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=clf, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(AAMI_CLASSES)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Sensitivity (Recall)")
        ax.set_title(f"Per-Class Sensitivity: Experiment {exp}")
        ax.legend(frameon=False)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        plt.tight_layout()

        out_path = out_dir / f"sensitivity_{exp}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Saved plot to {out_path}")


# Ablation chart

def plot_ablation(summary_csv: Path, out_dir: Path) -> None:
    """Compare macro-F1 across experiments (E1 vs E2 vs E3 vs E4)."""
    df = pd.read_csv(summary_csv)
    pivot = df.pivot(index="classifier", columns="experiment", values="macro_f1")

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, colormap="coolwarm", alpha=0.85, edgecolor="none")
    ax.set_ylabel("Macro F1 Score")
    ax.set_title("Ablation Study: Feature Group vs Classifier")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Experiment", frameon=False, loc="lower right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out_path = out_dir / "ablation_macro_f1.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved plot to {out_path}")


# Feature importance

def plot_feature_importance(
    model_path: Path,
    feature_cols: list[str],
    out_dir: Path,
    top_n: int = 15,
) -> None:
    """Bar chart of feature importance for tree-based models."""
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    # Drill into pipeline to find the classifier
    clf = pipeline.named_steps.get("clf") or pipeline[-1]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        log.warning(f"Model {model_path.name} has no feature_importances_ - skipping")
        return

    # Account for any feature scaling that might reduce feature count
    n_feats = min(len(importances), len(feature_cols))
    names   = feature_cols[:n_feats]
    imps    = importances[:n_feats]

    sorted_idx = np.argsort(imps)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [names[i] for i in sorted_idx],
        [imps[i] for i in sorted_idx],
        color=sns.color_palette("mako", n_colors=top_n),
        edgecolor="none",
    )
    clf_label = model_path.stem
    ax.set_title(f"Feature Importance (top {top_n}): {clf_label}")
    ax.set_xlabel("Importance")
    plt.tight_layout()

    out_path = out_dir / f"feature_importance_{model_path.stem}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved plot to {out_path}")


# Model size vs accuracy

def plot_size_vs_accuracy(summary_csv: Path, models_dir: Path, out_dir: Path) -> None:
    """Scatter of model file size vs macro-F1 (E3 experiment)."""
    df = pd.read_csv(summary_csv)
    df_e3 = df[df["experiment"] == "E3"].copy()

    sizes = []
    for _, row in df_e3.iterrows():
        pkl_path = models_dir / f"E3_{row['classifier']}.pkl"
        size_kb = pkl_path.stat().st_size / 1024 if pkl_path.exists() else 0.0
        sizes.append(size_kb)
    df_e3["size_kb"] = sizes

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = sns.color_palette("tab10", n_colors=len(df_e3))
    for i, (_, row) in enumerate(df_e3.iterrows()):
        ax.scatter(row["size_kb"], row["macro_f1"], s=150, color=colors[i],
                   label=row["classifier"], zorder=3)
        ax.annotate(row["classifier"], (row["size_kb"], row["macro_f1"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_xlabel("Model Size (KB)")
    ax.set_ylabel("Macro F1 (DS2)")
    ax.set_title("Model Size vs. Accuracy (E3: All Features, DS2 test)")
    ax.legend(frameon=False)
    plt.tight_layout()

    out_path = out_dir / "size_vs_accuracy.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved plot to {out_path}")


# Main

def main(config_path: str = "configs/default.yaml") -> None:
    cfg = load_config(config_path)
    root = project_root()

    features_csv = root / cfg["data"]["features_csv"]
    results_dir  = root / cfg["results"]["dir"]
    models_dir   = root / cfg["results"]["models_dir"]
    plots_dir    = results_dir / "plots"
    ensure_dirs(plots_dir)

    summary_csv = results_dir / "experiment_summary.csv"
    if not summary_csv.exists():
        log.error(f"No summary CSV at {summary_csv}. Run train.py first.")
        sys.exit(1)

    df_summary = pd.read_csv(summary_csv)

    # 1. Confusion matrices for best model per experiment
    for _, row in df_summary.iterrows():
        exp  = row["experiment"]
        clf  = row["classifier"]
        report_path = results_dir / f"{exp}_{clf}_report.json"
        model_path  = models_dir  / f"{exp}_{clf}.pkl"
        if report_path.exists() and model_path.exists() and features_csv.exists():
            try:
                plot_confusion_matrix(report_path, features_csv, model_path, plots_dir)
            except Exception as e:
                log.warning(f"  CM failed for {exp}/{clf}: {e}")

    # 2. Per-class sensitivity charts (all experiments)
    plot_per_class_metrics(summary_csv, plots_dir)

    # 3. Ablation comparison
    plot_ablation(summary_csv, plots_dir)

    # 4. Feature importance (RF and XGBoost, E3)
    for clf_name in ["random_forest", "xgboost"]:
        model_path = models_dir / f"E3_{clf_name}.pkl"
        report_path = results_dir / f"E3_{clf_name}_report.json"
        if model_path.exists() and report_path.exists():
            with open(report_path) as f:
                rpt = json.load(f)
            try:
                plot_feature_importance(model_path, rpt["features"], plots_dir)
            except Exception as e:
                log.warning(f"  FI failed for {clf_name}: {e}")

    # 5. Size vs accuracy
    if features_csv.exists():
        plot_size_vs_accuracy(summary_csv, models_dir, plots_dir)

    log.info(f"All plots saved to {plots_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
