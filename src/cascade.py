"""
Two-Stage Cascade: XGBoost -> 1D CNN
====================================
Stage 1 is the existing E3 XGBoost on the 24 handcrafted features. It is
cheap and handles N and V well, but catches only ~10% of S beats on DS2.
Stage 2 is a compact 1D CNN on the raw beat window (+ RR inputs). Only beats
that stage 1 is not sure about are escalated to it.

Two escalation rules are swept: send a beat to the CNN when XGBoost's
P(N) < tau, or when its max class probability < tau. The rule and tau are
picked together on DS1. XGBoost is often confidently wrong on S beats, which
limits how much any confidence-based trigger can recover.

Protocol
--------
  - The escalation rule and tau are chosen on DS1 only, using out-of-fold
    predictions from 5-fold CV grouped by record. The CNN recipe is fixed
    up front (CNN_RECIPE) and was not tuned.
  - DS2 is touched once, at the end, with the frozen models and frozen tau.
  - Q beats are excluded, as in train.py.

Steps
-----
  1. XGBoost baseline (same params as E3) and a class-weighted XGBoost
  2. 1D CNN alone
  3. Cascade sweep on DS1 OOF -> pick tau (max macro-F1) -> apply to DS2
  4. ONNX export, INT8 static quantisation, size and latency
  5. Arithmetic energy estimate per beat (Horowitz, ISSCC 2014, 45 nm)

Usage
-----
    python src/beat_windows.py --post_samples 110   # once, writes data/beat_windows.npz
    python src/cascade.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))

from aami_mappings import EVAL_CLASSES, AAMI_TO_INT
from cnn_model import BeatCNN, RR_INPUTS, prepare_beats, count_macs_params
from train import ALL_FEATURES
from utils import get_logger, load_config, set_seed, ensure_dirs, project_root

log = get_logger("cascade")

LABELS = [AAMI_TO_INT[c] for c in EVAL_CLASSES]   # N, S, V, F -> 0..3
S, N = AAMI_TO_INT["S"], AAMI_TO_INT["N"]

# Horowitz, "Computing's energy problem", ISSCC 2014 (45 nm, 0.9 V)
PJ = {
    "fp32_mult": 3.7, "fp32_add": 0.9,
    "int8_mult": 0.2, "int8_add": 0.03,
}
PJ_FP32_MAC = PJ["fp32_mult"] + PJ["fp32_add"]
PJ_INT8_MAC = PJ["int8_mult"] + PJ["int8_add"]

XGB_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", n_jobs=-1,
)

CNN_RECIPE = dict(epochs=20, batch_size=256, lr=2e-3, weight_decay=1e-4, class_weight_power=0.5)


# Data

def load_data(root: Path, cfg: dict) -> dict:
    """Join the feature table with raw windows on (record, beat_index)."""
    df = pd.read_csv(root / cfg["data"]["features_csv"])
    df["record"] = df["record"].astype(str)
    df = df.dropna(subset=ALL_FEATURES)
    df = df[df["label"].isin(LABELS)]

    npz = np.load(root / "data" / "beat_windows.npz")
    key = pd.DataFrame({
        "record": npz["record"].astype(str),
        "beat_index": npz["beat_index"],
        "row": np.arange(len(npz["beat_index"])),
        "label_w": npz["label"],
    })
    df = df.merge(key, on=["record", "beat_index"], how="inner")
    assert (df["label"] == df["label_w"]).all(), "label mismatch between features and windows"

    windows = npz["windows"][df["row"].values]
    out = {"win_len": int(windows.shape[1])}
    for split in ("ds1", "ds2"):
        m = (df["split"] == split).values
        out[split] = dict(
            X=df.loc[m, ALL_FEATURES].values.astype(np.float32),
            rr=df.loc[m, RR_INPUTS].values.astype(np.float32),
            beats=prepare_beats(windows[m]),
            y=df.loc[m, "label"].values.astype(int),
            groups=df.loc[m, "record"].values,
        )
        counts = {c: int((out[split]["y"] == AAMI_TO_INT[c]).sum()) for c in EVAL_CLASSES}
        log.info(f"{split.upper()}: {m.sum()} beats {counts}")
    return out


# Metrics

def summarise(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p, r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    res = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
    }
    for i, c in enumerate(EVAL_CLASSES):
        res[f"sens_{c}"] = float(r[i])
        res[f"ppv_{c}"] = float(p[i])
    return res


def fmt(res: dict) -> str:
    return (f"acc={res['accuracy']:.3f} macroF1={res['macro_f1']:.3f} "
            f"S sens/ppv={res['sens_S']:.3f}/{res['ppv_S']:.3f} "
            f"V sens/ppv={res['sens_V']:.3f}/{res['ppv_V']:.3f}")


# Stage 1: XGBoost

def fit_xgb(X, y, seed, weighted=False) -> XGBClassifier:
    clf = XGBClassifier(random_state=seed, **XGB_PARAMS)
    w = compute_sample_weight("balanced", y) if weighted else None
    clf.fit(X, y, sample_weight=w)
    return clf


# Stage 2: CNN

def fit_cnn(beats, rr, y, seed) -> tuple[BeatCNN, np.ndarray, np.ndarray]:
    """Train the CNN. Returns model and the RR standardisation (mean, std)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu, sd = rr.mean(0), rr.std(0) + 1e-6
    rr_n = (rr - mu) / sd

    counts = np.bincount(y, minlength=len(LABELS)).astype(float)
    cw = (counts.sum() / np.maximum(counts, 1)) ** CNN_RECIPE["class_weight_power"]
    cw = cw / cw.mean()

    model = BeatCNN(n_classes=len(LABELS))
    opt = torch.optim.AdamW(model.parameters(), lr=CNN_RECIPE["lr"],
                            weight_decay=CNN_RECIPE["weight_decay"])
    epochs, bs = CNN_RECIPE["epochs"], CNN_RECIPE["batch_size"]
    steps = epochs * int(np.ceil(len(y) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CNN_RECIPE["lr"], total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32))

    xb_all = torch.from_numpy(beats)
    xr_all = torch.from_numpy(rr_n.astype(np.float32))
    y_all = torch.from_numpy(y)

    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(y))
        total = 0.0
        for i in range(0, len(y), bs):
            idx = perm[i:i + bs]
            xb = xb_all[idx]
            # light augmentation: amplitude scaling, baseline offset, noise
            xb = xb * (1 + 0.1 * torch.randn(len(idx), 1, 1)) + 0.05 * torch.randn(len(idx), 1, 1)
            xb = xb + 0.02 * torch.randn_like(xb)
            loss = loss_fn(model(xb, xr_all[idx]), y_all[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        if ep == 0 or (ep + 1) % 5 == 0:
            log.info(f"    epoch {ep + 1:2d}/{epochs} loss={total / len(y):.4f}")
    model.eval()
    return model, mu, sd


def cnn_proba(model, mu, sd, beats, rr) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.from_numpy(beats), torch.from_numpy(((rr - mu) / sd).astype(np.float32)))
        return torch.softmax(logits, 1).numpy()


# Cascade

def cascade_predict(p1, p2, tau, rule="p_normal"):
    score = p1[:, N] if rule == "p_normal" else p1.max(1)
    esc = score < tau
    pred = p1.argmax(1)
    pred[esc] = p2[esc].argmax(1)
    return pred, esc


def sweep(p1, p2, y, taus, rule):
    rows = []
    for t in taus:
        pred, esc = cascade_predict(p1, p2, t, rule)
        rows.append({"rule": rule, "tau": float(t), "escalated": float(esc.mean()), **summarise(y, pred)})
    return pd.DataFrame(rows)


# Cost model

def xgb_comparisons(clf: XGBClassifier) -> float:
    """Mean number of split comparisons per beat (one root-to-leaf path per tree)."""
    trees = clf.get_booster().trees_to_dataframe()
    depth = {}
    for tid, t in trees.groupby("Tree"):
        t = t.set_index("ID")
        d = {}
        for nid in t.index:
            k, n = 0, nid
            while True:
                parent = t.index[(t["Yes"] == n) | (t["No"] == n)]
                if len(parent) == 0:
                    break
                n, k = parent[0], k + 1
            d[nid] = k
        leaves = t[t["Feature"] == "Leaf"].index
        depth[tid] = np.mean([d[l] for l in leaves])
    return float(sum(depth.values()))


def feature_extraction_ops(win_len: int, fs: float = 360.0) -> int:
    """Rough multiply-accumulate count for the 24 handcrafted features.

    db4 wavelet (4 levels, 8 taps, lo+hi filters), 6 DCT coefficients computed
    directly, morphology windows (~3 passes over the beat), and Welch PSD on a
    4 Hz resampled RR series of ~60 beats (64-pt segments, 50 % overlap,
    radix-2 FFT ~ (N/2) log2 N complex MACs = 4 real MACs each).
    """
    wav, n = 0, win_len
    for _ in range(4):
        wav += 2 * 8 * (n // 2)
        n //= 2
    dct6 = 6 * win_len
    morph = 3 * win_len
    rr_len = int(60 * 0.8 * 4)                       # ~60 beats at 0.8 s, 4 Hz
    n_seg = max(1, (rr_len - 64) // 32 + 1)
    welch = n_seg * (64 + 4 * 32 * 6) + rr_len       # window + FFT + interp
    return int(wav + dct6 + morph + welch)


# ONNX + INT8

def export_and_quantise(model, mu, sd, calib, out_dir: Path, seed: int):
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationDataReader, QuantFormat, QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    beat_len = calib["beats"].shape[-1]
    # onnxruntime writes and deletes scratch files next to the model, so
    # quantise in a temp dir and copy only the final models into out_dir
    tmp = Path(tempfile.mkdtemp(prefix="ecg_onnx_"))
    torch.onnx.export(
        model, (torch.zeros(1, 1, beat_len), torch.zeros(1, len(RR_INPUTS))), str(tmp / "fp32.onnx"),
        input_names=["beat", "rr"], output_names=["logits"],
        dynamic_axes={"beat": {0: "n"}, "rr": {0: "n"}, "logits": {0: "n"}},
        opset_version=17,
    )
    quant_pre_process(str(tmp / "fp32.onnx"), str(tmp / "fp32_pre.onnx"))

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(calib["y"]), size=min(1000, len(calib["y"])), replace=False)
    rr_n = ((calib["rr"][idx] - mu) / sd).astype(np.float32)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"beat": calib["beats"][i:i + 1], "rr": rr_n[j:j + 1]}
                            for j, i in enumerate(idx)])

        def get_next(self):
            return next(self.it, None)

    quantize_static(str(tmp / "fp32_pre.onnx"), str(tmp / "int8.onnx"), Reader(),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)

    fp32, int8 = out_dir / "beat_cnn_fp32.onnx", out_dir / "beat_cnn_int8.onnx"
    shutil.copyfile(tmp / "fp32.onnx", fp32)
    shutil.copyfile(tmp / "int8.onnx", int8)
    shutil.rmtree(tmp, ignore_errors=True)

    def session(path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    return {"fp32": (fp32, session(fp32)), "int8": (int8, session(int8))}


def ort_proba(sess, mu, sd, beats, rr) -> np.ndarray:
    rr_n = ((rr - mu) / sd).astype(np.float32)
    logits = sess.run(None, {"beat": beats, "rr": rr_n})[0]
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def latency_us(sess, beat, rr, n=5000) -> dict:
    feed = {"beat": beat, "rr": rr}
    for _ in range(200):
        sess.run(None, feed)
    t = np.empty(n)
    for i in range(n):
        t0 = perf_counter()
        sess.run(None, feed)
        t[i] = perf_counter() - t0
    t *= 1e6
    return {"mean_us": float(t.mean()), "p99_us": float(np.percentile(t, 99))}


# Main

def main(config_path: str = "configs/default.yaml"):
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    root = project_root()
    out_dir = root / cfg["results"]["dir"] / "cascade"
    model_dir = root / cfg["results"]["models_dir"]
    ensure_dirs(out_dir, model_dir)

    d = load_data(root, cfg)
    ds1, ds2 = d["ds1"], d["ds2"]
    results: dict = {"protocol": "DS1 train / DS2 test, tau chosen on DS1 out-of-fold predictions"}

    # 1. DS1 out-of-fold predictions for both stages
    cv = StratifiedGroupKFold(n_splits=int(cfg["training"].get("cv_folds", 5)),
                              shuffle=True, random_state=seed)
    # Each expensive step is cached, so an interrupted run resumes where it stopped.
    cache_dir = out_dir / "cache"
    ensure_dirs(cache_dir)

    def cached(name: str, fn) -> np.ndarray:
        path = cache_dir / f"{name}.npy"
        if path.exists():
            return np.load(path)
        arr = fn()
        np.save(path, arr)
        return arr

    oof_xgb = np.zeros((len(ds1["y"]), len(LABELS)), np.float32)
    oof_cnn = np.zeros_like(oof_xgb)
    for k, (tr, va) in enumerate(cv.split(ds1["X"], ds1["y"], ds1["groups"])):
        log.info(f"Fold {k + 1}: train {len(tr)} / val {len(va)} beats")
        oof_xgb[va] = cached(f"fold{k}_xgb", lambda: fit_xgb(
            ds1["X"][tr], ds1["y"][tr], seed).predict_proba(ds1["X"][va]))

        def cnn_fold():
            m, mu, sd = fit_cnn(ds1["beats"][tr], ds1["rr"][tr], ds1["y"][tr], seed)
            return cnn_proba(m, mu, sd, ds1["beats"][va], ds1["rr"][va])
        oof_cnn[va] = cached(f"fold{k}_cnn", cnn_fold)
    results["ds1_oof"] = {
        "xgboost": summarise(ds1["y"], oof_xgb.argmax(1)),
        "cnn": summarise(ds1["y"], oof_cnn.argmax(1)),
    }

    # 2. Final models on all of DS1
    log.info("Fitting final models on DS1")
    xgb_path, cnn_path = model_dir / "cascade_xgb.json", model_dir / "beat_cnn.pt"
    if xgb_path.exists():
        xgb = XGBClassifier()
        xgb.load_model(xgb_path)
    else:
        xgb = fit_xgb(ds1["X"], ds1["y"], seed)
        xgb.save_model(xgb_path)
    pred_w = cached("ds2_xgb_weighted", lambda: fit_xgb(
        ds1["X"], ds1["y"], seed, weighted=True).predict(ds2["X"]))

    if cnn_path.exists():
        ckpt = torch.load(cnn_path, weights_only=False)
        cnn = BeatCNN(n_classes=len(LABELS))
        cnn.load_state_dict(ckpt["state_dict"])
        cnn.eval()
        mu, sd = ckpt["rr_mean"], ckpt["rr_std"]
    else:
        cnn, mu, sd = fit_cnn(ds1["beats"], ds1["rr"], ds1["y"], seed)
        torch.save({"state_dict": cnn.state_dict(), "rr_mean": mu, "rr_std": sd,
                    "recipe": CNN_RECIPE}, cnn_path)

    p1 = xgb.predict_proba(ds2["X"])
    p2 = cnn_proba(cnn, mu, sd, ds2["beats"], ds2["rr"])
    results["ds2"] = {
        "xgboost": summarise(ds2["y"], p1.argmax(1)),
        "xgboost_class_weighted": summarise(ds2["y"], pred_w),
        "cnn_fp32": summarise(ds2["y"], p2.argmax(1)),
    }
    for k, v in results["ds2"].items():
        log.info(f"DS2 {k:24s} {fmt(v)}")

    # 3. Cascade: sweep on DS1 OOF, freeze tau, apply to DS2
    taus = np.round(np.concatenate([np.linspace(0.05, 0.95, 19), [0.97, 0.98, 0.99, 0.995, 0.999]]), 4)
    sweeps_ds1 = pd.concat([sweep(oof_xgb, oof_cnn, ds1["y"], taus, r) for r in ("p_normal", "max_prob")],
                           ignore_index=True)
    sweeps_ds2 = pd.concat([sweep(p1, p2, ds2["y"], taus, r) for r in ("p_normal", "max_prob")],
                           ignore_index=True)
    sweeps_ds1.to_csv(out_dir / "sweep_ds1_oof.csv", index=False)
    sweeps_ds2.to_csv(out_dir / "sweep_ds2.csv", index=False)

    best = sweeps_ds1.loc[sweeps_ds1["macro_f1"].idxmax()]
    rule, tau = best["rule"], float(best["tau"])
    log.info(f"Chosen on DS1 OOF: rule={rule} tau={tau} (OOF macroF1={best['macro_f1']:.3f}, "
             f"escalated={best['escalated']:.1%})")
    pred, esc = cascade_predict(p1, p2, tau, rule)
    results["cascade"] = {"rule": rule, "tau": tau, "ds1_oof": best.to_dict(),
                          "ds2": {"escalated": float(esc.mean()), **summarise(ds2["y"], pred)}}
    log.info(f"DS2 cascade (escalated {esc.mean():.1%}) {fmt(results['cascade']['ds2'])}")

    # 4. ONNX FP32 / INT8
    log.info("Exporting ONNX and quantising to INT8")
    sess = export_and_quantise(cnn, mu, sd, ds1, model_dir, seed)
    beat_len = ds1["beats"].shape[-1]
    one_beat, one_rr = ds2["beats"][:1], ((ds2["rr"][:1] - mu) / sd).astype(np.float32)
    onnx_res = {}
    for name, (path, s) in sess.items():
        p = ort_proba(s, mu, sd, ds2["beats"], ds2["rr"])
        cpred, cesc = cascade_predict(p1, p, tau, rule)
        onnx_res[name] = {
            "file_kb": path.stat().st_size / 1024,
            "cnn_alone": summarise(ds2["y"], p.argmax(1)),
            "cascade": summarise(ds2["y"], cpred),
            "latency_single_thread": latency_us(s, one_beat, one_rr),
        }
        log.info(f"{name}: {onnx_res[name]['file_kb']:.1f} KB, "
                 f"{onnx_res[name]['latency_single_thread']['mean_us']:.1f} us/beat, "
                 f"cascade {fmt(onnx_res[name]['cascade'])}")
    results["onnx"] = onnx_res

    # 5. Cost and energy per beat
    macs, params = count_macs_params(cnn, beat_len)
    comps = xgb_comparisons(xgb)
    feat_ops = feature_extraction_ops(d["win_len"])
    e_stage1 = feat_ops * PJ_FP32_MAC + comps * PJ["fp32_add"]
    e_cnn32, e_cnn8 = macs * PJ_FP32_MAC, macs * PJ_INT8_MAC
    rate = float(esc.mean())
    results["cost_per_beat"] = {
        "cnn_params": params,
        "cnn_macs": macs,
        "xgb_trees": int(xgb.get_booster().num_boosted_rounds() * len(LABELS)),
        "xgb_mean_comparisons": comps,
        "feature_extraction_macs_est": feat_ops,
        "cascade_avg_macs": feat_ops + rate * macs,
        "cnn_alone_macs": macs,
        "energy_pj_est": {
            "stage1_features_plus_xgb": e_stage1,
            "cnn_fp32": e_cnn32,
            "cnn_int8": e_cnn8,
            "cascade_fp32": e_stage1 + rate * e_cnn32,
            "cascade_int8": e_stage1 + rate * e_cnn8,
        },
        "energy_note": "Arithmetic only, Horowitz ISSCC 2014 45 nm figures; "
                       "excludes memory access, control and leakage.",
    }
    log.info(f"CNN: {params} params, {macs} MACs/beat | XGB: {comps:.0f} comparisons/beat | "
             f"features ~{feat_ops} MACs/beat")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    plot_tradeoff(sweeps_ds2, results, out_dir)
    log.info(f"Saved results to {out_dir}")


def plot_tradeoff(sw: pd.DataFrame, res: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for rule, style in (("p_normal", "o-"), ("max_prob", "s--")):
        s = sw[sw["rule"] == rule].sort_values("escalated")
        ax.plot(100 * s["escalated"], 100 * s["sens_S"], style, ms=3,
                label="escalate if P(N) < tau" if rule == "p_normal" else "escalate if max P < tau")
    ax.axhline(100 * res["ds2"]["xgboost"]["sens_S"], color="grey", ls=":", lw=1)
    ax.axhline(100 * res["ds2"]["cnn_fp32"]["sens_S"], color="black", ls=":", lw=1)
    ax.text(101, 100 * res["ds2"]["xgboost"]["sens_S"], "XGBoost", va="center", fontsize=8)
    ax.text(101, 100 * res["ds2"]["cnn_fp32"]["sens_S"], "CNN", va="center", fontsize=8)
    c = res["cascade"]
    ax.plot(100 * c["ds2"]["escalated"], 100 * c["ds2"]["sens_S"], "r*", ms=12,
            label=f"operating point (tau={c['tau']}, chosen on DS1)")
    ax.set_xlabel("Beats escalated to CNN (%)")
    ax.set_ylabel("S sensitivity on DS2 (%)")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cascade_tradeoff.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stage XGBoost -> CNN cascade")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
