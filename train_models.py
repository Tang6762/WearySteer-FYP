"""
train_models.py
---------------
Trains three drowsiness classifiers on the CSV produced by feature_extractor.py:

    1. Random Forest   (frame-level)
    2. XGBoost         (frame-level)
    3. GRU             (sequence-level, 30-sample windows ~ 5 s of video)

KEY DIFFERENCE FROM THE ORIGINAL train_model.py
    The original script used a random 80/20 split of individual frames. Frames from the
    same video are almost identical, so the same person ends up in both train and test and
    the accuracy is inflated (data leakage). Here the split is done BY SUBJECT:
    train / validation / test subjects are completely disjoint. This is the "cross-subject
    testing" the proposal's evaluation plan calls for, and the held-out subject list is
    saved to split_manifest.json so compare_models.py can test on videos the models never saw.

Outputs
    models/wearysteer_scaler.pkl, wearysteer_rf.pkl, wearysteer_xgb.pkl, wearysteer_gru.pt
    models/split_manifest.json
    evidence/training_report.json, training_confusion_matrices.png,
             feature_importance.png, gru_training_curve.png

Usage
    pip install xgboost torch matplotlib scikit-learn pandas joblib
    python train_models.py --csv wearysteer_features.csv
    python train_models.py --skip-gru            # if PyTorch is not installed
"""
import argparse
import json
import os
import time
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from wearysteer_common import (
    CLASS_NAMES, FEATURE_COLS, SEQ_LEN, assign_video_ids, build_gru, compute_metrics,
    make_windows, package_versions, save_gru, torch_available,
)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def subject_split(subjects, val_frac, test_frac, seed):
    rng = np.random.RandomState(seed)
    subjects = np.array(sorted(subjects))
    rng.shuffle(subjects)
    n = len(subjects)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test = subjects[:n_test]
    val = subjects[n_test:n_test + n_val]
    train = subjects[n_test + n_val:]
    return list(train), list(val), list(test)


def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"'{csv_path}' not found. Run feature_extractor.py first.")
    df = pd.read_csv(csv_path).dropna()
    df["subject_id"] = df["subject_id"].astype(str)
    df["label"] = df["label"].astype(int)
    df["video_id"] = assign_video_ids(df)
    return df


# --------------------------------------------------------------------------
# GRU training
# --------------------------------------------------------------------------
def train_gru(Xw_tr, yw_tr, Xw_val, yw_val, args, device):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)
    cfg = {"n_features": len(FEATURE_COLS), "hidden": args.hidden, "layers": 2,
           "n_classes": 3, "dropout": 0.3, "seq_len": args.seq_len}
    model = build_gru(cfg["n_features"], cfg["hidden"], cfg["layers"], cfg["n_classes"], cfg["dropout"]).to(device)

    weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=yw_tr)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_dl = DataLoader(TensorDataset(torch.tensor(Xw_tr), torch.tensor(yw_tr)),
                          batch_size=args.batch_size, shuffle=True)
    Xv, yv = torch.tensor(Xw_val).to(device), torch.tensor(yw_val).to(device)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * len(xb)
            seen += len(xb)

        model.eval()
        with torch.no_grad():
            logits = model(Xv)
            val_loss = criterion(logits, yv).item()
            val_acc = (logits.argmax(1) == yv).float().mean().item()
        history["train_loss"].append(running / seen)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"    epoch {epoch:02d}  train_loss={running / seen:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val - 1e-4:
            best_val, wait = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f"    early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, cfg, history


def gru_predict(model, Xw, device, batch=1024):
    import torch
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xw), batch):
            xb = torch.tensor(Xw[i:i + batch]).to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds) if preds else np.empty((0,), dtype=int)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_confusions(results, path):
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.4))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        cm = np.array(results[name]["confusion_matrix"])
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j] * 100:.0f}%)", ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES, rotation=20); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{name}\nmacro-F1 {results[name]['macro_f1']:.3f}")
    fig.suptitle("Held-out subjects: confusion matrices (rows normalised)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(rf, xgb, path):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(FEATURE_COLS))
    ax.bar(x - 0.2, rf.feature_importances_, 0.4, label="Random Forest")
    ax.bar(x + 0.2, xgb.feature_importances_, 0.4, label="XGBoost")
    ax.set_xticks(x); ax.set_xticklabels(FEATURE_COLS)
    ax.set_ylabel("Importance"); ax.set_title("Feature importance"); ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gru_curve(history, path):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(history["train_loss"], label="train loss")
    ax.plot(history["val_loss"], label="val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy"); ax.set_title("GRU training curve"); ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train RF, XGBoost and GRU drowsiness models")
    ap.add_argument("--csv", default="wearysteer_features.csv")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--evidence-dir", default="evidence")
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--window-stride", type=int, default=5, help="stride between training windows (samples)")
    ap.add_argument("--val-frac", type=float, default=0.15, help="fraction of SUBJECTS used for validation")
    ap.add_argument("--test-frac", type=float, default=0.15, help="fraction of SUBJECTS held out for testing")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-gru", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.evidence_dir, exist_ok=True)

    use_gru = not args.skip_gru
    if use_gru and not torch_available():
        print("[!] PyTorch is not installed -> skipping the GRU. (pip install torch)")
        use_gru = False
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise SystemExit("[-] xgboost is not installed. Run: pip install xgboost")

    # ---- data + subject-wise split -------------------------------------
    df = load_data(args.csv)
    train_s, val_s, test_s = subject_split(df["subject_id"].unique(), args.val_frac, args.test_frac, args.seed)
    print(f"[*] {len(df)} samples, {df['subject_id'].nunique()} subjects, {df['video_id'].nunique()} videos")
    print(f"[*] Subjects  train={len(train_s)}  val={len(val_s)}  test={len(test_s)}  (no overlap)")

    parts = {}
    for name, subj in [("train", train_s), ("val", val_s), ("test", test_s)]:
        d = df[df["subject_id"].isin(subj)].reset_index(drop=True)
        parts[name] = (d[FEATURE_COLS].to_numpy(dtype=np.float32), d["label"].to_numpy(), d["video_id"].to_numpy())
        print(f"    {name:5s}: {len(d):7d} samples | class counts {np.bincount(d['label'], minlength=3).tolist()}")

    # Scaler is fitted on TRAIN only (the original script's approach, kept) and shared by all 3 models
    scaler = StandardScaler().fit(parts["train"][0])
    scaled = {k: (scaler.transform(v[0]).astype(np.float32), v[1], v[2]) for k, v in parts.items()}
    X_tr, y_tr, _ = scaled["train"]
    X_val, y_val, _ = scaled["val"]
    X_te, y_te, vid_te = scaled["test"]

    # Test windows define the common evaluation points for ALL models (aligned comparison)
    Xw_te, yw_te, end_te = make_windows(X_te, y_te, vid_te, args.seq_len, stride=args.window_stride)
    if len(yw_te) == 0:
        raise SystemExit("[-] No test windows could be built; the test videos are shorter than --seq-len.")
    y_eval = yw_te

    results, timings = {}, {}
    sw_tr = compute_sample_weight("balanced", y_tr)

    # ---- Random Forest ---------------------------------------------------
    print("\n[*] Training Random Forest ...")
    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=2,
                                class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    timings["Random Forest"] = time.perf_counter() - t0
    results["Random Forest"] = compute_metrics(y_eval, rf.predict(X_te)[end_te])

    # ---- XGBoost ---------------------------------------------------------
    print("[*] Training XGBoost ...")
    t0 = time.perf_counter()
    xgb = XGBClassifier(objective="multi:softprob", n_estimators=600, learning_rate=0.05, max_depth=6,
                        subsample=0.8, colsample_bytree=1.0, eval_metric="mlogloss",
                        early_stopping_rounds=30, random_state=args.seed, n_jobs=-1)
    xgb.fit(X_tr, y_tr, sample_weight=sw_tr, eval_set=[(X_val, y_val)], verbose=False)
    timings["XGBoost"] = time.perf_counter() - t0
    print(f"    best iteration: {xgb.best_iteration}")
    results["XGBoost"] = compute_metrics(y_eval, xgb.predict(X_te)[end_te])

    # ---- GRU -------------------------------------------------------------
    history = None
    if use_gru:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Training GRU on {device} (window = {args.seq_len} samples) ...")
        Xw_tr, yw_tr, _ = make_windows(X_tr, y_tr, scaled["train"][2], args.seq_len, args.window_stride)
        Xw_val, yw_val, _ = make_windows(X_val, y_val, scaled["val"][2], args.seq_len, args.window_stride)
        print(f"    windows: train={len(yw_tr)}  val={len(yw_val)}  test={len(yw_te)}")
        t0 = time.perf_counter()
        gru, gru_cfg, history = train_gru(Xw_tr, yw_tr, Xw_val, yw_val, args, device)
        timings["GRU"] = time.perf_counter() - t0
        results["GRU"] = compute_metrics(y_eval, gru_predict(gru, Xw_te, device))
        save_gru(gru, os.path.join(args.models_dir, "wearysteer_gru.pt"), gru_cfg)

    # ---- save artefacts --------------------------------------------------
    joblib.dump(scaler, os.path.join(args.models_dir, "wearysteer_scaler.pkl"))
    joblib.dump(rf, os.path.join(args.models_dir, "wearysteer_rf.pkl"))
    joblib.dump(xgb, os.path.join(args.models_dir, "wearysteer_xgb.pkl"))
    with open(os.path.join(args.models_dir, "split_manifest.json"), "w") as f:
        json.dump({"seed": args.seed, "seq_len": args.seq_len, "train_subjects": train_s,
                   "val_subjects": val_s, "test_subjects": test_s}, f, indent=2)

    # ---- report ----------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"HELD-OUT SUBJECT RESULTS  ({len(y_eval)} aligned test samples, {len(test_s)} unseen subjects)")
    print("=" * 78)
    print(f"{'Model':<15}{'Acc':>8}{'BalAcc':>9}{'MacroF1':>10}{'Drowsy P':>10}{'Drowsy R':>10}{'Drowsy F1':>11}{'Train s':>9}")
    for name, m in results.items():
        print(f"{name:<15}{m['accuracy']:>8.3f}{m['balanced_accuracy']:>9.3f}{m['macro_f1']:>10.3f}"
              f"{m['drowsy_precision']:>10.3f}{m['drowsy_recall']:>10.3f}{m['drowsy_f1']:>11.3f}{timings[name]:>9.1f}")
    for name, m in results.items():
        print(f"\n{name} confusion matrix (rows=true, cols=pred; Alert / LowVig / Drowsy):")
        print(np.array(m["confusion_matrix"]))

    plot_confusions(results, os.path.join(args.evidence_dir, "training_confusion_matrices.png"))
    plot_importance(rf, xgb, os.path.join(args.evidence_dir, "feature_importance.png"))
    if history:
        plot_gru_curve(history, os.path.join(args.evidence_dir, "gru_training_curve.png"))

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "split": {"train_subjects": train_s, "val_subjects": val_s, "test_subjects": test_s},
        "training_seconds": timings,
        "test_metrics": results,
        "gru_history": history,
        "environment": package_versions(),
    }
    with open(os.path.join(args.evidence_dir, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Models saved to '{args.models_dir}/', evidence saved to '{args.evidence_dir}/'.")


if __name__ == "__main__":
    main()
