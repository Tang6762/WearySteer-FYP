"""
wearysteer_common.py
--------------------
Shared code for train_models.py and compare_models.py.

Everything that must stay IDENTICAL between training and evaluation lives here
(landmark indices, EAR/MAR/slump formulas, sampling stride, sequence length,
GRU architecture, metric definitions) so the two scripts can never drift apart.

The feature formulas are copied verbatim from feature_extractor.py.
"""
import platform
import sys
from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# --------------------------------------------------------------------------
# Constants (must match feature_extractor.py)
# --------------------------------------------------------------------------
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 81, 13, 312, 317, 14, 87, 178]

FEATURE_COLS = ["ear", "mar", "shoulder_slump"]
CLASS_NAMES = ["Alert", "Low Vigilance", "Drowsy"]
FILENAME_TO_LABEL = {"0": 0, "5": 1, "10": 2}   # UTA-RLDD naming convention
FRAME_STRIDE = 5        # feature_extractor.py keeps every 5th frame
SEQ_LEN = 30            # GRU window: 30 samples * 5 frames @30fps = ~5 s
DROWSY = 2


# --------------------------------------------------------------------------
# Feature extraction primitives (verbatim from feature_extractor.py)
# --------------------------------------------------------------------------
def calculate_ear(landmarks, eye_indices, img_w, img_h):
    pts = [(int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h)) for idx in eye_indices]
    d_v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    d_v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    d_h = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if d_h == 0:
        return 0.0
    return (d_v1 + d_v2) / (2.0 * d_h)


def calculate_mar(landmarks, mouth_indices, img_w, img_h):
    pts = [(int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h)) for idx in mouth_indices]
    d_v = np.linalg.norm(np.array(pts[2]) - np.array(pts[5]))
    d_h = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if d_h == 0:
        return 0.0
    return d_v / d_h


# --------------------------------------------------------------------------
# Sequence helpers
# --------------------------------------------------------------------------
def assign_video_ids(df):
    """
    The CSV written by feature_extractor.py has no video-id column, but rows are
    appended one whole video at a time, and each subject has exactly one video
    per label (0 / 5 / 10). A new video therefore starts whenever subject_id or
    label changes between consecutive rows.
    """
    new_video = (df["subject_id"] != df["subject_id"].shift()) | (df["label"] != df["label"].shift())
    return new_video.cumsum().to_numpy()


def make_windows(X, y, video_ids, seq_len, stride):
    """
    Slice per-video sliding windows for the GRU.
    Returns (windows [N, seq_len, F], labels [N], end_idx [N]) where end_idx is the
    row index (into X) of the LAST sample in each window. end_idx lets us score the
    frame-level models (RF / XGB) on exactly the same samples as the GRU.
    """
    windows, labels, ends = [], [], []
    for vid in np.unique(video_ids):
        idx = np.where(video_ids == vid)[0]
        if len(idx) < seq_len:
            continue
        for end in range(seq_len - 1, len(idx), stride):
            rows = idx[end - seq_len + 1: end + 1]
            windows.append(X[rows])
            labels.append(y[idx[end]])
            ends.append(idx[end])
    if not windows:
        return (np.empty((0, seq_len, X.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=int), np.empty((0,), dtype=int))
    return np.asarray(windows, dtype=np.float32), np.asarray(labels, dtype=int), np.asarray(ends, dtype=int)


def smooth_majority(preds, k):
    """Causal majority vote over the last k predictions (same idea as realtime_detector.py)."""
    if k <= 1:
        return np.asarray(preds)
    out = np.empty_like(preds)
    for i in range(len(preds)):
        window = preds[max(0, i - k + 1): i + 1]
        out[i] = Counter(window.tolist()).most_common(1)[0][0]
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    labels = [0, 1, 2]
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)) if len(set(y_true)) > 1 else float("nan"),
        "per_class": {
            CLASS_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
            for i in labels
        },
        "drowsy_precision": float(p[DROWSY]),
        "drowsy_recall": float(r[DROWSY]),
        "drowsy_f1": float(f[DROWSY]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


# --------------------------------------------------------------------------
# GRU (PyTorch is imported lazily so RF/XGB still work without it)
# --------------------------------------------------------------------------
def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def build_gru(n_features=3, hidden=64, layers=2, n_classes=3, dropout=0.3):
    import torch.nn as nn

    class GRUClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

        def forward(self, x):                      # x: [B, T, F]
            out, _ = self.gru(x)
            return self.head(out[:, -1, :])        # classify from the last time step

    return GRUClassifier()


def save_gru(model, path, config):
    import torch
    torch.save({"state_dict": model.state_dict(), "config": config}, path)


def load_gru(path, device="cpu"):
    import torch
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = build_gru(cfg["n_features"], cfg["hidden"], cfg["layers"], cfg["n_classes"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, cfg


def package_versions():
    """Logged into the evidence files (report section 8.8: external components and dependencies)."""
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ["numpy", "pandas", "sklearn", "xgboost", "torch", "cv2", "mediapipe"]:
        try:
            mod = __import__(name)
            info[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[name] = "not installed"
    return info
