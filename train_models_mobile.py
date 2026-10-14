"""
train_models_mobile.py
----------------------
Trains mobile-ready drowsiness classifiers on the CSV produced by feature_extractor.py 
and exports them directly to TensorFlow Lite (.tflite) format for Flutter apps.

Models trained:
    1. Dense Classifier (replaces Random Forest for mobile efficiency)
    2. XGB-style Deep Net (replaces XGBoost)
    3. GRU Sequence Model (replaces PyTorch GRU with a native TensorFlow GRU)

KEY DIFFERENCE FROM THE ORIGINAL
    Maintains subject-wise disjoint splits (train / val / test) to prevent data leakage,
    while utilizing TensorFlow/Keras architectures that compile seamlessly into .tflite.

Outputs
    models/wearysteer_scaler.pkl
    models/wearysteer_dense.tflite
    models/wearysteer_deep.tflite
    models/wearysteer_gru.tflite
    models/split_manifest.json
    evidence/mobile_training_report.json

Usage
    pip install tensorflow scikit-learn pandas joblib matplotlib
    python train_models_mobile.py --csv wearysteer_features.csv
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
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models

from wearysteer_common import (
    CLASS_NAMES, FEATURE_COLS, SEQ_LEN, assign_video_ids, make_windows, compute_metrics
)

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

def save_tflite(keras_model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"    [+] Exported TFLite model to {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")

def build_dense_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_deep_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_gru_model(seq_len, n_features):
    model = models.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.GRU(64, return_sequences=False, dropout=0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    ap = argparse.ArgumentParser(description="Train mobile-ready TFLite drowsiness models")
    ap.add_argument("--csv", default="wearysteer_features.csv")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--evidence-dir", default="evidence")
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--window-stride", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.evidence_dir, exist_ok=True)

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Data preparation & subject splitting ----
    df = load_data(args.csv)
    train_s, val_s, test_s = subject_split(df["subject_id"].unique(), args.val_frac, args.test_frac, args.seed)
    print(f"[*] {len(df)} samples, {df['subject_id'].nunique()} subjects, video-disjoint split initialized.")

    parts = {}
    for name, subj in [("train", train_s), ("val", val_s), ("test", test_s)]:
        d = df[df["subject_id"].isin(subj)].reset_index(drop=True)
        parts[name] = (d[FEATURE_COLS].to_numpy(dtype=np.float32), d["label"].to_numpy(), d["video_id"].to_numpy())

    scaler = StandardScaler().fit(parts["train"][0])
    scaled = {k: (scaler.transform(v[0]).astype(np.float32), v[1], v[2]) for k, v in parts.items()}
    
    X_tr, y_tr, _ = scaled["train"]
    X_val, y_val, _ = scaled["val"]
    X_te, y_te, vid_te = scaled["test"]

    # Compute class weights for balanced loss training
    class_weights_arr = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weights = dict(enumerate(class_weights_arr))

    results, timings = {}, {}

    # ---- 1. Frame-level Dense Model (Replaces RF) ----
    print("\n[*] Training Frame-level Dense Model ...")
    t0 = time.perf_counter()
    dense_model = build_dense_model(X_tr.shape[1])
    dense_model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=args.epochs, 
                    batch_size=args.batch_size, class_weight=class_weights, verbose=0)
    timings["Dense (Mobile RF)"] = time.perf_counter() - t0
    
    preds_dense = dense_model.predict(X_te, verbose=0).argmax(axis=1)
    results["Dense (Mobile RF)"] = compute_metrics(y_te, preds_dense)
    save_tflite(dense_model, os.path.join(args.models_dir, "wearysteer_dense.tflite"))

    # ---- 2. Deep Classifier (Replaces XGBoost) ----
    print("\n[*] Training Deep Feature Classifier ...")
    t0 = time.perf_counter()
    deep_model = build_deep_model(X_tr.shape[1])
    deep_model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=args.epochs, 
                   batch_size=args.batch_size, class_weight=class_weights, verbose=0)
    timings["Deep Net (Mobile XGB)"] = time.perf_counter() - t0
    
    preds_deep = deep_model.predict(X_te, verbose=0).argmax(axis=1)
    results["Deep Net (Mobile XGB)"] = compute_metrics(y_te, preds_deep)
    save_tflite(deep_model, os.path.join(args.models_dir, "wearysteer_deep.tflite"))

    # ---- 3. Sequence GRU Model ----
    print("\n[*] Training Sequence GRU Model ...")
    Xw_tr, yw_tr, _ = make_windows(X_tr, y_tr, scaled["train"][2], args.seq_len, args.window_stride)
    Xw_val, yw_val, _ = make_windows(X_val, y_val, scaled["val"][2], args.seq_len, args.window_stride)
    Xw_te, yw_te, end_te = make_windows(X_te, y_te, vid_te, args.seq_len, stride=args.window_stride)

    if len(yw_te) > 0:
        gru_weights_arr = compute_class_weight("balanced", classes=np.unique(yw_tr), y=yw_tr)
        gru_weights = dict(enumerate(gru_weights_arr))

        t0 = time.perf_counter()
        gru_model = build_gru_model(args.seq_len, len(FEATURE_COLS))
        gru_model.fit(Xw_tr, yw_tr, validation_data=(Xw_val, yw_val), epochs=args.epochs,
                      batch_size=args.batch_size, class_weight=gru_weights, verbose=0)
        timings["GRU (Mobile Sequence)"] = time.perf_counter() - t0

        preds_gru = gru_model.predict(Xw_te, verbose=0).argmax(axis=1)
        results["GRU (Mobile Sequence)"] = compute_metrics(yw_te, preds_gru)
        save_tflite(gru_model, os.path.join(args.models_dir, "wearysteer_gru.tflite"))

    # ---- Save Scaler and Manifest Artifacts ----
    joblib.dump(scaler, os.path.join(args.models_dir, "wearysteer_scaler.pkl"))
    with open(os.path.join(args.models_dir, "split_manifest.json"), "w") as f:
        json.dump({"seed": args.seed, "seq_len": args.seq_len, "train_subjects": train_s,
                   "val_subjects": val_s, "test_subjects": test_s}, f, indent=2)

    print("\n" + "=" * 70)
    print("MOBILE TFLITE TRAINING COMPLETE")
    print("=" * 70)
    for name, m in results.items():
        print(f"{name:<25} Accuracy: {m['accuracy']:.3f} | Macro F1: {m['macro_f1']:.3f}")

if __name__ == "__main__":
    main()