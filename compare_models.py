"""
compare_models.py
-----------------
Runs Random Forest, XGBoost and GRU on UTA-RLDD video(s), records performance metrics for
each, and prints a side-by-side comparison.

Video selection (pick one)
    --video  E:\\FYP-Dataset\\UTA-RLDD\\Fold1\\12\\10.mp4  [more videos ...]
    --subject-dir E:\\FYP-Dataset\\UTA-RLDD\\Fold1\\12        (uses its 0 / 5 / 10 videos)
    --heldout --dataset-root E:\\FYP-Dataset\\UTA-RLDD      (every video of the test subjects
                                                          saved by train_models.py)

The ground-truth label comes from the file name (0 -> Alert, 5 -> Low Vigilance, 10 -> Drowsy),
the same convention feature_extractor.py used. For a meaningful macro-F1 supply videos of more
than one class (e.g. --subject-dir or --heldout); a single video only has one true class.

Metrics reported per model
    Accuracy, balanced accuracy, macro-F1, drowsy precision / recall / F1, Cohen's kappa,
    drowsy false alarms per minute (on Alert videos), time to first drowsy alert (on Drowsy
    videos), per-sample inference latency, model size, plus per-video breakdown.

Fair-comparison rules
    * Features are extracted ONCE per video and shared by all models (identical every-5th-frame
      sampling as in training).
    * All models are scored on the same samples: the first SEQ_LEN-1 samples of each video are
      excluded for everyone, because the GRU needs a full window.
    * The same causal majority-vote smoothing (--smooth) is applied to every model.

Usage
    python compare_models.py --subject-dir E:\\FYP-Dataset\\UTA-RLDD\\Fold1\\12
    python compare_models.py --heldout --dataset-root E:\\FYP-Dataset\\UTA-RLDD
"""
import argparse
import hashlib
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

from wearysteer_common import (
    CLASS_NAMES, DROWSY, FILENAME_TO_LABEL, FRAME_STRIDE, LEFT_EYE, MOUTH, RIGHT_EYE, SEQ_LEN,
    calculate_ear, calculate_mar, compute_metrics, load_gru, package_versions, smooth_majority,
    torch_available,
)

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


# --------------------------------------------------------------------------
# Video discovery
# --------------------------------------------------------------------------
def label_from_path(path):
    return FILENAME_TO_LABEL.get(os.path.splitext(os.path.basename(path))[0])


def collect_videos(args, manifest):
    videos = []
    if args.video:
        videos = list(args.video)
    elif args.subject_dir:
        for f in sorted(os.listdir(args.subject_dir)):
            if f.lower().endswith(VIDEO_EXTS):
                videos.append(os.path.join(args.subject_dir, f))
    elif args.heldout:
        if not manifest:
            raise SystemExit("[-] --heldout needs models/split_manifest.json (run train_models.py first).")
        if not args.dataset_root:
            raise SystemExit("[-] --heldout needs --dataset-root.")
        test_subjects = set(manifest["test_subjects"])
        for root, _, files in os.walk(args.dataset_root):
            if os.path.basename(root) in test_subjects:
                videos += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(VIDEO_EXTS)]
    else:
        raise SystemExit("[-] Provide --video, --subject-dir, or --heldout --dataset-root.")

    kept = []
    for v in videos:
        if not os.path.exists(v):
            print(f"[!] Not found, skipped: {v}")
        elif args.label is None and label_from_path(v) is None:
            print(f"[!] Cannot infer label from file name (expected 0/5/10), skipped: {v}")
        else:
            kept.append(v)
    if args.max_videos:
        kept = kept[:args.max_videos]
    if not kept:
        raise SystemExit("[-] No usable videos.")
    return kept


# --------------------------------------------------------------------------
# Feature extraction (shared by all three models, cached on disk)
# --------------------------------------------------------------------------
def cache_path(cache_dir, video):
    key = f"{os.path.abspath(video)}|{os.path.getsize(video)}|{FRAME_STRIDE}"
    return os.path.join(cache_dir, hashlib.md5(key.encode()).hexdigest() + ".npz")


def extract_video_features(video, cache_dir, use_cache=True):
    cp = cache_path(cache_dir, video)
    if use_cache and os.path.exists(cp):
        d = np.load(cp)
        return {k: d[k] for k in d.files}

    import cv2
    import mediapipe as mp   # imported lazily so cached runs don't need it

    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.5)
    pose_lm = mp.solutions.pose.PoseLandmark

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    feats, times, attempted = [], [], 0
    t0 = time.perf_counter()
    frame_no = 0
    print(f"    extracting features ({total} frames @ {fps:.1f} fps) ...", flush=True)
    while True:
        if not cap.grab():
            break
        frame_no += 1
        if frame_no % FRAME_STRIDE != 0:       # same sampling rule as feature_extractor.py
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        attempted += 1
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fr = face_mesh.process(rgb)
        if not fr.multi_face_landmarks:
            continue
        lm = fr.multi_face_landmarks[0].landmark
        ear = (calculate_ear(lm, LEFT_EYE, w, h) + calculate_ear(lm, RIGHT_EYE, w, h)) / 2.0
        mar = calculate_mar(lm, MOUTH, w, h)
        slump = 0.0
        pr = pose.process(rgb)
        if pr.pose_landmarks:
            p = pr.pose_landmarks.landmark
            slump = (p[pose_lm.LEFT_SHOULDER].y + p[pose_lm.RIGHT_SHOULDER].y) / 2.0 - p[pose_lm.NOSE].y
        feats.append([ear, mar, slump])
        times.append(frame_no / fps)
    cap.release()
    face_mesh.close()
    pose.close()
    elapsed = time.perf_counter() - t0

    out = {
        "feats": np.asarray(feats, dtype=np.float32).reshape(-1, 3),
        "times": np.asarray(times, dtype=np.float32),
        "fps": np.float32(fps),
        "frames_read": np.int32(frame_no),
        "attempted": np.int32(attempted),
        "extract_seconds": np.float32(elapsed),
    }
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cp, **out)
    return out


# --------------------------------------------------------------------------
# Model wrappers: batch predict (for metrics) + single-sample predict (for latency)
# --------------------------------------------------------------------------
class SklearnLike:
    def __init__(self, name, model, scaler, path):
        self.name, self.model, self.scaler = name, model, scaler
        self.size_kb = os.path.getsize(path) / 1024.0
        if hasattr(model, "set_params"):
            try:
                model.set_params(n_jobs=1)      # fair single-thread latency, like a phone/edge CPU
            except Exception:
                pass

    def predict_seq(self, scaled):               # -> prediction for every sample
        return self.model.predict(scaled).astype(int)

    def predict_one(self, scaled, i):
        return self.model.predict(scaled[i:i + 1])


class GRUWrapper:
    def __init__(self, path):
        self.name = "GRU"
        self.device = "cpu"                      # CPU on purpose: matches the edge-deployment story
        self.model, self.cfg = load_gru(path, self.device)
        self.seq_len = self.cfg["seq_len"]
        self.size_kb = os.path.getsize(path) / 1024.0

    def _windows(self, scaled, idxs):
        return np.stack([scaled[i - self.seq_len + 1: i + 1] for i in idxs]).astype(np.float32)

    def predict_seq(self, scaled):
        import torch
        n = len(scaled)
        preds = np.zeros(n, dtype=int)           # first seq_len-1 samples have no full window -> excluded later
        idxs = list(range(self.seq_len - 1, n))
        with torch.no_grad():
            for s in range(0, len(idxs), 1024):
                chunk = idxs[s:s + 1024]
                logits = self.model(torch.tensor(self._windows(scaled, chunk)))
                preds[chunk] = logits.argmax(1).numpy()
        return preds

    def predict_one(self, scaled, i):
        import torch
        with torch.no_grad():
            return self.model(torch.tensor(self._windows(scaled, [i]))).argmax(1).numpy()


def measure_latency_ms(wrapper, scaled, start, n_max):
    idxs = list(range(start, min(len(scaled), start + n_max)))
    if not idxs:
        return float("nan")
    wrapper.predict_one(scaled, idxs[0])         # warm-up
    t0 = time.perf_counter()
    for i in idxs:
        wrapper.predict_one(scaled, i)
    return (time.perf_counter() - t0) / len(idxs) * 1000.0


# --------------------------------------------------------------------------
# Alert-oriented metrics (proposal evaluation plan: false alarms + early warning)
# --------------------------------------------------------------------------
def alert_events(pred, times):
    """Rising edges of the Drowsy state = distinct alerts the driver would hear."""
    is_d = (pred == DROWSY)
    edges = np.where(is_d & ~np.concatenate([[False], is_d[:-1]]))[0]
    return edges, (times[edges[0]] if len(edges) else None)


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
METRIC_ROWS = [
    # key, label, higher_is_better, format
    ("accuracy", "Accuracy", True, "{:.3f}"),
    ("balanced_accuracy", "Balanced accuracy", True, "{:.3f}"),
    ("macro_f1", "Macro F1", True, "{:.3f}"),
    ("drowsy_precision", "Drowsy precision", True, "{:.3f}"),
    ("drowsy_recall", "Drowsy recall", True, "{:.3f}"),
    ("drowsy_f1", "Drowsy F1", True, "{:.3f}"),
    ("kappa", "Cohen's kappa", True, "{:.3f}"),
    ("false_alarms_per_min", "False alarms / min (Alert videos)", False, "{:.2f}"),
    ("mean_time_to_alert_s", "Time to 1st drowsy alert (s, Drowsy videos)", False, "{:.1f}"),
    ("drowsy_videos_detected", "Drowsy videos with >=1 alert", True, "{:.0%}"),
    ("latency_ms", "Inference latency (ms/sample)", False, "{:.3f}"),
    ("size_kb", "Model size (KB)", False, "{:.0f}"),
]


def best_flags(summary, models):
    flags = {}
    for key, _, higher, _ in METRIC_ROWS:
        vals = {m: summary[m].get(key) for m in models}
        vals = {m: v for m, v in vals.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
        if len(vals) < 2:
            continue
        best = max(vals.values()) if higher else min(vals.values())
        for m, v in vals.items():
            if v == best:
                flags[(key, m)] = True
    return flags


def print_comparison(summary, models, flags):
    w = 46
    print("\n" + "=" * (w + 16 * len(models)))
    print("MODEL COMPARISON   (* = best in row)")
    print("=" * (w + 16 * len(models)))
    print(f"{'Metric':<{w}}" + "".join(f"{m:>16}" for m in models))
    print("-" * (w + 16 * len(models)))
    for key, label, _, fmt in METRIC_ROWS:
        row = f"{label:<{w}}"
        for m in models:
            v = summary[m].get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cell = "n/a"
            else:
                cell = fmt.format(v) + (" *" if flags.get((key, m)) else "")
            row += f"{cell:>16}"
        print(row)
    wins = {m: sum(1 for (k, mm) in flags if mm == m) for m in models}
    print("-" * (w + 16 * len(models)))
    print(f"{'Rows won':<{w}}" + "".join(f"{wins[m]:>16}" for m in models))


def write_latex(summary, models, flags, path):
    def esc(s):
        return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
    lines = ["\\begin{tabular}{l" + "r" * len(models) + "}", "\\hline",
             "Metric & " + " & ".join(models) + " \\\\", "\\hline"]
    for key, label, _, fmt in METRIC_ROWS:
        cells = []
        for m in models:
            v = summary[m].get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("n/a")
            else:
                txt = esc(fmt.format(v))
                cells.append("\\textbf{" + txt + "}" if flags.get((key, m)) else txt)
        lines.append(esc(label) + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def plot_confusions(summary, models, path):
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.4))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        cm = np.array(summary[m]["confusion_matrix"])
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES, rotation=20); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{m}\nmacro-F1 {summary[m]['macro_f1']:.3f}")
    fig.suptitle("Video evaluation: confusion matrices")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_timeline(video_name, true_label, times, preds_by_model, path):
    fig, ax = plt.subplots(figsize=(10, 2.2 + 0.5 * len(preds_by_model)))
    for k, (m, p) in enumerate(preds_by_model.items()):
        ax.scatter(times, np.full(len(times), k), c=p, cmap="RdYlGn_r", vmin=0, vmax=2, s=6, marker="s")
    ax.set_yticks(range(len(preds_by_model))); ax.set_yticklabels(list(preds_by_model))
    ax.set_xlabel("Video time (s)")
    ax.set_title(f"{video_name} (true: {CLASS_NAMES[true_label]})  green=Alert  yellow=Low vigilance  red=Drowsy")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Compare RF / XGBoost / GRU on UTA-RLDD videos")
    ap.add_argument("--video", nargs="+", help="one or more video files")
    ap.add_argument("--subject-dir", help="folder of one subject (0/5/10 videos)")
    ap.add_argument("--heldout", action="store_true", help="use every video of the held-out test subjects")
    ap.add_argument("--dataset-root", help="UTA-RLDD root (needed with --heldout)")
    ap.add_argument("--label", type=int, choices=[0, 1, 2], help="override ground truth for ALL videos")
    ap.add_argument("--max-videos", type=int, help="cap the number of videos (quick tests)")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--evidence-dir", default="evidence")
    ap.add_argument("--cache-dir", default="feature_cache")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--smooth", type=int, default=5,
                    help="majority-vote window (samples; 5 samples ~ 0.8 s of video). 1 = off")
    ap.add_argument("--latency-samples", type=int, default=100)
    ap.add_argument("--timelines", action="store_true", help="also save a prediction timeline per video")
    args = ap.parse_args()

    os.makedirs(args.evidence_dir, exist_ok=True)

    manifest = None
    mpath = os.path.join(args.models_dir, "split_manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            manifest = json.load(f)
    seq_len = manifest["seq_len"] if manifest else SEQ_LEN

    # ---- load models -----------------------------------------------------
    scaler_path = os.path.join(args.models_dir, "wearysteer_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise SystemExit(f"[-] {scaler_path} not found. Run train_models.py first.")
    scaler = joblib.load(scaler_path)
    wrappers = []
    for name, fname in [("Random Forest", "wearysteer_rf.pkl"), ("XGBoost", "wearysteer_xgb.pkl")]:
        p = os.path.join(args.models_dir, fname)
        if os.path.exists(p):
            wrappers.append(SklearnLike(name, joblib.load(p), scaler, p))
        else:
            print(f"[!] {p} missing -> {name} skipped")
    gp = os.path.join(args.models_dir, "wearysteer_gru.pt")
    if os.path.exists(gp) and torch_available():
        wrappers.append(GRUWrapper(gp))
    else:
        print("[!] GRU skipped (model file missing or PyTorch not installed)")
    if not wrappers:
        raise SystemExit("[-] No models loaded.")
    names = [w.name for w in wrappers]
    print(f"[+] Models: {', '.join(names)} | warm-up excluded: first {seq_len - 1} samples/video | smoothing: {args.smooth}")

    videos = collect_videos(args, manifest)
    train_like = set()
    if manifest:
        train_like = set(manifest["train_subjects"]) | set(manifest["val_subjects"])

    # ---- per-video evaluation -------------------------------------------
    per_video_rows = []
    pooled = {n: {"y": [], "p": []} for n in names}
    alert_stats = {n: {"fa": 0.0, "fa_minutes": 0.0, "tta": [], "drowsy_total": 0, "drowsy_hit": 0} for n in names}
    latencies = {n: [] for n in names}
    extraction = []

    for vi, video in enumerate(videos, 1):
        subject = os.path.basename(os.path.dirname(video))
        true_label = args.label if args.label is not None else label_from_path(video)
        print(f"\n[{vi}/{len(videos)}] subject {subject}  {os.path.basename(video)}  true = {CLASS_NAMES[true_label]}")
        if subject in train_like:
            print("    [!] WARNING: this subject was used for TRAINING/VALIDATION -> results will be optimistic.")

        feat = extract_video_features(video, args.cache_dir, use_cache=not args.no_cache)
        X, times = feat["feats"], feat["times"]
        n = len(X)
        det_rate = n / max(int(feat["attempted"]), 1)
        extraction.append({"video": video, "samples": n, "face_detect_rate": det_rate,
                           "extract_seconds": float(feat["extract_seconds"]),
                           "extract_fps": float(feat["frames_read"]) / max(float(feat["extract_seconds"]), 1e-9)})
        print(f"    {n} samples with a face ({det_rate:.0%} of sampled frames)")
        if n <= seq_len:
            print(f"    [!] too short for a {seq_len}-sample window, skipped")
            continue

        scaled = scaler.transform(X).astype(np.float32)
        start = seq_len - 1
        eval_times = times[start:]
        y_true = np.full(n - start, true_label)
        preds_for_plot = {}

        for w in wrappers:
            raw = w.predict_seq(scaled)
            smooth = smooth_majority(raw[start:], args.smooth)
            pooled[w.name]["y"].append(y_true)
            pooled[w.name]["p"].append(smooth)
            preds_for_plot[w.name] = smooth

            acc = float((smooth == y_true).mean())
            counts = np.bincount(smooth, minlength=3) / len(smooth)
            verdict = CLASS_NAMES[int(np.bincount(smooth, minlength=3).argmax())]
            edges, first_t = alert_events(smooth, eval_times)
            minutes = (eval_times[-1] - eval_times[0]) / 60.0 if len(eval_times) > 1 else 0.0
            if true_label == 0:
                alert_stats[w.name]["fa"] += len(edges)
                alert_stats[w.name]["fa_minutes"] += minutes
            if true_label == DROWSY:
                alert_stats[w.name]["drowsy_total"] += 1
                if first_t is not None:
                    alert_stats[w.name]["drowsy_hit"] += 1
                    alert_stats[w.name]["tta"].append(float(first_t - eval_times[0]))
            latencies[w.name].append(measure_latency_ms(w, scaled, start, args.latency_samples))

            per_video_rows.append({
                "subject": subject, "video": os.path.basename(video), "true": CLASS_NAMES[true_label],
                "model": w.name, "samples": len(smooth), "accuracy": acc,
                "pct_alert": counts[0], "pct_low_vigilance": counts[1], "pct_drowsy": counts[2],
                "video_verdict": verdict, "alerts_fired": len(edges),
                "first_alert_s": None if first_t is None else float(first_t - eval_times[0]),
            })
        if args.timelines:
            plot_timeline(f"{subject}/{os.path.basename(video)}", true_label, eval_times, preds_for_plot,
                          os.path.join(args.evidence_dir, f"timeline_{subject}_{os.path.splitext(os.path.basename(video))[0]}.png"))

    if not pooled[names[0]]["y"]:
        raise SystemExit("[-] No video produced evaluable samples.")

    # ---- aggregate -------------------------------------------------------
    summary = {}
    for w in wrappers:
        y = np.concatenate(pooled[w.name]["y"])
        p = np.concatenate(pooled[w.name]["p"])
        m = compute_metrics(y, p)
        st = alert_stats[w.name]
        m["false_alarms_per_min"] = (st["fa"] / st["fa_minutes"]) if st["fa_minutes"] > 0 else None
        m["mean_time_to_alert_s"] = float(np.mean(st["tta"])) if st["tta"] else None
        m["drowsy_videos_detected"] = (st["drowsy_hit"] / st["drowsy_total"]) if st["drowsy_total"] else None
        m["latency_ms"] = float(np.nanmean(latencies[w.name]))
        m["size_kb"] = w.size_kb
        summary[w.name] = m

    # ---- print -----------------------------------------------------------
    pv = pd.DataFrame(per_video_rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    show = pv.copy()
    for c in ["accuracy", "pct_alert", "pct_low_vigilance", "pct_drowsy"]:
        show[c] = (show[c] * 100).round(1)
    show["first_alert_s"] = show["first_alert_s"].round(1)
    print("\n" + "=" * 100)
    print("PER-VIDEO RESULTS  (percentages of evaluated samples predicted as each state)")
    print("=" * 100)
    print(show.to_string(index=False))

    flags = best_flags(summary, names)
    print_comparison(summary, names, flags)

    ex = pd.DataFrame(extraction)
    print(f"\nShared feature extraction (MediaPipe, every model pays this once): "
          f"{ex['extract_fps'].mean():.1f} frames/s on average, face found in {ex['face_detect_rate'].mean():.0%} of sampled frames.")
    print("Latency numbers are the classifier alone (single thread, one sample at a time).")
    only_one_class = len({r['true'] for r in per_video_rows}) == 1
    if only_one_class:
        print("[note] All videos have the same true class, so macro-F1 / kappa / per-class recall are not meaningful. "
              "Add videos of other classes (e.g. --subject-dir).")

    # ---- save evidence ---------------------------------------------------
    pv.to_csv(os.path.join(args.evidence_dir, "per_video_results.csv"), index=False)
    flat = pd.DataFrame({m: {k: v for k, v in summary[m].items() if k not in ("per_class", "confusion_matrix")}
                         for m in names})
    flat.to_csv(os.path.join(args.evidence_dir, "comparison_summary.csv"))
    write_latex(summary, names, flags, os.path.join(args.evidence_dir, "comparison_table.tex"))
    plot_confusions(summary, names, os.path.join(args.evidence_dir, "comparison_confusion_matrices.png"))
    with open(os.path.join(args.evidence_dir, "comparison_run.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "args": vars(args),
            "videos": videos,
            "subjects_overlapping_training": sorted({os.path.basename(os.path.dirname(v)) for v in videos} & train_like),
            "summary": summary,
            "extraction": extraction,
            "environment": package_versions(),
        }, f, indent=2, default=lambda o: None if o is None else float(o))
    print(f"\n[+] Evidence written to '{args.evidence_dir}/': comparison_summary.csv, per_video_results.csv, "
          f"comparison_table.tex, comparison_confusion_matrices.png, comparison_run.json")


if __name__ == "__main__":
    main()
