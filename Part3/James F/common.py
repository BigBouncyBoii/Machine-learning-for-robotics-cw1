# CODE MAP of common.py   (shared by main.py, plots.py and every file in extractors/ and classifiers/)
#   Paths          data folder, labels, split, videos, caches, MediaPipe model, results folder
#   Clips          load_clips(): labels.csv + split.csv (never redrawn), test-set fingerprint must be 3696c80d
#   Skin           skin_mask(): YCrCb skin box, used by both colour extractors
#   Hand tracker   hand_tracker(): pretrained MediaPipe hand landmarker in video mode (nothing is trained here)
#   Pipeline       make_pipeline(): imputer + scaler + classifier; fitted on training clips only
#   Validation     validate(): leave-one-person-out on the 120 TRAIN clips; the 40 test clips are never touched
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting

# %% Paths
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline as _sk_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent                  # Part3/James F
DATA = HERE.parent / "data"                             # Part3/data
LABELS = DATA / "labels.csv"
SPLIT = DATA / "split.csv"
VIDEO_DIR = DATA / "videos_480p"
CACHE_DIR = HERE / "cache"                              # per-frame colours, no labels: safe for all 160 clips
HAND_MODEL = HERE / "models" / "hand_landmarker.task"   # MediaPipe hand landmarker (float16), pretrained
RESULTS = HERE / "results"

FINGERPRINT = "3696c80d"
SEED = 0


# %% Clips
def load_clips():
    """All 160 clips in labels.csv order (the order of every cache), with their set from split.csv.

    Columns: file, person, object, label, y (1 = firm), set ('train' / 'test'), video (path of the 480p copy).
    """
    clips = pd.read_csv(LABELS)[["file", "person", "object", "label"]]
    split = pd.read_csv(SPLIT)
    test_files = split.loc[split["set"] == "test", "file"]
    fp = hashlib.md5("\n".join(sorted(test_files)).encode()).hexdigest()[:8]
    if fp != FINGERPRINT:
        raise SystemExit(f"wrong split: test-set fingerprint {fp}, must be {FINGERPRINT}")
    clips["set"] = clips["file"].map(dict(zip(split["file"], split["set"])))   # match by name, not row order
    assert (clips["set"] == "train").sum() == 120 and (clips["set"] == "test").sum() == 40
    clips["y"] = (clips["label"] == "firm").astype(int)
    clips["video"] = [VIDEO_DIR / f.replace(".MOV", ".mp4") for f in clips["file"]]
    return clips


# %% Skin
CR_MIN, CR_MAX = 135, 180    # skin box in YCrCb: the scene is white wall/table, black/grey object, dark sleeves,
CB_MIN, CB_MAX = 85, 135     # so skin is the only pixel with a clearly red chroma
Y_MIN = 60                   # drop dark shadow pixels, whose chroma is noisy


def skin_mask(bgr, clean=False):
    """Boolean mask of skin pixels; clean=True removes single-pixel noise with a 3x3 opening."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    m = (cr >= CR_MIN) & (cr <= CR_MAX) & (cb >= CB_MIN) & (cb <= CB_MAX) & (y >= Y_MIN)
    if clean:
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    return m


# %% Hand tracker
def hand_tracker():
    """A fresh MediaPipe hand landmarker for one clip (video mode tracks the hand from frame to frame)."""
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.3, min_tracking_confidence=0.3)
    return vision.HandLandmarker.create_from_options(opts)


def track_clip(path, per_frame):
    """Run the tracker over every frame of a clip; per_frame(bgr, landmarks or None) -> one row per frame."""
    import mediapipe as mp
    detector = hand_tracker()
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, i = [], 0
    ok, bgr = cap.read()
    while ok:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / fps))
        rows.append(per_frame(bgr, res.hand_landmarks[0] if res.hand_landmarks else None))
        i += 1
        ok, bgr = cap.read()
    cap.release()
    detector.close()
    return rows


# %% Pipeline
def make_pipeline(classifier):
    """Imputer (frames where MediaPipe lost the hand give NaN) + scaler + classifier, all fitted together."""
    return _sk_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), classifier)


# %% Validation
def validate(X, clips, make_classifier, return_oof=False):
    """Leave-one-person-out on the train clips: train on 3 people, validate on the 4th, for each person.

    X has one row per clip in clips order (all 160); only the train rows are used.
    Returns per-person accuracy and log loss, their means, out-of-fold accuracy per object, and the mean
    training accuracy (how much the model memorises). return_oof=True also returns P(firm) of every train clip,
    each predicted by the model that did not see its person.
    """
    tr = (clips["set"] == "train").values
    Xt, y, person, obj = X[tr], clips["y"].values[tr], clips["person"].values[tr], clips["object"].values[tr]
    people = sorted(set(person))
    p_oof = np.zeros(len(y))
    out, train_acc = {}, []
    for p in people:
        va = person == p
        model = make_pipeline(make_classifier()).fit(Xt[~va], y[~va])
        p_oof[va] = model.predict_proba(Xt[va])[:, 1]
        out[f"val_{p}"] = ((p_oof[va] > 0.5) == y[va]).mean()
        out[f"logloss_{p}"] = log_loss(y[va], p_oof[va], labels=[0, 1])
        train_acc.append(model.score(Xt[~va], y[~va]))
    correct = (p_oof > 0.5) == y
    out["mean_acc"] = np.mean([out[f"val_{p}"] for p in people])
    out["mean_logloss"] = np.mean([out[f"logloss_{p}"] for p in people])
    out["hard_acc"] = correct[obj == "hard"].mean()
    out["soft_acc"] = correct[obj == "soft"].mean()
    out["train_acc"] = np.mean(train_acc)
    return (out, p_oof) if return_oof else out
