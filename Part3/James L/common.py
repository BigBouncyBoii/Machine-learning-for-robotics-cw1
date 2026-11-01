# CODE MAP of common.py   (shared by main.py, ../FinalMethod.py and every file in extractors/ and classifiers/)
#   Paths          data folder, labels, split, videos, feature caches, results folder
#   Clips          load_clips(): labels.csv + split.csv (never redrawn), test-set fingerprint must be 3696c80d
#   Frames         sample_frames(): decode a clip, keep n evenly spaced frames (not random)
#   Pipeline       make_pipeline(): scaler + classifier; fitted on training clips only
#   Validation     validate(): leave-one-person-out on the 120 TRAIN clips; the 40 test clips are never touched
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting

# %% Paths
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline as _sk_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent                  # Part3/James L
DATA = HERE.parent / "data"                             # Part3/data
LABELS = DATA / "labels.csv"
SPLIT = DATA / "split.csv"
VIDEO_DIR = DATA / "videos_480p"
CACHE_DIR = HERE / "cache"                              # encoder outputs for all 160 clips (no labels), git-ignored
RESULTS = HERE / "results"

FINGERPRINT = "3696c80d"
N_FRAMES = 16          # evenly spaced over the whole clip (clips are 60-269 frames); VideoMAE needs exactly 16


# %% Clips
def load_clips():
    """All 160 clips in labels.csv order, with their set from split.csv.

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


# %% Frames
def sample_frames(path, n=N_FRAMES):
    """Return n evenly spaced RGB frames of the clip, their frame numbers, and the clip length.

    The whole clip is decoded and then indexed, because the frame count stored in
    an mp4 header can be wrong and seeking in h264 is not frame-exact.
    """
    cap = cv2.VideoCapture(str(path))
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ok, frame = cap.read()
    cap.release()
    if len(frames) < n:
        raise ValueError(f"{Path(path).name}: only {len(frames)} frames")
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx], idx, len(frames)


# %% Pipeline
def make_pipeline(classifier):
    """Scaler + classifier, fitted together, so the scaler never sees validation or test clips."""
    return _sk_pipeline(StandardScaler(), classifier)


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
