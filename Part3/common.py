# CODE MAP of common.py
#   Paths          data, videos, labels, split and feature files
#   Frames         sample_frames: decode a clip, keep N_FRAMES evenly spaced frames (not random)
#   Features       FEATURES: every feature version the tree compares (encoder + how to pool it to one vector per clip)
#   Load           load_features: one vector per clip + person/object/label/set, read from the cached .npz
#   Classifiers    CLASSIFIERS: logistic regression and MLP, each a scaler + classifier pipeline
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting
#
# Shared by extract_features.py, explore.py and predict_clip.py, so every script samples and pools the same way.
# To add a tree node that only changes the pooling, add one line to FEATURES; nothing needs re-extracting.

# %% Paths
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
VIDEO_DIR = HERE / "data" / "videos_480p"
LABELS = HERE / "data" / "labels.csv"
SPLIT = HERE / "data" / "split.csv"
FEATURE_DIR = HERE / "features"

N_FRAMES = 16          # evenly spaced over the whole clip (clips are 60-269 frames); VideoMAE needs exactly 16


def video_path(file):
    """labels.csv name (e.g. 'A/A_hard_firm_01.MOV') -> path of its 480p copy."""
    return VIDEO_DIR / file.replace(".MOV", ".mp4")


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


# %% Features
# name -> (encoder in encoders/, cached file features/<file>.npz, pooling: arrays of the file -> (n_clips, dim))
FEATURES = {
    "dinov2_cls_mean":   ("dinov2", "dinov2_small", lambda f: f["cls"].mean(axis=1)),     # mean of the 16 frames' CLS
    "dinov2_patch_mean": ("dinov2", "dinov2_small", lambda f: f["patch"].mean(axis=1)),   # mean of the 16 frames' patch means
    # fewer frames (subsets of the cached 16, evenly spaced): does averaging more frames help?
    "dinov2_1frame":     ("dinov2", "dinov2_small", lambda f: f["cls"][:, 8]),            # middle frame only
    "dinov2_4frames":    ("dinov2", "dinov2_small", lambda f: f["cls"][:, ::4].mean(axis=1)),   # frames 0, 4, 8, 12
    "dinov2_8frames":    ("dinov2", "dinov2_small", lambda f: f["cls"][:, ::2].mean(axis=1)),   # every 2nd frame
    # one phase of the grip: where is the signal?
    "dinov2_start":      ("dinov2", "dinov2_small", lambda f: f["cls"][:, :3].mean(axis=1)),    # first 3 frames (hand resting / approaching)
    "dinov2_middle":     ("dinov2", "dinov2_small", lambda f: f["cls"][:, 5:11].mean(axis=1)),  # middle 6 frames (squeeze / hold)
    "videomae":          ("videomae", "videomae_base", lambda f: f["vec"]),               # one vector for the whole clip
}


# %% Load
def load_features(name):
    """Return X (n_clips, dim) and a DataFrame with file, person, object, label, set (same row order)."""
    _, file, pool = FEATURES[name]
    feat = np.load(FEATURE_DIR / f"{file}.npz")
    meta = pd.DataFrame({k: feat[k] for k in ["clip", "person", "object", "label"]}).rename(columns={"clip": "file"})
    split = pd.read_csv(SPLIT).set_index("file")
    meta["set"] = split.loc[meta["file"], "set"].values      # match by file name, not by row order
    return pool(feat), meta


# %% Classifiers
# name -> function(seed) returning a fresh, unfitted scaler + classifier. The scaler is part of the pipeline,
# so it is always fitted on the same clips as the classifier (never on validation or test clips).
CLASSIFIERS = {
    # linear, strong L2 penalty (smaller C = stronger); no randomness, the seed is ignored
    "lr": lambda seed: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
    # 1 hidden layer of 64 ReLU units, L2 penalty alpha; result depends on the random initial weights
    "mlp": lambda seed: make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(64,), alpha=1e-2, max_iter=500, random_state=seed)),
}
SEEDS = {"lr": [0], "mlp": list(range(5))}   # MLP: average over 5 seeds
