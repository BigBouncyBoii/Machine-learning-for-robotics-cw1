# CODE MAP of extractors/joint_angles.py   (tree node: finger joint angles from MediaPipe hand landmarks)
#   Settings        frame step, MediaPipe model (downloaded once), cache file
#   Joint triples   which 3 landmarks define each of the 15 finger joint angles
#   Angle features  per frame: 15 angles; per clip: mean, std, min of each angle + hand-detection rate = 46
#   Hand model      download the official MediaPipe hand landmarker once; image mode, 1 hand
#   Cache           every clip once (no labels used) -> cache/joint_angles.npz
#   Feature sets    "joint_angles": the 46 features; load() -> {name: X (160, 46)}
#   AI use          Claude Opus 5.5 (claude-opus-5-5), default setting (restructure); original code: <Jeevan: model, version, setting>
#
# IDEA  A firm grip bends the fingers more and holds them more steadily, which should be visible in the pose.
#       (was FingerAnglesJeevan.py)

# %% Settings
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import cv2
import numpy as np
import pandas as pd

import common

CACHE = common.CACHE_DIR / "joint_angles.npz"
MODEL_PATH = Path.home() / ".cache" / "mediapipe" / "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
FRAME_STEP = 2               # use every 2nd frame

# %% Joint triples
# MediaPipe landmarks: 0 wrist; thumb 1-4; index 5-8; middle 9-12; ring 13-16; pinky 17-20.
# Each triple (a, b, c) gives the angle at b between segments b->a and b->c.
# 180 degrees = straight finger, smaller = more bent.
TRIPLES = []
for base in (1, 5, 9, 13, 17):                      # thumb + 4 fingers
    TRIPLES += [(0, base, base + 1),                # knuckle (MCP / CMC)
                (base, base + 1, base + 2),         # middle joint (PIP / MCP)
                (base + 1, base + 2, base + 3)]     # upper joint (DIP / IP)
N_ANG = len(TRIPLES)  # 15


# %% Angle features
def frame_angles(pts):
    """pts: (21, 3) landmarks -> (15,) joint angles in degrees."""
    out = np.empty(N_ANG)
    for i, (a, b, c) in enumerate(TRIPLES):
        v1, v2 = pts[a] - pts[b], pts[c] - pts[b]
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        out[i] = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    return out


def summarize_video_angles(angles, detection_rate):
    """Mean, std and min of each angle over the clip, plus the detection rate; zeros if no hand was found."""
    if angles:
        values = np.asarray(angles)
        summary = np.concatenate([values.mean(0), values.std(0), values.min(0)])
    else:
        summary = np.zeros(N_ANG * 3)
    return np.concatenate([summary, [detection_rate]])


# %% Hand model
def ensure_hand_model():
    """Download the official MediaPipe Tasks model once if it is not cached."""
    if MODEL_PATH.is_file():
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".task.download")
    try:
        with urlopen(MODEL_URL, timeout=60) as response, tmp.open("wb") as model:
            while chunk := response.read(1024 * 1024):
                model.write(chunk)
        tmp.replace(MODEL_PATH)
    except (OSError, URLError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download the MediaPipe hand-landmarker model: {MODEL_URL}") from exc


def hand_landmarker():
    """One image-mode detector (each frame on its own), reused across all clips."""
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    ensure_hand_model()
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE, num_hands=1, min_hand_detection_confidence=0.5)
    return mp_vision.HandLandmarker.create_from_options(options)


def video_features(path, hands):
    """One clip -> 46 features and the hand-detection rate (fraction of the used frames with a hand)."""
    import mediapipe as mp
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {path}")
    angs, n_seen, i = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % FRAME_STEP == 0:
            n_seen += 1
            res = hands.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if res.hand_landmarks:
                angs.append(frame_angles(np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]])))
        i += 1
    cap.release()
    rate = len(angs) / max(n_seen, 1)
    return summarize_video_angles(angs, rate), rate


# %% Cache
def load_cache(clips):
    """X (160, 46) and the hand-detection rate per clip, in clips order; built on the first call (~10 min CPU)."""
    if not CACHE.exists():
        hands, rows, X = hand_landmarker(), [], []
        for k, (file, video) in enumerate(zip(clips["file"], clips["video"])):
            f, rate = video_features(video, hands)
            rows.append({"file": file, "hand_detection_rate": rate})
            X.append(f)
            if (k + 1) % 20 == 0 or k + 1 == len(clips):
                print(f"  joint angles: {k + 1}/{len(clips)} clips")
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, meta=np.array(rows, dtype=object), X=np.array(X))
    d = np.load(CACHE, allow_pickle=True)
    meta = pd.DataFrame(d["meta"].tolist())
    row = {f: i for i, f in enumerate(meta["file"])}
    order = [row[f] for f in clips["file"]]                  # match by file name, not by row order
    return d["X"][order], meta["hand_detection_rate"].to_numpy()[order]


# %% Feature sets
FEATURE_SETS = ["joint_angles"]


def load(clips, names=FEATURE_SETS):
    """{feature set name: X (n_clips, 46)}, rows in clips order."""
    X, _ = load_cache(clips)
    return {"joint_angles": X}
