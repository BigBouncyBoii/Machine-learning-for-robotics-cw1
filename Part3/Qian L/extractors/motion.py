# CODE MAP of extractors/motion.py   (shared by speed.py, shaking.py and both.py; not a tree node itself)
#   Settings        frame size, fixed hand/object and background regions, skin thresholds, phase rules
#   Decode          every frame: Farneback optical flow, skin pixels in the region -> one motion sample per frame pair
#   Cache           motion series of every clip (no labels used) -> cache/motion_series.json.gz
#   Phase           phase(): the frame pairs inside a time window; enough_valid(): is that window usable?
#   AI use          GPT-6.1 Sol (setting: medium); GPT-6 Astra (setting: light) (original code);; GitHub Copilot (VS Code; extension version, model/version and setting unavailable)
#                   Claude Opus 5.5 (claude-opus-5-5), default setting (restructure)
#
# IDEA  A firm grip may close the fingers faster and make the hand shake more under strain. Both are motion,
#       so every frame pair gives one optical-flow sample; speed.py and shaking.py summarise different windows.
#       No landmarks and no MediaPipe: a fixed region and a skin-colour threshold (a coarse heuristic).
#       (was features.py)

# %% Settings
import gzip
import json
import time

import cv2
import numpy as np

import common

CACHE = common.CACHE_DIR / "motion_series.json.gz"
LONG_EDGE = 320                       # every frame is resized so its long edge is 320 px
ROI = (0.20, 0.30, 0.80, 0.78)        # hand/object region (x0, y0, x1, y1 as fractions of the frame)
BACKGROUND = (0.05, 0.08, 0.95, 0.23) # top band of wall: camera motion reference
SKIN_Y_MIN, SKIN_CR, SKIN_CB = 35, (133, 180), (70, 135)   # YCrCb skin box
MIN_SKIN_PIXELS = 30                  # a frame pair needs this many skin pixels in the region
MIN_PHASE_PAIRS = 8                   # a window needs >= 8 valid frame pairs ...
MIN_PHASE_FRACTION = 0.7              # ... and >= 70 % of its pairs valid, else its features are NaN


def box(shape, bounds):
    h, w = shape[:2]
    x0, y0, x1, y1 = bounds
    return round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)


# %% Decode
def decode_motion(path):
    """One clip -> list of per-frame-pair motion samples, fps, number of frames."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise OSError(f"Cannot decode {path} or invalid FPS")
    samples = []
    previous = previous_skin = None
    n_frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = LONG_EDGE / max(h, w)
            image = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            skin = ((ycc[:, :, 0] > SKIN_Y_MIN) &
                    (ycc[:, :, 1] >= SKIN_CR[0]) & (ycc[:, :, 1] <= SKIN_CR[1]) &
                    (ycc[:, :, 2] >= SKIN_CB[0]) & (ycc[:, :, 2] <= SKIN_CB[1]))
            if previous is not None:
                x0, y0, x1, y1 = box(gray.shape, ROI)
                bx0, by0, bx1, by1 = box(gray.shape, BACKGROUND)
                flow = cv2.calcOpticalFlowFarneback(previous, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
                mask = previous_skin[y0:y1, x0:x1]
                sample = dict(frame=n_frames, skin_pixels=int(mask.sum()), valid=False)
                if mask.sum() >= MIN_SKIN_PIXELS:
                    vectors = flow[y0:y1, x0:x1][mask].astype(float)
                    bg = np.median(flow[by0:by1, bx0:bx1].reshape(-1, 2), axis=0)
                    hand = np.median(vectors, axis=0)
                    # Local motion relative to whole-hand translation: a curl-speed proxy.
                    local = vectors - hand
                    diagonal = float(np.hypot(y1 - y0, x1 - x0))
                    local_speed = np.linalg.norm(local, axis=1) * fps / diagonal
                    xy = np.argwhere(mask)[:, ::-1].astype(float)
                    inward_direction = xy.mean(axis=0) - xy
                    inward_direction /= np.maximum(np.linalg.norm(inward_direction, axis=1, keepdims=True), 1.0)
                    inward = np.maximum(np.sum(local * inward_direction, axis=1), 0) * fps / diagonal
                    sample.update(valid=True, local_speed_mean=float(local_speed.mean()),
                                  local_speed_p90=float(np.percentile(local_speed, 90)),
                                  inward_speed_mean=float(inward.mean()),
                                  inward_speed_p90=float(np.percentile(inward, 90)),
                                  hand_velocity=((hand - bg) * fps / diagonal).tolist())
                samples.append(sample)
            previous, previous_skin = gray, skin
            n_frames += 1
    finally:
        cap.release()
    if n_frames < 12:
        raise ValueError(f"{path}: too few decoded frames")
    return samples, fps, n_frames


# %% Cache
def load_series(clips):
    """{file: {'samples', 'fps', 'frame_count'}} for every clip; clips not yet cached are decoded (~3 s each)."""
    cache = json.loads(gzip.decompress(CACHE.read_bytes())) if CACHE.exists() else {}
    missing = [(f, v) for f, v in zip(clips["file"], clips["video"]) if f not in cache]
    t0 = time.time()
    for k, (file, video) in enumerate(missing):
        samples, fps, n = decode_motion(video)
        cache[file] = dict(samples=samples, fps=fps, frame_count=n)
        if (k + 1) % 20 == 0 or k + 1 == len(missing):
            print(f"  optical flow: {k + 1}/{len(missing)} clips ({time.time() - t0:.0f} s)")
    if missing:
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_bytes(gzip.compress(json.dumps(cache, allow_nan=False).encode()))
    return cache


# %% Phase
def phase(samples, frame_count, window):
    """The frame pairs whose BOTH frames lie inside window (fractions of the clip, e.g. (0.10, 0.45))."""
    lo, hi = window
    return [s for s in samples if lo <= (s["frame"] - 1) / (frame_count - 1) and s["frame"] / (frame_count - 1) <= hi]


def enough_valid(selected):
    """True if the window has enough frame pairs with visible skin to be summarised."""
    valid = [s for s in selected if s["valid"]]
    fraction = len(valid) / len(selected) if selected else 0
    return len(valid) >= MIN_PHASE_PAIRS and fraction >= MIN_PHASE_FRACTION


def table(clips, window_features, names):
    """X (n_clips, len(names)) in clips order; window_features(series) -> dict or None (row of NaN)."""
    cache = load_series(clips)
    rows = []
    for f in clips["file"]:
        values = window_features(cache[f])
        rows.append([values[n] for n in names] if values is not None else [np.nan] * len(names))
    return np.asarray(rows, dtype=float)
