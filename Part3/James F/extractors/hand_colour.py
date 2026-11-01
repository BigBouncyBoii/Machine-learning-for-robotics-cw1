# CODE MAP of extractors/hand_colour.py   (tree node 1: whole-hand and nail colour, rest vs hold)
#   Settings        colour thresholds, nail patch size, rest / hold windows
#   Frame colour    per frame: CIELAB of all skin pixels (whole hand) and of the nail discs, minus the white wall
#   Clip curves     every frame of every clip, cached in cache/colour_curves.npz (no labels used)
#   Clip features   rest (hand reaching in) vs hold (object lifted) -> hold value and hold-minus-rest change
#   Feature sets    FEATURE_SETS: the six versions compared in the tree; load() -> {name: X (160, d)}
#   AI use          Claude Opus 5.5 (claude-opus-5-5), default setting
#
# IDEA  Squeezing hard pushes blood out of the fingertips and nail beds, so they go paler / less red
#       (blanching). Lab a* is the red-green axis: a firm grip should lower a* from rest to hold.
#       First version of the idea (was Big_File_of_Big_Dreams.py): one colour per window, no time course.

# %% Settings
import time

import cv2
import numpy as np
import pandas as pd

import common

CACHE = common.CACHE_DIR / "colour_curves.npz"

MIN_SKIN_FRAC = 0.005        # frame counts as "hand visible" if >= 0.5 % of pixels are skin
WALL_ROWS = 0.10             # top 10 % of the frame is always wall: white reference
TIPS, DIPS = [4, 8, 12, 16, 20], [3, 7, 11, 15, 19]   # MediaPipe landmark ids: fingertip, joint below it
NAIL_RADIUS = 0.45           # nail patch radius as a fraction of the tip-DIP distance
REST = (0.00, 0.20)          # fraction of the hand-visible frames: hand reaching in, not gripping yet
HOLD = (0.40, 0.70)          # fraction of the clip: object lifted and held
CHANNELS = ["L", "a", "b"]   # CIELAB: lightness, red(+)/green(-), yellow(+)/blue(-)
COLUMNS = ["skin_frac", "hand_L", "hand_a", "hand_b", "nail_L", "nail_a", "nail_b"]   # one row per frame


# %% Frame colour
def to_lab(bgr):
    """uint8 BGR -> float CIELAB (L 0..100, a and b roughly -128..127)."""
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def nail_mask(shape, landmarks):
    """Union of one disc per finger, centred halfway between the tip and the joint below (the nail)."""
    h, w = shape[:2]
    p = np.array([[l.x * w, l.y * h] for l in landmarks])
    m = np.zeros((h, w), np.uint8)
    for t, d in zip(TIPS, DIPS):
        c = 0.5 * (p[t] + p[d])
        r = max(int(NAIL_RADIUS * np.linalg.norm(p[t] - p[d])), 2)
        cv2.circle(m, (int(c[0]), int(c[1])), r, 1, -1)
    return m.astype(bool)


def frame_colour(bgr, landmarks):
    """One frame -> [skin_frac, hand L a b, nail L a b]; colours relative to the wall, NaN if not seen."""
    lab = to_lab(bgr)
    wall = np.median(lab[: int(WALL_ROWS * lab.shape[0])].reshape(-1, 3), axis=0)
    skin = common.skin_mask(bgr, clean=True)
    out = np.full(7, np.nan, np.float32)
    out[0] = skin.mean()
    if out[0] >= MIN_SKIN_FRAC:
        out[1:4] = lab[skin].mean(0) - wall
    if landmarks is not None:
        nail = nail_mask(bgr.shape, landmarks) & skin       # only skin pixels inside the discs
        if nail.sum() >= 20:
            out[4:7] = lab[nail].mean(0) - wall
    return out


# %% Clip curves
def load_curves(clips):
    """List of (n_frames, 7) arrays, one per clip in clips order; built once (~5 min on CPU), then cached."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        assert list(z["clip_file"]) == list(clips["file"]), "cache is from a different labels.csv; delete it"
        return np.split(z["frames"], np.cumsum(z["n_frames"])[:-1])
    curves, t0 = [], time.time()
    for k, path in enumerate(clips["video"]):
        curves.append(np.array(common.track_clip(path, frame_colour)))
        if (k + 1) % 20 == 0 or k + 1 == len(clips):
            print(f"{k + 1}/{len(clips)} clips  ({time.time() - t0:.0f} s)")
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, frames=np.concatenate(curves), n_frames=[len(c) for c in curves],
                        clip_file=clips["file"].values)
    return curves


# %% Clip features
def window_mean(c, lo, hi):
    """nanmean of rows lo..hi (fractions of len(c)) of a curve; NaN if the window has no values."""
    seg = c[int(lo * len(c)): max(int(hi * len(c)), int(lo * len(c)) + 1)]
    return np.nanmean(seg, axis=0) if np.isfinite(seg).any() else np.full(c.shape[1], np.nan)


def clip_features(c):
    """(n_frames, 7) curve -> dict of hold and hold-minus-rest values for hand and nail colour."""
    visible = c[c[:, 0] >= MIN_SKIN_FRAC]                  # frames where the hand is in view
    rest = window_mean(visible, *REST) if len(visible) else np.full(7, np.nan)
    hold = window_mean(c, *HOLD)
    f = {}
    for part, cols in (("hand", slice(1, 4)), ("nail", slice(4, 7))):
        for ch, r, h in zip(CHANNELS, rest[cols], hold[cols]):
            f[f"{part}_{ch}_hold"] = h
            f[f"{part}_{ch}_delta"] = h - r
    return f


# %% Feature sets
FEATURE_SETS = {
    "hand_a_hold":          ["hand_a_hold"],                                  # how red the hand is while holding
    "hand_a_change":        ["hand_a_delta"],                                 # ... relative to its own rest colour
    "hand_lab_change":      [f"hand_{c}_delta" for c in CHANNELS],
    "nail_a_change":        ["nail_a_delta"],                                 # nails only (MediaPipe discs)
    "nail_lab_change":      [f"nail_{c}_delta" for c in CHANNELS],
    "hand_nail_lab_change": [f"{p}_{c}_delta" for p in ("hand", "nail") for c in CHANNELS],
}


def load(clips):
    """{feature set name: X (n_clips, d)}, rows in clips order."""
    table = pd.DataFrame([clip_features(c) for c in load_curves(clips)])
    return {name: table[cols].to_numpy(np.float64) for name, cols in FEATURE_SETS.items()}
