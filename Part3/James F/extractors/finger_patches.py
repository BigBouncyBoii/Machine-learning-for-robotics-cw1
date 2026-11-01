# CODE MAP of extractors/finger_patches.py   (tree nodes 2+: colour of each finger over time)
#   Settings         patch size, time bins, landmark ids
#   Finger patches   MediaPipe landmarks -> 10 discs per frame: nail and middle segment of each finger
#   Patch colour     per frame: CIELAB of the skin pixels in each disc, minus the white wall
#   Clip curves      every frame of every clip, cached in cache/finger_patches.npz (no labels used)
#   Change curves    per clip: colour change from rest, averaged into 10 time bins -> (10 bins, 10 patches, 3)
#   Feature sets     FEATURE_SETS: the four inputs compared in the tree; load() -> {name: X (160, d)}
#   AI use           Claude Opus 5.5 (claude-opus-5-5), default setting
#
# IDEA  Same blanching idea as hand_colour.py, but per finger and as a time course: the whole hand mixes
#       fingers that press with ones that do not, and one hold window throws the timing away.
#       MediaPipe only says WHERE the fingers are; the classifier only sees colour numbers.
#       (was colour_mlp.py and colour_method/colour_method.py)

# %% Settings
import time
import warnings

import cv2
import numpy as np

import common

CACHE = common.CACHE_DIR / "finger_patches.npz"

WALL_ROWS = 0.10             # top 10 % of the frame is always wall: white reference
MIN_PIXELS = 15              # a patch needs this many skin pixels, else NaN
FINGERS = ["thumb", "index", "middle", "ring", "little"]
TIPS = [4, 8, 12, 16, 20]    # MediaPipe landmark ids
DIPS = [3, 7, 11, 15, 19]    # joint below the tip (IP joint for the thumb)
PIPS = [2, 6, 10, 14, 18]    # next joint down (MCP for the thumb)
REGIONS = [f"{f}_nail" for f in FINGERS] + [f"{f}_mid" for f in FINGERS]
CHANNELS = ["L", "a", "b"]
PATCH_RADIUS = 0.45          # disc radius as a fraction of the segment length
N_BINS = 10                  # each clip is cut into 10 equal time bins (clips are 2.5-3.5 s)
REST_BINS = 2                # first 2 bins = hand reaching in, before the grip: the reference colour
NAIL, MID, A = slice(0, 5), slice(5, 10), CHANNELS.index("a")

# nanmean over a patch/bin where MediaPipe never found the hand gives NaN (wanted; the imputer fills it)
warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)


# %% Finger patches
def patch_centres(p):
    """(21, 2) landmark pixels -> (10, 2) disc centres and (10,) radii: 5 nails, then 5 middle segments."""
    c, r = [], []
    for a, b in list(zip(TIPS, DIPS)) + list(zip(DIPS, PIPS)):
        c.append(0.5 * (p[a] + p[b]))
        r.append(max(PATCH_RADIUS * np.linalg.norm(p[a] - p[b]), 2.0))
    return np.array(c), np.array(r)


# %% Patch colour
def patch_colour(bgr, p):
    """One frame and its (21, 2) landmarks -> (10, 3) Lab colour of each patch minus the wall colour."""
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    wall = np.median(lab[: int(WALL_ROWS * lab.shape[0])].reshape(-1, 3), axis=0)
    skin = common.skin_mask(bgr)
    h, w = skin.shape
    yy, xx = np.ogrid[:h, :w]
    out = np.full((len(REGIONS), 3), np.nan, np.float32)
    for k, (c, r) in enumerate(zip(*patch_centres(p))):
        x0, x1 = int(max(c[0] - r, 0)), int(min(c[0] + r + 1, w))   # work in the disc's bounding box only
        y0, y1 = int(max(c[1] - r, 0)), int(min(c[1] + r + 1, h))
        if x1 <= x0 or y1 <= y0:
            continue
        disc = (xx[:, x0:x1] - c[0]) ** 2 + (yy[y0:y1] - c[1]) ** 2 <= r ** 2
        m = disc & skin[y0:y1, x0:x1]
        if m.sum() >= MIN_PIXELS:
            out[k] = lab[y0:y1, x0:x1][m].mean(0) - wall
    return out


def frame_row(bgr, landmarks):
    """One frame -> (patch colours (10, 3), landmark pixels (21, 2)); NaN where no hand was found."""
    if landmarks is None:
        return np.full((len(REGIONS), 3), np.nan, np.float32), np.full((21, 2), np.nan, np.float32)
    p = np.array([[l.x * bgr.shape[1], l.y * bgr.shape[0]] for l in landmarks], np.float32)
    return patch_colour(bgr, p), p


# %% Clip curves
def load_curves(clips):
    """Per clip: colours (n_frames, 10, 3) and landmarks (n_frames, 21, 2); built once (~10 min), then cached."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        assert list(z["clip_file"]) == list(clips["file"]), "cache is from a different labels.csv; delete it"
        cuts = np.cumsum(z["n_frames"])[:-1]
        return np.split(z["colours"], cuts), np.split(z["landmarks"], cuts)
    colours, landmarks, t0 = [], [], time.time()
    for k, path in enumerate(clips["video"]):
        rows = common.track_clip(path, frame_row)
        colours.append(np.array([r[0] for r in rows]))
        landmarks.append(np.array([r[1] for r in rows]))
        if (k + 1) % 20 == 0 or k + 1 == len(clips):
            print(f"{k + 1}/{len(clips)} clips  ({time.time() - t0:.0f} s)")
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, colours=np.concatenate(colours), landmarks=np.concatenate(landmarks),
                        n_frames=[len(c) for c in colours], clip_file=clips["file"].values)
    return colours, landmarks


# %% Change curves
def binned(c):
    """(n_frames, 10, 3) -> (N_BINS, 10, 3): nanmean of each equal time bin (NaN if the hand was never seen)."""
    return np.array([np.nanmean(seg, axis=0) for seg in np.array_split(c, N_BINS)], np.float32)


def change_curves(c):
    """Colour change from rest: binned colour minus the mean of the first REST_BINS bins that have data."""
    b = binned(c)
    rest = np.nanmean(b[:REST_BINS], axis=0)
    rest = np.where(np.isfinite(rest), rest, np.nanmean(b, axis=0))   # hand not seen early: use clip mean
    return b - rest                                    # (N_BINS, 10, 3), negative a* = paler than at rest


def load_change_curves(clips):
    """(n_clips, N_BINS, 10, 3) change curves and the fraction of frames where the hand was found, per clip."""
    colours, landmarks = load_curves(clips)
    curves = np.array([change_curves(c) for c in colours])
    found = np.array([np.isfinite(l[:, 0, 0]).mean() for l in landmarks])
    return curves, found


# %% Feature sets
FEATURE_SETS = {
    "finger_a_hold":       "one number per patch: a* change in the middle of the clip (bins 4-6)",
    "nail_mid_a_curve":    "time course of a*, averaged over the 5 nails and over the 5 middle segments",
    "nail_mid_lab_curves": "time course of L, a*, b* for nails and middle segments",
    "all_patches":         "every patch, every channel, every bin (300 numbers per clip)",
}


def load(clips):
    """{feature set name: X (n_clips, d)}, rows in clips order."""
    curves, _ = load_change_curves(clips)
    n = len(curves)
    return {
        "finger_a_hold":       curves[:, 4:7, :, A].mean(1),
        "nail_mid_a_curve":    np.concatenate([np.nanmean(curves[:, :, NAIL, A], 2),
                                               np.nanmean(curves[:, :, MID, A], 2)], 1),
        "nail_mid_lab_curves": np.concatenate([np.nanmean(curves[:, :, NAIL, :], 2).reshape(n, -1),
                                               np.nanmean(curves[:, :, MID, :], 2).reshape(n, -1)], 1),
        "all_patches":         curves.reshape(n, -1),
    }
