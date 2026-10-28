# CODE MAP of Big_File_of_Big_Dreams.py   (finger-colour branch of the Open Exploration)
#   Settings        paths, test person, colour thresholds, time windows
#   Load labels     read Part3/data/labels.csv, mark train / test by person
#   Frame colour    per frame: skin mask -> CIELAB colour of the whole hand and of the nail patches,
#                   both relative to the white wall at the top of the frame (cancels auto white balance)
#   Clip curves     run Frame colour on every clip once and cache the per-frame curves
#   Clip features   rest (start of clip, hand not yet gripping) vs hold (middle of clip) -> one vector per grip
#   Validation      leave-one-person-out on the training people only, logistic regression, per feature set
#   Test            train the chosen feature set on all training people, evaluate once on the test person
#   Plots           redness over normalised time (light vs firm), per-clip change in a* by object
#   AI use          Claude Opus 5.5 (claude-opus-5-5), default setting
#
# IDEA  Squeezing hard pushes blood out of the fingertips and nail beds, so they go paler / less red
#       (blanching). Lab a* is the red-green axis: a firm grip should lower a* from rest to hold.
#
# HOW TO RUN (from the repo root)
#   pip install opencv-python-headless mediapipe numpy pandas matplotlib scikit-learn
#   python "James F/Big_File_of_Big_Dreams.py"
#   The first run decodes all 160 clips (a few minutes on CPU) and writes James F/cache/colour_curves.npz;
#   later runs only reload that file. Delete it to recompute.

# %% Settings
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np  
import pandas as pd
from mediapipe.tasks.python import BaseOptions, vision
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
PART3 = HERE.parent / "Part3" / "data"
VIDEO_DIR = PART3 / "videos_480p"
LABELS = PART3 / "labels.csv"
HAND_MODEL = HERE / "models" / "hand_landmarker.task"   # MediaPipe hand landmarker (float16)
CACHE = HERE / "cache" / "colour_curves.npz"
FIG_DIR = HERE / "figures"

TEST_PERSON = "D"            # all clips of one member = 25 % test set, never used for training (rule 2)
SEED = 0

# Skin mask in YCrCb: the scene is white wall/table, black/grey object, dark sleeves,
# so skin is the only pixel with a clearly red chroma.
CR_MIN, CR_MAX = 135, 180
CB_MIN, CB_MAX = 85, 135
Y_MIN = 60                   # drop dark shadow pixels, whose chroma is noisy
MIN_SKIN_FRAC = 0.005        # frame counts as "hand visible" if >= 0.5 % of pixels are skin
WALL_ROWS = 0.10             # top 10 % of the frame is always wall: white reference

TIPS, DIPS = [4, 8, 12, 16, 20], [3, 7, 11, 15, 19]   # MediaPipe landmark ids: fingertip, joint below it
NAIL_RADIUS = 0.45           # nail patch radius as a fraction of the tip-DIP distance

REST = (0.00, 0.20)          # fraction of the hand-visible frames: hand reaching in, not gripping yet
HOLD = (0.40, 0.70)          # fraction of the clip: object lifted and held (see the frame strips)

CHANNELS = ["L", "a", "b"]   # CIELAB: lightness, red(+)/green(-), yellow(+)/blue(-)

# %% Load labels
df = pd.read_csv(LABELS)
df["video"] = [VIDEO_DIR / f.replace(".MOV", ".mp4") for f in df["file"]]
df["y"] = (df["label"] == "firm").astype(int)
df["test"] = df["person"] == TEST_PERSON
missing = [str(v) for v in df["video"] if not v.exists()]
if missing:
    raise FileNotFoundError(f"{len(missing)} clips missing, e.g. {missing[0]}")


# %% Frame colour
def skin_mask(bgr):
    """Boolean mask of skin pixels (YCrCb box), lightly cleaned with a 3x3 opening."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    m = (cr >= CR_MIN) & (cr <= CR_MAX) & (cb >= CB_MIN) & (cb <= CB_MAX) & (y >= Y_MIN)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m.astype(bool)


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
    """One frame -> [skin_frac, hand L a b, nail L a b], colours relative to the wall.

    Hand and nail values are NaN when there is no hand (or no detected landmarks).
    """
    lab = to_lab(bgr)
    wall = np.median(lab[: int(WALL_ROWS * lab.shape[0])].reshape(-1, 3), axis=0)
    skin = skin_mask(bgr)
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
def clip_curves(path, detector_options):
    """Every frame of one clip -> (n_frames, 7) array of frame_colour values."""
    detector = vision.HandLandmarker.create_from_options(detector_options)   # fresh tracker per clip
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, i = [], 0
    ok, bgr = cap.read()
    while ok:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / fps))
        rows.append(frame_colour(bgr, res.hand_landmarks[0] if res.hand_landmarks else None))
        i += 1
        ok, bgr = cap.read()
    cap.release()
    detector.close()
    return np.array(rows)


if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True)
    assert list(z["clip_file"]) == list(df["file"]), "cache is from a different labels.csv; delete it"
    curves = np.split(z["frames"], np.cumsum(z["n_frames"])[:-1])
    print(f"loaded {CACHE.name}")
else:
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.3, min_tracking_confidence=0.3)
    curves, t0 = [], time.time()
    for k, path in enumerate(df["video"]):
        curves.append(clip_curves(path, opts))
        if (k + 1) % 20 == 0 or k + 1 == len(df):
            print(f"{k + 1}/{len(df)} clips  ({time.time() - t0:.0f} s)")
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, frames=np.concatenate(curves), n_frames=[len(c) for c in curves],
                        clip_file=df["file"].values)


# %% Clip features
def window_mean(c, lo, hi):
    """nanmean of rows lo..hi (fractions of len(c)) of a curve; NaN if the window has no values."""
    seg = c[int(lo * len(c)): max(int(hi * len(c)), int(lo * len(c)) + 1)]
    return np.nanmean(seg, axis=0) if np.isfinite(seg).any() else np.full(c.shape[1], np.nan)


def clip_features(c):
    """(n_frames, 7) curve -> dict of rest, hold and hold-minus-rest values for hand and nail colour."""
    visible = c[c[:, 0] >= MIN_SKIN_FRAC]                  # frames where the hand is in view
    rest = window_mean(visible, *REST) if len(visible) else np.full(7, np.nan)
    hold = window_mean(c, *HOLD)
    f = {}
    for part, cols in (("hand", slice(1, 4)), ("nail", slice(4, 7))):
        for ch, r, h in zip(CHANNELS, rest[cols], hold[cols]):
            f[f"{part}_{ch}_hold"] = h
            f[f"{part}_{ch}_delta"] = h - r
    f["nail_found"] = np.isfinite(c[:, 4]).mean()          # fraction of frames with nail patches
    return f


feat = pd.DataFrame([clip_features(c) for c in curves])
data = pd.concat([df[["file", "person", "object", "label", "y", "test"]], feat], axis=1)

# Each feature set is one node of the exploration tree.
FEATURE_SETS = {
    "hand a* hold":            ["hand_a_hold"],
    "hand a* change":          ["hand_a_delta"],
    "hand Lab change":         [f"hand_{c}_delta" for c in CHANNELS],
    "nail a* change":          ["nail_a_delta"],
    "nail Lab change":         [f"nail_{c}_delta" for c in CHANNELS],
    "hand + nail Lab change":  [f"{p}_{c}_delta" for p in ("hand", "nail") for c in CHANNELS],
}


# %% Validation
def make_model():
    # Imputer and scaler are fitted inside the pipeline, so on training clips only.
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=1.0, random_state=SEED))


train = data[~data["test"]].reset_index(drop=True)
people = sorted(train["person"].unique())
print(f"\ntrain people {people}, test person {TEST_PERSON}; "
      f"{len(train)} train clips, {int(data['test'].sum())} test clips")
print(f"nail patches found in {100 * train['nail_found'].mean():.0f} % of frames (train clips)\n")

print("Leave-one-person-out validation accuracy (guessing = 50 %)")
print(f"{'feature set':<26}" + "".join(f"{'val ' + p:>8}" for p in people) + f"{'mean':>8}"
      + f"{'hard':>8}{'soft':>8}")
val_results = {}
for name, cols in FEATURE_SETS.items():
    pred = np.zeros(len(train), int)
    for p in people:
        tr, va = train["person"] != p, train["person"] == p
        pred[va] = make_model().fit(train.loc[tr, cols], train.loc[tr, "y"]).predict(train.loc[va, cols])
    correct = pred == train["y"]
    per_person = [correct[train["person"] == p].mean() for p in people]
    per_object = [correct[train["object"] == o].mean() for o in ("hard", "soft")]
    val_results[name] = np.mean(per_person)
    print(f"{name:<26}" + "".join(f"{100 * a:>7.0f}%" for a in per_person)
          + f"{100 * np.mean(per_person):>7.0f}%" + "".join(f"{100 * a:>7.0f}%" for a in per_object))

# %% Test
best = max(val_results, key=val_results.get)       # chosen on validation only
cols = FEATURE_SETS[best]
test = data[data["test"]]
model = make_model().fit(train[cols], train["y"])
test_acc = (model.predict(test[cols]) == test["y"]).mean()
n_correct = int(round(test_acc * len(test)))
print(f"\nchosen on validation: '{best}' (val {100 * val_results[best]:.0f} %)")
print(f"test person {TEST_PERSON}: {100 * test_acc:.1f} % ({n_correct}/{len(test)}), guessing = 50 %")
lr = model[-1]
print("logistic regression weights (standardised): "
      + ", ".join(f"{c} {w:+.2f}" for c, w in zip(cols, lr.coef_[0])))

# %% Plots
FIG_DIR.mkdir(exist_ok=True)
grid = np.linspace(0, 1, 50)


def resampled(c, col):
    """Curve column on a 0..1 time axis, so clips of different lengths can be averaged."""
    t = np.linspace(0, 1, len(c))
    ok = np.isfinite(c[:, col])
    return np.interp(grid, t[ok], c[ok, col]) if ok.sum() > 1 else np.full(len(grid), np.nan)


fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for row, (part, col) in enumerate((("hand", 2), ("nail", 5))):
    for ax, obj in zip(axes[row], ("hard", "soft")):
        for lab, colour in (("light", "tab:blue"), ("firm", "tab:red")):
            idx = data.index[(~data["test"]) & (data["object"] == obj) & (data["label"] == lab)]
            m = np.nanmean([resampled(curves[i], col) for i in idx], axis=0)
            ax.plot(grid, m, color=colour, label=f"{lab} (n={len(idx)})")
        ax.axvspan(*HOLD, color="grey", alpha=0.15, label="hold window")
        ax.set_title(f"{part}, {obj} object")
        ax.set_ylabel("a* relative to wall (redness)")
        ax.legend(fontsize=8)
for ax in axes[1]:
    ax.set_xlabel("normalised time in clip")
fig.suptitle("Mean redness over the clip, training people (Big_File_of_Big_Dreams.py)")
fig.tight_layout()
fig.savefig(FIG_DIR / "redness_curves.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for ax, part in zip(axes, ("hand", "nail")):
    groups, names = [], []
    for obj in ("hard", "soft"):
        for lab in ("light", "firm"):
            v = train.loc[(train["object"] == obj) & (train["label"] == lab), f"{part}_a_delta"].dropna()
            groups.append(v)
            names.append(f"{obj}\n{lab}")
    ax.boxplot(groups, tick_labels=names)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"{part}: change in a*, hold minus rest")
    ax.set_ylabel("delta a* (negative = paler)")
fig.suptitle("Per-grip change in redness, training people (Big_File_of_Big_Dreams.py)")
fig.tight_layout()
fig.savefig(FIG_DIR / "delta_a_boxplot.png", dpi=150)
print(f"\nfigures saved in {FIG_DIR}")
plt.show()
