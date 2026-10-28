# CODE MAP of colour_method.py   (shared code of the colour method; run it through test_A.py ... test_D.py)
#   Settings         paths, skin thresholds, time bins, MLP size
#   Load labels      read Part3/data/labels.csv
#   Finger patches   MediaPipe landmarks -> 10 discs per frame: nail and middle segment of each finger
#   Patch colour     per frame: CIELAB of the skin pixels in each disc, relative to the white wall
#   Clip curves      run on every clip once and cache colours (and landmarks) per frame
#   Network input    per grip: colour change from rest, averaged into time bins -> one vector per grip
#   Grip figures     one input -> output image per grip: frame with patches, colour curves, firm probability
#   Run one case     run(test_person): validation on the other three people (leave-one-person-out),
#                    choose input + model, train on all three, test once on test_person, save one image
#                    per test grip in figures/test_<person>/ and the validation bar chart
#   AI use           Claude Opus 5.5 (claude-opus-5-5), default setting
#
# IDEA  A firm grip pushes blood out of the fingertips (blanching), so nail and finger colour change over
#       the clip in a way that depends on grip force. MediaPipe only says WHERE the fingers are; the light/firm
#       decision is made by a small neural network (MLP) that only sees colour.
#
# The cache James F/cache/finger_patches.npz is shared with ../colour_mlp.py. If it is missing, the first
# test_X.py run rebuilds it (~10 min on CPU); every later run takes seconds.

# %% Settings
import time
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent                 # James F/colour_method
JAMES = HERE.parent                                     # James F
PART3 = JAMES.parent / "Part3" / "data"
VIDEO_DIR = PART3 / "videos_480p"
LABELS = PART3 / "labels.csv"
HAND_MODEL = JAMES / "models" / "hand_landmarker.task"  # MediaPipe hand landmarker (float16), pretrained
CACHE = JAMES / "cache" / "finger_patches.npz"
FIG_DIR = HERE / "figures"

SEED = 0

CR_MIN, CR_MAX = 135, 180    # skin box in YCrCb (same as Big_File_of_Big_Dreams.py)
CB_MIN, CB_MAX = 85, 135
Y_MIN = 60
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
MLP_HIDDEN = (16,)           # one hidden layer of 16 neurons
MLP_ALPHA = 1.0              # strong L2 penalty: ~80 training grips per fold, so overfitting is the risk

# nanmean over a patch/bin where MediaPipe never found the hand gives NaN (wanted; the imputer fills it)
warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)

# %% Load labels
df = pd.read_csv(LABELS)
df["video"] = [VIDEO_DIR / f.replace(".MOV", ".mp4") for f in df["file"]]
df["y"] = (df["label"] == "firm").astype(int)
PEOPLE = sorted(df["person"].unique())


# %% Finger patches
def patch_centres(p):
    """(21, 2) landmark pixels -> (10, 2) disc centres and (10,) radii: 5 nails, then 5 middle segments."""
    c, r = [], []
    for a, b in list(zip(TIPS, DIPS)) + list(zip(DIPS, PIPS)):
        c.append(0.5 * (p[a] + p[b]))
        r.append(max(PATCH_RADIUS * np.linalg.norm(p[a] - p[b]), 2.0))
    return np.array(c), np.array(r)


def skin_mask(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    return (cr >= CR_MIN) & (cr <= CR_MAX) & (cb >= CB_MIN) & (cb <= CB_MAX) & (y >= Y_MIN)


# %% Patch colour
def patch_colour(bgr, p):
    """One frame and its (21, 2) landmarks -> (10, 3) Lab colour of each patch minus the wall colour."""
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    wall = np.median(lab[: int(WALL_ROWS * lab.shape[0])].reshape(-1, 3), axis=0)
    skin = skin_mask(bgr)
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


# %% Clip curves
def clip_patches(path, detector_options):
    """One clip -> colours (n_frames, 10, 3) and landmarks (n_frames, 21, 2); NaN where no hand was found."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    detector = vision.HandLandmarker.create_from_options(detector_options)   # fresh tracker per clip
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cols, lms, i = [], [], 0
    ok, bgr = cap.read()
    while ok:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / fps))
        if res.hand_landmarks:
            p = np.array([[l.x * bgr.shape[1], l.y * bgr.shape[0]] for l in res.hand_landmarks[0]], np.float32)
            cols.append(patch_colour(bgr, p))
        else:
            p = np.full((21, 2), np.nan, np.float32)
            cols.append(np.full((len(REGIONS), 3), np.nan, np.float32))
        lms.append(p)
        i += 1
        ok, bgr = cap.read()
    cap.release()
    detector.close()
    return np.array(cols), np.array(lms)


def load_curves():
    """Per-frame patch colours and landmarks of every clip, from the cache (built on the first call)."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        assert list(z["clip_file"]) == list(df["file"]), "cache is from a different labels.csv; delete it"
        cuts = np.cumsum(z["n_frames"])[:-1]
        return np.split(z["colours"], cuts), np.split(z["landmarks"], cuts)

    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.3, min_tracking_confidence=0.3)
    colours, landmarks, t0 = [], [], time.time()
    for k, path in enumerate(df["video"]):
        c, l = clip_patches(path, opts)
        colours.append(c)
        landmarks.append(l)
        if (k + 1) % 20 == 0 or k + 1 == len(df):
            print(f"{k + 1}/{len(df)} clips  ({time.time() - t0:.0f} s)")
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, colours=np.concatenate(colours), landmarks=np.concatenate(landmarks),
                        n_frames=[len(c) for c in colours], clip_file=df["file"].values)
    return colours, landmarks


# %% Network input
def binned(c):
    """(n_frames, 10, 3) -> (N_BINS, 10, 3): nanmean of each equal time bin (NaN if the hand was never seen)."""
    return np.array([np.nanmean(seg, axis=0) for seg in np.array_split(c, N_BINS)], np.float32)


def change_curves(c):
    """Colour change from rest: binned colour minus the mean of the first REST_BINS bins that have data."""
    b = binned(c)
    rest = np.nanmean(b[:REST_BINS], axis=0)
    rest = np.where(np.isfinite(rest), rest, np.nanmean(b, axis=0))   # hand not seen early: use clip mean
    return b - rest                                    # (N_BINS, 10, 3), negative a* = paler than at rest


colours, landmarks = load_curves()
curves = np.array([change_curves(c) for c in colours])   # (n_clips, N_BINS, 10, 3)
found = np.array([np.isfinite(l[:, 0, 0]).mean() for l in landmarks])

nail = slice(0, 5)
mid = slice(5, 10)
a = CHANNELS.index("a")

# Each input is one node of the exploration tree; the network sees nothing but these colour numbers.
INPUTS = {
    # one number per finger region: a* change in the middle of the clip (bins 4-6)
    "per-finger a* change, hold":   curves[:, 4:7, :, a].mean(1),
    # the whole time course of a*, averaged over the 5 nails and over the 5 middle segments
    "nail + mid a* curve":          np.concatenate([np.nanmean(curves[:, :, nail, a], 2),
                                                    np.nanmean(curves[:, :, mid, a], 2)], 1),
    # time course of all three Lab channels for nails and middle segments
    "nail + mid Lab curves":        np.concatenate([np.nanmean(curves[:, :, nail, :], 2).reshape(len(df), -1),
                                                    np.nanmean(curves[:, :, mid, :], 2).reshape(len(df), -1)], 1),
    # everything: every region, every channel, every bin (300 numbers per grip)
    "all regions, all Lab, all bins": curves.reshape(len(df), -1),
}

MODELS = {
    "MLP": lambda: MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, alpha=MLP_ALPHA, max_iter=3000,
                                 random_state=SEED),
    "logistic regression": lambda: LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
}


def make_model(kind):
    # Imputer (for frames where MediaPipe lost the hand) and scaler are fitted on training clips only.
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         MODELS[kind]())


# %% Grip figures
def middle_frame(path, lm):
    """RGB frame nearest the middle of the clip where the hand was found (or the middle frame), and its index."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    ok, bgr = cap.read()
    while ok:
        frames.append(bgr)
        ok, bgr = cap.read()
    cap.release()
    seen = np.flatnonzero(np.isfinite(lm[:, 0, 0]))
    f = seen[np.argmin(np.abs(seen - len(frames) // 2))] if len(seen) else len(frames) // 2
    return cv2.cvtColor(frames[f], cv2.COLOR_BGR2RGB), int(f)


def plot_grip(k, p, is_train, model_name, test_person, out_dir):
    """Save one input -> output image for grip k: frame with patches, colour curves, firm probability."""
    row = df.loc[k]
    img, f_show = middle_frame(row["video"], landmarks[k])
    hand_seen = np.isfinite(landmarks[k][f_show, 0, 0])
    if hand_seen:
        for j, (c, r) in enumerate(zip(*patch_centres(landmarks[k][f_show]))):
            cv2.circle(img, (int(c[0]), int(c[1])), int(r), (0, 200, 0) if j < 5 else (255, 140, 0), 2)  # RGB
    pred = "firm" if p > 0.5 else "light"
    right = pred == row["label"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), gridspec_kw={"width_ratios": [0.8, 1.4, 0.6]})
    axes[0].imshow(img)
    axes[0].set_title(f"input: {Path(row['file']).stem}\nframe {f_show}; "
                      + ("green = nail,\norange = middle segment" if hand_seen else "\nhand not found in any frame"),
                      fontsize=9)
    axes[0].axis("off")
    t = (np.arange(N_BINS) + 0.5) / N_BINS
    axes[1].plot(t, np.nanmean(curves[k, :, nail, a], 1), "o-", color="tab:green", label="nails, this grip")
    axes[1].plot(t, np.nanmean(curves[k, :, mid, a], 1), "o-", color="tab:orange", label="middle segments, this grip")
    for lab, ls in (("light", ":"), ("firm", "--")):
        m = is_train & (df["label"] == lab).values & (df["object"] == row["object"]).values
        axes[1].plot(t, np.nanmean(curves[m][:, :, nail, a], axis=(0, 2)), ls, color="grey",
                     label=f"nails, mean of training {lab} ({row['object']})")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("normalised time in clip")
    axes[1].set_ylabel("change in a* from rest (negative = paler)")
    axes[1].set_title(f"colour signal the network sees (hand found in {100 * found[k]:.0f} % of frames)")
    axes[1].legend(fontsize=7)
    axes[2].bar(["light", "firm"], [1 - p, p], color=["tab:blue", "tab:red"])
    axes[2].axhline(0.5, color="k", ls="--", lw=0.8)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("probability")
    axes[2].set_title(f"output ({model_name})\ntrue: {row['label']}, predicted: {pred}",
                      color="tab:green" if right else "tab:red")
    fig.suptitle(f"Colour method, test person {test_person}: {Path(row['file']).stem}, "
                 f"{'correct' if right else 'WRONG'} (test_{test_person}.py)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{Path(row['file']).stem}_{'correct' if right else 'wrong'}.png", dpi=110)
    plt.close(fig)                                     # 40 figures per case: do not keep them open


# %% Run one case
def run(test_person, show=True):
    """Hold out test_person, choose input + model on the other three, test once. Returns a result dict."""
    y = df["y"].values
    is_train = (df["person"] != test_person).values
    people = [p for p in PEOPLE if p != test_person]
    print(f"=== test person {test_person} ===")
    print(f"train people {people}; {is_train.sum()} train clips, {(~is_train).sum()} test clips")
    print(f"MediaPipe found the hand in {100 * found[is_train].mean():.0f} % of frames (train clips), "
          f"{100 * found[~is_train].mean():.0f} % (test clips)\n")

    # Validation: leave-one-person-out inside the three training people; the test person is never touched.
    print("Leave-one-person-out validation accuracy (guessing = 50 %)")
    print(f"{'input':<32}{'model':<21}" + "".join(f"{'val ' + p:>8}" for p in people) + f"{'mean':>8}")
    val_results = {}
    for name, X in INPUTS.items():
        for kind in MODELS:
            acc = []
            for p in people:
                tr = is_train & (df["person"] != p).values
                va = is_train & (df["person"] == p).values
                acc.append((make_model(kind).fit(X[tr], y[tr]).predict(X[va]) == y[va]).mean())
            val_results[(name, kind)] = np.mean(acc)
            print(f"{name:<32}{kind:<21}" + "".join(f"{100 * v:>7.0f}%" for v in acc)
                  + f"{100 * np.mean(acc):>7.0f}%")

    # Test: train the chosen version on all three training people, evaluate once on test_person.
    best_input, best_model = max(val_results, key=val_results.get)   # chosen on validation only
    X = INPUTS[best_input]
    model = make_model(best_model).fit(X[is_train], y[is_train])
    p_firm = model.predict_proba(X)[:, 1]
    correct = (p_firm > 0.5) == y
    test_acc = correct[~is_train].mean()
    n_test = (~is_train).sum()
    print(f"\nchosen on validation: '{best_input}' + {best_model} "
          f"(val {100 * val_results[(best_input, best_model)]:.0f} %)")
    print(f"test person {test_person}: {100 * test_acc:.1f} % ({correct[~is_train].sum()}/{n_test}), "
          f"guessing = 50 %")
    per_object = {}
    for obj in ("hard", "soft"):
        m = ~is_train & (df["object"] == obj).values
        per_object[obj] = correct[m].mean()
        print(f"  {obj} object: {100 * per_object[obj]:.0f} %")

    # Figures: one input -> output image per test grip, then the validation bars.
    grip_dir = FIG_DIR / f"test_{test_person}"
    grip_dir.mkdir(parents=True, exist_ok=True)
    for k in np.flatnonzero(~is_train):
        plot_grip(k, p_firm[k], is_train, best_model, test_person, grip_dir)
    print(f"{n_test} grip images saved in {grip_dir}")

    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(INPUTS)
    for j, kind in enumerate(MODELS):
        ax.bar(np.arange(len(names)) + (j - 0.5) * 0.38, [100 * val_results[(n, kind)] for n in names],
               width=0.38, label=kind)
    ax.axhline(50, color="k", ls="--", lw=1, label="guessing")
    ax.set_xticks(np.arange(len(names)), names, fontsize=8)
    ax.set_ylabel("validation accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Leave-one-person-out validation on {', '.join(people)}; test person {test_person} "
                 f"(test_{test_person}.py)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"test_{test_person}_validation.png", dpi=150)
    print(f"\nfigures saved in {FIG_DIR}")
    if show:
        plt.show()

    return {"test_person": test_person, "input": best_input, "model": best_model,
            "val_acc": val_results[(best_input, best_model)], "test_acc": test_acc,
            "test_correct": int(correct[~is_train].sum()), "n_test": int(n_test), **per_object}
