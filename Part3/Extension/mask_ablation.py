# CODE MAP of Extension/mask_ablation.py   (Going further, extension B: what does the final method rely on?)
#   Settings      the final method's frames (middle 6 of 16) and classifier; the two masks; output folders
#   Masks         skin_mask(): hand = skin-coloured pixels (YCrCb threshold, cleaned); apply_mask():
#                 "hand_removed" paints the hand the background colour; "hand_only" paints everything else
#   Features      the 16 frames of every clip are masked and encoded by the same frozen DINOv2 as the final
#                 method -> cache/dinov2_<mask>.npz (no labels used; ~2 min CPU per mask, built once)
#   Pictures      ONLY for the report: 2 example clips (EXAMPLES) -> figures/mask_examples.png and
#                 videos/*.mp4 side by side. The ablation itself masks every sampled frame of all 160 clips.
#   Evaluate      for each mask: (1) RETRAIN on masked train clips, validate leave-one-person-out, test on
#                 masked test clips; (2) TEST ONLY: the unchanged final model on masked test clips.
#                 The 40 test clips are never used for training. Table -> results/mask_ablation.csv
#   AI use        Claude Opus 5.5 (claude-opus-5-5), default setting
#
# Why colour masks: the background is a plain white wall and table, so skin is easy to separate by colour
# (MediaPipe found the hand in only ~42% of frames, see Jeevan's node). The dark sleeves look like the
# objects (black phone, grey toy) and touch them, so the object cannot be cut out alone; "hand_only"
# therefore removes the object AND the sleeve / arm, and "hand_removed" keeps both.
#
# HOW TO RUN (venv activated, from any folder)
#   python Part3/Extension/mask_ablation.py                 # features (first run), figure, 2 example videos, table
#   python Part3/Extension/mask_ablation.py --videos none   # table only

# %% Settings
import argparse
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent                     # Part3/Extension
sys.path.insert(0, str(HERE.parent / "James L"))           # James L's modules (same code as FinalMethod.py)

import cv2
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.exceptions import ConvergenceWarning

import common
from classifiers import mlp_ensemble
from extractors import dinov2, dinov2_phase

MASKS = ["hand_removed", "hand_only"]
MIDDLE = dinov2_phase.FRAMES["dinov2_middle"]               # [5..10]: the frames the final method averages
CACHE_DIR = HERE / "cache"
FIG_DIR = HERE / "figures"
VIDEO_DIR = HERE / "videos"
RESULTS = HERE / "results"

# skin in YCrCb (chosen by looking at frames of all 4 people; the wall has Cr ~ 134, the grey toy and the
# sleeves ~ 128, skin 140-175)
CR_RANGE, CB_RANGE, MIN_Y = (140, 175), (80, 125), 60
EXAMPLES = ["A/A_hard_firm_01.MOV", "A/A_soft_firm_01.MOV"]   # clips shown in the figure / videos (illustration only)
GROW = 7                    # px (at 480x854) the hand mask is grown by, so no skin edge is left after masking

parser = argparse.ArgumentParser()
parser.add_argument("--videos", default="examples", choices=["examples", "none"])
args = parser.parse_args()
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# %% Masks
def skin_mask(frame):
    """RGB frame -> bool mask of the hand (skin colour), small holes closed, grown by GROW px."""
    ycc = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    m = ((cr >= CR_RANGE[0]) & (cr <= CR_RANGE[1]) & (cb >= CB_RANGE[0]) & (cb <= CB_RANGE[1])
         & (y > MIN_Y)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))       # drop single-pixel specks
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))      # fill gaps between fingers' edges
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GROW + 1, 2 * GROW + 1)))
    return m.astype(bool)


def background_colour(frame, hand):
    """Median colour of the bright, grey pixels (wall and table) of this frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    bg = (hsv[..., 2] > 170) & (hsv[..., 1] < 60) & ~hand
    return np.median(frame[bg], axis=0).astype(np.uint8) if bg.any() else np.array([235, 235, 235], np.uint8)


def apply_mask(frame, mask):
    """'hand_removed': hand painted background colour. 'hand_only': everything except the hand painted."""
    hand = skin_mask(frame)
    out = frame.copy()
    out[hand if mask == "hand_removed" else ~hand] = background_colour(frame, hand)
    return out


# %% Features
def masked_features(clips, mask):
    """{'cls': (160, 16, 384)} of the masked frames, in clips order; encoded once and cached."""
    path = CACHE_DIR / f"dinov2_{mask}.npz"
    if not path.exists():
        encode, cls, t0 = dinov2.load_encoder(), [], time.time()
        for i, video in enumerate(clips["video"]):
            frames = common.sample_frames(video, common.N_FRAMES)[0]          # the same 16 frames as the cache
            cls.append(encode([apply_mask(f, mask) for f in frames])["cls"])
            if (i + 1) % 20 == 0 or i + 1 == len(clips):
                print(f"  DINOv2, {mask}: {i + 1}/{len(clips)} clips ({time.time() - t0:.0f} s)")
        CACHE_DIR.mkdir(exist_ok=True)
        np.savez_compressed(path, cls=np.stack(cls), clip=np.asarray(clips["file"], dtype=str))
    z = np.load(path)
    row = {c: i for i, c in enumerate(z["clip"])}
    return z["cls"][[row[f] for f in clips["file"]]]


def middle(cls):
    """The final method's pooling: mean CLS of the middle 6 of the 16 frames -> (n_clips, 384)."""
    return cls[:, MIDDLE].mean(axis=1)


# %% Pictures
def label(img, text):
    out = img.copy()
    cv2.putText(out, text, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def example_figure(clips):
    """Middle frame of each EXAMPLES clip: original / hand removed / hand only (no labels in the frames)."""
    import matplotlib.pyplot as plt
    picks = [r for _, r in clips[clips["file"].isin(EXAMPLES)].iterrows()]
    fig, ax = plt.subplots(len(picks), 3, figsize=(3 * 1.9, len(picks) * 3.2), squeeze=False)
    for j, r in enumerate(picks):
        f = common.sample_frames(r["video"], common.N_FRAMES)[0][MIDDLE[len(MIDDLE) // 2]]
        for i, (name, img) in enumerate([("original", f)] + [(m, apply_mask(f, m)) for m in MASKS]):
            ax[j, i].imshow(img)
            ax[j, i].set_xticks([]), ax[j, i].set_yticks([])
            if j == 0:
                ax[j, i].set_title(name.replace("_", " "), fontsize=9)
        ax[j, 0].set_ylabel(f"{r['object']} object", fontsize=9)
    fig.suptitle("Masks used in the ablation (Extension/mask_ablation.py)", fontsize=10)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "mask_examples.png", dpi=200)
    plt.close(fig)
    print(f"figure -> {FIG_DIR / 'mask_examples.png'}")


def write_videos(rows):
    """Full-length side-by-side videos: original | hand removed | hand only."""
    VIDEO_DIR.mkdir(exist_ok=True)
    for _, r in rows.iterrows():
        cap = cv2.VideoCapture(str(r["video"]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        out_path = VIDEO_DIR / Path(r["file"]).with_suffix(".mp4").name
        writer = None
        ok, bgr = cap.read()
        while ok:
            f = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            panels = [label(f, "original")] + [label(apply_mask(f, m), m.replace("_", " ")) for m in MASKS]
            tile = cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR)
            if writer is None:
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (tile.shape[1], tile.shape[0]))
            writer.write(tile)
            ok, bgr = cap.read()
        cap.release()
        if writer is not None:
            writer.release()
    print(f"{len(rows)} videos -> {VIDEO_DIR}")


# %% Evaluate
clips = common.load_clips()                                   # stops if the test-set fingerprint is wrong
train = (clips["set"] == "train").values
test = (clips["set"] == "test").values
y = clips["y"].values
assert not set(clips["file"][train]) & set(clips["file"][test])

example_figure(clips)
if args.videos != "none":
    write_videos(clips[clips["file"].isin(EXAMPLES)])


def test_acc(model, X):
    correct = model.predict(X[test]) == y[test]
    return correct.mean(), {o: correct[clips["object"].values[test] == o].mean() for o in ("hard", "soft")}


X_orig = dinov2_phase.load(clips, ["dinov2_middle"])["dinov2_middle"]          # the final method's features
final_model = common.make_pipeline(mlp_ensemble.make()).fit(X_orig[train], y[train])   # = FinalMethod.py

rows = []
acc, by_obj = test_acc(final_model, X_orig)
val = common.validate(X_orig, clips, mlp_ensemble.make)["mean_acc"]
rows.append(dict(input="original (final method)", mode="-", val=val, test=acc, **by_obj))

for mask in MASKS:
    X = middle(masked_features(clips, mask))
    # (1) retrain: scaler + 5 MLPs fitted on the 120 masked train clips only
    val = common.validate(X, clips, mlp_ensemble.make)["mean_acc"]
    model = common.make_pipeline(mlp_ensemble.make()).fit(X[train], y[train])
    acc, by_obj = test_acc(model, X)
    rows.append(dict(input=mask, mode="retrained", val=val, test=acc, **by_obj))
    # (2) test only: the unchanged final model (trained on original frames) sees masked test clips
    acc, by_obj = test_acc(final_model, X)
    rows.append(dict(input=mask, mode="test only", val=np.nan, test=acc, **by_obj))

table = pd.DataFrame(rows)
n = int(test.sum())
table["p_vs_guess"] = [binomtest(round(a * n), n, 0.5, alternative="greater").pvalue for a in table["test"]]
RESULTS.mkdir(exist_ok=True)
table.to_csv(RESULTS / "mask_ablation.csv", index=False)

print(f"\nMask ablation, DINOv2 middle-6 + 5-MLP ensemble; {train.sum()} train / {n} test clips; guessing = 0.50")
print("val = leave-one-person-out on the train clips; test = the 40 held-out clips (never trained on)\n")
print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="-"))
