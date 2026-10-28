# CODE MAP of frames_node.py
#   Settings       frame counts to compare (command line), DINOv2 + MLP from common.CLASSIFIERS
#   Extract        for each frame count N: N evenly spaced frames per clip -> frozen DINOv2 -> CLS per frame,
#                  cached as features/dinov2_small_<N>f.npz (N = 16 reuses features/dinov2_small.npz)
#   Pool           mean of the N frames' CLS vectors -> one 384-number vector per clip
#   Validate       leave-one-person-out on the 120 train clips (the number for the tree)
#   Test           optional (--test): train on all 120 train clips, accuracy on the 40 test clips (same split.csv)
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting
#
# Tree node: "DINOv2 + MLP with 8 vs 16 frames: does seeing more of the grip help?"
# The frames are re-sampled from the video for each N (not a subset of the 16), so 8 frames still span the whole clip.
#
# HOW TO RUN (from the Part3 folder, venv activated)
#   python frames_node.py                  # 8 and 16 frames, validation only
#   python frames_node.py 4 8 16 --test    # any frame counts, plus test accuracy

# %% Settings
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

import encoders
from common import CLASSIFIERS, FEATURE_DIR, SEEDS, SPLIT, sample_frames, video_path

parser = argparse.ArgumentParser()
parser.add_argument("frames", nargs="*", type=int, default=[8, 16])
parser.add_argument("--test", action="store_true", help="also report accuracy on the 40 test clips")
args = parser.parse_args()

warnings.filterwarnings("ignore", category=ConvergenceWarning)
split = pd.read_csv(SPLIT)                       # same fixed split as every other script
enc = encoders.get("dinov2")
encode = None                                    # DINOv2 is loaded only if something needs extracting


# %% Extract
def features(n):
    """Return (cls (n_clips, n, 384), split rows) for n frames per clip, extracting once and caching."""
    global encode
    out = FEATURE_DIR / ("dinov2_small.npz" if n == 16 else f"dinov2_small_{n}f.npz")
    if not out.exists():
        encode = encode or enc.load()
        t0 = time.time()
        cls = np.stack([encode(sample_frames(video_path(f), n)[0])["cls"] for f in split["file"]])
        np.savez_compressed(out, cls=cls, clip=np.asarray(split["file"], dtype=str))
        print(f"  extracted {n} frames/clip for {len(split)} clips ({time.time() - t0:.0f} s) -> {out.name}")
    feat = np.load(out)
    rows = split.set_index("file").loc[feat["clip"]].reset_index()   # match by file name, not by row order
    return feat["cls"], rows


# %% Validate / Test
def mlp_probs(X_tr, y_tr, X_ev):
    """Firm-probability of X_ev, averaged over the MLP seeds."""
    return np.mean([CLASSIFIERS["mlp"](s).fit(X_tr, y_tr).predict_proba(X_ev)[:, 1] for s in SEEDS["mlp"]], axis=0)


results = []
for n in args.frames:
    print(f"\n{n} frames per clip")
    cls, rows = features(n)
    X = cls.mean(axis=1)                                              # Pool: mean over the n frames
    y = (rows["label"] == "firm").values.astype(int)
    train = (rows["set"] == "train").values
    person, obj = rows["person"].values, rows["object"].values

    per_person = {}
    for p in sorted(set(person[train])):
        tr, va = train & (person != p), train & (person == p)
        per_person[p] = ((mlp_probs(X[tr], y[tr], X[va]) >= 0.5) == y[va]).mean()
    row = {"frames": n, "val": np.mean(list(per_person.values())), **{f"val {p}": a for p, a in per_person.items()}}

    if args.test:
        correct = (mlp_probs(X[train], y[train], X[~train]) >= 0.5) == y[~train]
        row["test"] = correct.mean()
        row.update({f"test {o}": correct[obj[~train] == o].mean() for o in sorted(set(obj))})
    results.append(row)

# %% Report
pd.set_option("display.float_format", "{:.3f}".format)
print("\nDINOv2 CLS, mean over frames -> MLP (64), 5 seeds")
print("val = mean leave-one-person-out accuracy on the 120 train clips (the tree number); guessing = 0.50")
print(pd.DataFrame(results).set_index("frames"))
