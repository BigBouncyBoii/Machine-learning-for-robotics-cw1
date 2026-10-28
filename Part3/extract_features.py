# CODE MAP of extract_features.py
#   Settings       which encoder (command-line argument)
#   Load labels    read data/labels.csv (one row per grip clip, train AND test: the encoder is frozen and
#                  sees each clip on its own, so nothing is learned from the test clips)
#   Encode         for every clip: 16 evenly spaced frames (common.sample_frames) -> frozen encoder (encoders/)
#   Save           features/<encoder file>.npz, one row per clip; pooling is chosen later (common.FEATURES)
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting
#
# HOW TO RUN (from the Part3 folder, venv activated)
#   python extract_features.py dinov2      # -> features/dinov2_small.npz   (~2 min on CPU)
#   python extract_features.py videomae    # -> features/videomae_base.npz  (~10 min on CPU, downloads ~350 MB once)
#
# OUTPUT  features/<file>.npz
#   the encoder's arrays, e.g. cls/patch (n_clips, 16, 384) for DINOv2, vec (n_clips, 768) for VideoMAE
#   frame_idx  (n_clips, 16)   which frames of the clip were sampled
#   n_total    (n_clips,)      frames in the clip
#   clip, person, object, label  (n_clips,)  copied from labels.csv (clip = the 'file' column)

# %% Settings
import argparse
import time

import numpy as np
import pandas as pd
import torch

import encoders
from common import FEATURE_DIR, LABELS, sample_frames, video_path

parser = argparse.ArgumentParser()
parser.add_argument("encoder", choices=encoders.ENCODERS)
enc = encoders.get(parser.parse_args().encoder)
out = FEATURE_DIR / f"{enc.FILE}.npz"
torch.manual_seed(0)

# %% Load labels
df = pd.read_csv(LABELS)
paths = [video_path(f) for f in df["file"]]
missing = [str(p) for p in paths if not p.exists()]
if missing:
    raise FileNotFoundError(f"{len(missing)} clips missing, e.g. {missing[0]}")

# %% Encode
encode = enc.load()
arrays, frame_idx, n_total = {}, [], []
t0 = time.time()
for i, path in enumerate(paths):
    frames, idx, n = sample_frames(path)
    for k, v in encode(frames).items():
        arrays.setdefault(k, []).append(v)
    frame_idx.append(idx)
    n_total.append(n)
    if (i + 1) % 10 == 0 or i + 1 == len(paths):
        print(f"{i + 1}/{len(paths)} clips  ({time.time() - t0:.0f} s)")

# %% Save
FEATURE_DIR.mkdir(exist_ok=True)
arrays = {k: np.stack(v) for k, v in arrays.items()}
np.savez_compressed(
    out, **arrays, frame_idx=np.stack(frame_idx), n_total=np.array(n_total),
    clip=np.asarray(df["file"], dtype=str), person=np.asarray(df["person"], dtype=str),
    object=np.asarray(df["object"], dtype=str), label=np.asarray(df["label"], dtype=str),
)
print(f"saved {out}  " + "  ".join(f"{k} {v.shape}" for k, v in arrays.items()))
