# CODE MAP of extract_features.py
#   Settings       encoder name, frames per clip, input size, paths
#   Load labels    read data/labels.csv (one row per grip clip)
#   Sample frames  decode a clip, keep N_FRAMES evenly spaced frames (rest -> squeeze -> release)
#   Preprocess     resize whole frame to 224x392 (no crop), ImageNet normalisation
#   Encode         frozen DINOv2: CLS vector and mean of patch vectors for every sampled frame
#   Save           features/dinov2_small.npz, one row per clip, frames kept separate
#                  (pooling over time is done later in explore.py, so each pooling is a tree node)
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting
#
# HOW TO RUN (from the Part3 folder)
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#   pip install transformers opencv-python-headless numpy pandas
#   python extract_features.py
#
# OUTPUT  features/dinov2_small.npz
#   cls        (n_clips, N_FRAMES, 384)  CLS token of each sampled frame
#   patch      (n_clips, N_FRAMES, 384)  mean of the patch tokens of each sampled frame
#   frame_idx  (n_clips, N_FRAMES)       which frame of the clip was sampled
#   n_total    (n_clips,)                frames in the clip
#   clip, person, object, label  (n_clips,)  copied from labels.csv (clip = the 'file' column)

# %% Settings
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel

MODEL_ID = "facebook/dinov2-small"   # ViT-S/14, 384-dim features, ~22M parameters
N_FRAMES = 16                        # evenly spaced over the ~3 s clip (~90 frames at 30 fps)
IN_W, IN_H = 224, 392                # clips are portrait 480x854; both sides are multiples of the 14-px patch
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet statistics used to train DINOv2
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

HERE = Path(__file__).resolve().parent
VIDEO_DIR = HERE / "data" / "videos_480p"
LABELS = HERE / "data" / "labels.csv"
OUT = HERE / "features" / "dinov2_small.npz"

# %% Load labels
df = pd.read_csv(LABELS)
df["video"] = [VIDEO_DIR / f.replace(".MOV", ".mp4") for f in df["file"]]
missing = [str(v) for v in df["video"] if not v.exists()]
if missing:
    raise FileNotFoundError(f"{len(missing)} clips missing, e.g. {missing[0]}")


# %% Sample frames
def sample_frames(path, n=N_FRAMES):
    """Return n evenly spaced RGB frames of the clip and their frame numbers.

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
        raise ValueError(f"{path.name}: only {len(frames)} frames")
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx], idx, len(frames)


# %% Preprocess
def to_tensor(frames):
    """List of HxWx3 uint8 RGB frames -> normalised float tensor (n, 3, IN_H, IN_W)."""
    x = np.stack([cv2.resize(f, (IN_W, IN_H), interpolation=cv2.INTER_AREA) for f in frames])
    x = (x.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x).permute(0, 3, 1, 2)


# %% Encode
torch.manual_seed(0)
model = AutoModel.from_pretrained(MODEL_ID).eval()   # frozen: never trained, only run forward


@torch.inference_mode()
def encode(frames):
    """Return (cls, patch_mean), each (n, 384), for a batch of frames."""
    h = model(pixel_values=to_tensor(frames)).last_hidden_state   # (n, 1 + n_patches, 384), after final LayerNorm
    return h[:, 0].numpy(), h[:, 1:].mean(1).numpy()


n = len(df)
cls = np.zeros((n, N_FRAMES, model.config.hidden_size), dtype=np.float32)
patch = np.zeros_like(cls)
frame_idx = np.zeros((n, N_FRAMES), dtype=int)
n_total = np.zeros(n, dtype=int)

t0 = time.time()
for i, path in enumerate(df["video"]):
    frames, frame_idx[i], n_total[i] = sample_frames(path)
    cls[i], patch[i] = encode(frames)                 # one clip = one batch of N_FRAMES frames
    if (i + 1) % 10 == 0 or i + 1 == n:
        print(f"{i + 1}/{n} clips  ({time.time() - t0:.0f} s)")

# %% Save
OUT.parent.mkdir(exist_ok=True)
np.savez_compressed(
    OUT, cls=cls, patch=patch, frame_idx=frame_idx, n_total=n_total,
    clip=np.asarray(df["file"], dtype=str), person=np.asarray(df["person"], dtype=str),
    object=np.asarray(df["object"], dtype=str), label=np.asarray(df["label"], dtype=str),
)
print(f"saved {OUT}  cls {cls.shape}  patch {patch.shape}")
