# CODE MAP of predict_clip.py
#   Settings       paths, frame sampling and MLP settings (must match extract_features.py and FinalMethod.py)
#   Train          scaler + MLPs fitted on the TRAIN rows of the cached features (features/dinov2_small.npz)
#   Test list      print the clips in data/split.csv marked "test" (never used for training)
#   Encode a clip  video -> 16 evenly spaced frames -> frozen DINOv2 -> mean CLS vector (same steps as extract_features.py)
#   Predict        vector -> MLPs -> light / firm, with the firm-probability; shows the true label if the clip is known
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting
#
# Demo tool: classify one video at a time, starting from the raw video (not the cached features).
# Using it on test clips is fine for a demo, but do not change the method based on what it shows.
#
# HOW TO RUN (from the Part3 folder)
#   ../.venv/bin/python predict_clip.py                                   # then type/paste video paths when asked
#   ../.venv/bin/python predict_clip.py data/videos_480p/A/A_hard_firm_03.mp4 ...   # or give paths directly

# %% Settings
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel

HERE = Path(__file__).resolve().parent
FEATURES = HERE / "features" / "dinov2_small.npz"
SPLIT = HERE / "data" / "split.csv"
VIDEO_DIR = HERE / "data" / "videos_480p"

MODEL_ID = "facebook/dinov2-small"     # same as extract_features.py
N_FRAMES = 16
IN_W, IN_H = 224, 392
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

HIDDEN = (64,)                         # same as FinalMethod.py
ALPHA = 1e-2
MAX_ITER = 500
SEEDS = range(5)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# %% Train
feat = np.load(FEATURES)
split = pd.read_csv(SPLIT)
clip_set = split.set_index("file").loc[feat["clip"], "set"].values
train = clip_set == "train"
X_tr = feat["cls"][train].mean(axis=1)                 # mean over the 16 frames, as in explore.py
y_tr = (feat["label"][train] == "firm").astype(int)    # 1 = firm, 0 = light
models = [make_pipeline(StandardScaler(), MLPClassifier(
    hidden_layer_sizes=HIDDEN, alpha=ALPHA, max_iter=MAX_ITER, random_state=s)).fit(X_tr, y_tr)
    for s in SEEDS]
print(f"MLP trained on {train.sum()} train clips (cached features)")

# %% Test list
test = split[split["set"] == "test"]
print(f"\n{len(test)} test clips (not used for training):")
for f in test["file"]:
    print("  " + str((VIDEO_DIR / f.replace(".MOV", ".mp4")).relative_to(HERE)))

# known clips by video name (e.g. "A_hard_firm_03"), to show set and true label
known = {Path(f).stem: (s, l) for f, s, l in zip(split["file"], split["set"], split["label"])}


# %% Encode a clip
def sample_frames(path, n=N_FRAMES):
    """Return n evenly spaced RGB frames of the clip (same as extract_features.py)."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ok, frame = cap.read()
    cap.release()
    if len(frames) < n:
        raise ValueError(f"only {len(frames)} frames could be read")
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


def to_tensor(frames):
    """List of HxWx3 uint8 RGB frames -> normalised float tensor (n, 3, IN_H, IN_W)."""
    x = np.stack([cv2.resize(f, (IN_W, IN_H), interpolation=cv2.INTER_AREA) for f in frames])
    x = (x.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x).permute(0, 3, 1, 2)


print(f"\nloading {MODEL_ID} ...")
encoder = AutoModel.from_pretrained(MODEL_ID).eval()   # frozen, forward pass only


@torch.inference_mode()
def clip_vector(path):
    """Video file -> one 384-number vector (mean of the 16 frames' CLS tokens)."""
    h = encoder(pixel_values=to_tensor(sample_frames(path))).last_hidden_state
    return h[:, 0].mean(0).numpy()


# %% Predict
def predict(path):
    path = Path(path.strip().strip("'\"")).expanduser()
    if not path.exists():
        print(f"  not found: {path}")
        return
    x = clip_vector(path)[None]
    p_firm = np.mean([m.predict_proba(x)[0, 1] for m in models])
    guess = "firm" if p_firm >= 0.5 else "light"
    line = f"  {path.name}: {guess.upper()}  (firm-probability {p_firm:.2f})"
    if path.stem in known:
        s, true = known[path.stem]
        line += f"  | true: {true} ({'correct' if true == guess else 'WRONG'})  | in {s} set"
        if s == "train":
            line += "  <- the MLP was trained on this clip"
    print(line)


if len(sys.argv) > 1:
    for p in sys.argv[1:]:
        predict(p)
else:
    while (p := input("\nvideo path (Enter to quit): ")).strip():
        predict(p)
