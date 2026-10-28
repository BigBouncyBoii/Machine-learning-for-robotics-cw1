# CODE MAP of predict_clip.py
#   Settings       which feature version (default dinov2_cls_mean), MLP settings (same as FinalMethod.py)
#   Train          scaler + MLPs fitted on the TRAIN rows of the cached features
#   Test list      print the clips in data/split.csv marked "test" (never used for training)
#   Encode a clip  video -> 16 evenly spaced frames -> frozen encoder -> pooled vector
#                  (common.sample_frames + encoders/ + common.FEATURES: the same code that made the cached features)
#   Predict        vector -> MLPs -> light / firm, with the firm-probability; shows the true label if the clip is known
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting
#
# Demo tool: classify one video at a time, starting from the raw video (not the cached features).
#
# HOW TO RUN (from the Part3 folder, venv activated)
#   python predict_clip.py                                               # then paste video paths when asked
#   python predict_clip.py data/videos_480p/A/A_hard_firm_02.mp4 ...     # or give paths directly
#   python predict_clip.py --features videomae                           # use another feature version

# %% Settings
import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import encoders
from common import FEATURES, HERE, load_features, sample_frames, video_path

parser = argparse.ArgumentParser()
parser.add_argument("--features", default="dinov2_cls_mean", choices=FEATURES)
parser.add_argument("videos", nargs="*")
args = parser.parse_args()

HIDDEN = (64,)
ALPHA = 1e-2
MAX_ITER = 500
SEEDS = range(5)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# %% Train
X_all, meta = load_features(args.features)
train = (meta["set"] == "train").values
y = (meta["label"].values == "firm").astype(int)   # 1 = firm, 0 = light
models = [make_pipeline(StandardScaler(), MLPClassifier(
    hidden_layer_sizes=HIDDEN, alpha=ALPHA, max_iter=MAX_ITER, random_state=s)).fit(X_all[train], y[train])
    for s in SEEDS]
print(f"MLP trained on {train.sum()} train clips ({args.features}, cached features)")

# %% Test list
test_files = meta.loc[meta["set"] == "test", "file"]
print(f"\n{len(test_files)} test clips (not used for training):")
for f in test_files:
    print("  " + str(video_path(f).relative_to(HERE)))

# known clips by video name (e.g. "A_hard_firm_03"), to show set and true label
known = {Path(f).stem: (s, l) for f, s, l in zip(meta["file"], meta["set"], meta["label"])}

# %% Encode a clip
encoder_name, _, pool = FEATURES[args.features]
print(f"\nloading {encoders.get(encoder_name).MODEL_ID} ...")
encode = encoders.get(encoder_name).load()


def clip_vector(path):
    """Video file -> one pooled vector, exactly as stored for the cached clips."""
    out = encode(sample_frames(path)[0])
    return pool({k: v[None] for k, v in out.items()})   # add the clip axis that pool expects


# %% Predict
def predict(path):
    path = Path(path.strip().strip("'\"")).expanduser()
    if not path.exists():
        print(f"  not found: {path}")
        return
    x = clip_vector(path)
    p_firm = np.mean([m.predict_proba(x)[0, 1] for m in models])
    guess = "firm" if p_firm >= 0.5 else "light"
    line = f"  {path.name}: {guess.upper()}  (firm-probability {p_firm:.2f})"
    if path.stem in known:
        s, true = known[path.stem]
        line += f"  | true: {true} ({'correct' if true == guess else 'WRONG'})  | in {s} set"
        if s == "train":
            line += "  <- the MLP was trained on this clip"
    print(line)


if args.videos:
    for p in args.videos:
        predict(p)
else:
    while (p := input("\nvideo path (Enter to quit): ")).strip():
        predict(p)
