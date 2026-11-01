# CODE MAP of predict_clip.py   (demo for the viva; not part of the tree, not needed for any reported number)
#   Settings       which feature set (default dinov2_middle) and classifier (default mlp_ensemble)
#   Train          scaler + classifier fitted on the 120 TRAIN clips of the cached features
#   Test list      print the clips marked "test" in data/split.csv (never used for training)
#   Encode a clip  video -> evenly spaced frames -> frozen encoder -> pooled vector (same code as the cache)
#   Predict        vector -> classifier -> light / firm, with the firm-probability; true label if the clip is known
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting
#
# HOW TO RUN (from anywhere, venv activated)
#   python "Part3/James L/predict_clip.py"                                         # then paste video paths
#   python "Part3/James L/predict_clip.py" Part3/data/videos_480p/A/A_hard_firm_02.mp4 ...
#   python "Part3/James L/predict_clip.py" --features videomae --classifier lr

# %% Settings
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.exceptions import ConvergenceWarning

import common
from classifiers import CLASSIFIERS
from extractors import EXTRACTORS

SET_TO_EXTRACTOR = {s: e for e, m in EXTRACTORS.items() for s in m.FEATURE_SETS}

parser = argparse.ArgumentParser()
parser.add_argument("--features", default="dinov2_middle", choices=list(SET_TO_EXTRACTOR))
parser.add_argument("--classifier", default="mlp_ensemble", choices=list(CLASSIFIERS))
parser.add_argument("videos", nargs="*")
args = parser.parse_args()

warnings.filterwarnings("ignore", category=ConvergenceWarning)
extractor = EXTRACTORS[SET_TO_EXTRACTOR[args.features]]

# %% Train
clips = common.load_clips()
X = extractor.load(clips, [args.features])[args.features]
train = (clips["set"] == "train").values
model = common.make_pipeline(CLASSIFIERS[args.classifier].make()).fit(X[train], clips["y"].values[train])
print(f"{args.classifier} trained on {train.sum()} train clips ({args.features}, cached features)")

# %% Test list
print(f"\n{(~train).sum()} test clips (not used for training):")
for v in clips.loc[~train, "video"]:
    print("  " + str(v.relative_to(common.HERE.parent.parent)))

known = {Path(f).stem: (s, l) for f, s, l in zip(clips["file"], clips["set"], clips["label"])}

# %% Encode a clip
print(f"\nloading {extractor.MODEL_ID} ...")
encode = extractor.load_encoder()
n_frames, pool = extractor.POOL[args.features]


def clip_vector(path):
    """Video file -> one pooled vector, exactly as stored for the cached clips."""
    out = encode(common.sample_frames(path, n_frames)[0])
    return pool({k: v[None] for k, v in out.items()})   # add the clip axis that the pooling expects


# %% Predict
def predict(path):
    path = Path(path.strip().strip("'\"")).expanduser()
    if not path.exists():
        print(f"  not found: {path}")
        return
    p_firm = model.predict_proba(clip_vector(path))[0, 1]
    guess = "firm" if p_firm >= 0.5 else "light"
    line = f"  {path.name}: {guess.upper()}  (firm-probability {p_firm:.2f})"
    if path.stem in known:
        s, true = known[path.stem]
        line += f"  | true: {true} ({'correct' if true == guess else 'WRONG'})  | in {s} set"
        if s == "train":
            line += "  <- the classifier was trained on this clip"
    print(line)


if args.videos:
    for p in args.videos:
        predict(p)
else:
    while (p := input("\nvideo path (Enter to quit): ")).strip():
        predict(p)
