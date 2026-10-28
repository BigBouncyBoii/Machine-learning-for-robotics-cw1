# CODE MAP of FinalMethod.py
#   Settings       which features and classifier (defaults = the final method), from the command line
#   Load           cached features + data/split.csv, split into TRAIN (120 clips) and TEST (40 clips)
#                  the split is read from the file, never redrawn, so every run uses the same test clips
#   Train          scaler + classifier fitted on ALL train clips (nothing is fitted on test clips);
#                  MLP: 5 seeds, their firm-probabilities averaged
#   Test           predict the test clips; accuracy overall, per object and per person, vs 50% guessing
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting
#
# Final method = tree node "VideoMAE clip vector -> LR" (best validation accuracy, 65.8%).
# The options let us run the other combinations on the same test clips for comparison.
#
# HOW TO RUN (from the Part3 folder, venv activated, after extract_features.py and make_split.py)
#   python FinalMethod.py                                         # final method: videomae + lr
#   python FinalMethod.py --features dinov2_cls_mean --classifier mlp
#   options: --features dinov2_cls_mean | dinov2_patch_mean | videomae    --classifier lr | mlp

# %% Settings
import argparse
import hashlib
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix

from common import CLASSIFIERS, FEATURES, SEEDS, load_features

parser = argparse.ArgumentParser()
parser.add_argument("--features", default="videomae", choices=FEATURES)
parser.add_argument("--classifier", default="lr", choices=CLASSIFIERS)
args = parser.parse_args()

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# %% Load
X, meta = load_features(args.features)
train = (meta["set"] == "train").values
test = (meta["set"] == "test").values
y = (meta["label"].values == "firm").astype(int)       # 1 = firm, 0 = light
assert not set(meta["file"][train]) & set(meta["file"][test])

# fingerprint of the test-clip list: identical in every run means the same test set
fingerprint = hashlib.md5("\n".join(sorted(meta["file"][test])).encode()).hexdigest()[:8]
print(f"method: {args.features} + {args.classifier}")
print(f"train clips: {train.sum()}  test clips: {test.sum()}  (test set fingerprint {fingerprint})")

# %% Train
seeds = SEEDS[args.classifier]
models = [CLASSIFIERS[args.classifier](s).fit(X[train], y[train]) for s in seeds]

# %% Test
p_firm = np.mean([m.predict_proba(X[test])[:, 1] for m in models], axis=0)
pred = (p_firm >= 0.5).astype(int)
correct = pred == y[test]
n_ok, n = int(correct.sum()), len(correct)

print(f"\nTEST accuracy: {n_ok}/{n} = {n_ok / n:.3f}  (guessing = 0.50)")
print(f"chance of doing this well by guessing (one-sided binomial p): "
      f"{binomtest(n_ok, n, 0.5, alternative='greater').pvalue:.2g}")
if len(models) > 1:
    print(f"single-seed accuracies: {', '.join(f'{m.score(X[test], y[test]):.3f}' for m in models)}")

for col in ["object", "person"]:
    groups = meta[col].values[test]
    print(f"\nby {col}: " + "  ".join(f"{g} {correct[groups == g].mean():.2f}" for g in sorted(set(groups))))

cm = confusion_matrix(y[test], pred, labels=[0, 1])
print("\nconfusion matrix (rows = true, cols = predicted)")
print(pd.DataFrame(cm, index=["true light", "true firm"], columns=["pred light", "pred firm"]))
