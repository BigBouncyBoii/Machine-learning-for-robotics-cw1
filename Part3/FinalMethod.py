# CODE MAP of FinalMethod.py
#   Settings       which features and classifier (defaults = the final method), from the command line
#   Load           clips + data/split.csv (never redrawn, fingerprint 3696c80d) and the cached encoder features
#   Train          scaler + classifier fitted on ALL 120 train clips (nothing is fitted on test clips);
#                  mlp_ensemble: 5 MLPs (seeds 0-4), their firm-probabilities averaged
#   Test           predict the 40 test clips; accuracy overall, per object and per person, vs 50% guessing
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting
#
# GROUP FINAL METHOD = James L's tree node "DINOv2, mean CLS of the middle 6 of 16 frames -> MLP (64), 5 seeds"
# (best mean leave-one-person-out validation accuracy of all members' nodes: 74.3% mean of 5 single MLPs,
# 75.0% as the 5-MLP ensemble trained here; see James L/results/validation.csv).
# Uses James L's code (common.py, extractors/dinov2.py, classifiers/) and cached features (James L/cache/).
# Run the default ONCE for the headline test number; the test clips must not be used to choose the method.
#
# HOW TO RUN (from any folder, venv activated; James L/cache/ is built by James L/main.py if missing)
#   python Part3/FinalMethod.py                                   # final method: dinov2_middle + mlp_ensemble
#   python Part3/FinalMethod.py --features videomae --classifier lr
#   options: --features <any James L feature set>    --classifier lr | mlp_ensemble

# %% Settings
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "James L"))   # James L's modules

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix

import common
from classifiers import lr, mlp_ensemble
from extractors import EXTRACTORS

SET_TO_EXTRACTOR = {s: e for e, m in EXTRACTORS.items() for s in m.FEATURE_SETS}
CLASSIFIERS = {"lr": lr, "mlp_ensemble": mlp_ensemble}

parser = argparse.ArgumentParser()
parser.add_argument("--features", default="dinov2_middle", choices=list(SET_TO_EXTRACTOR))
parser.add_argument("--classifier", default="mlp_ensemble", choices=list(CLASSIFIERS))
args = parser.parse_args()

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# %% Load
clips = common.load_clips()                                   # stops if the test-set fingerprint is wrong
X = EXTRACTORS[SET_TO_EXTRACTOR[args.features]].load(clips, [args.features])[args.features]
train = (clips["set"] == "train").values
test = (clips["set"] == "test").values
y = clips["y"].values                                         # 1 = firm, 0 = light
assert not set(clips["file"][train]) & set(clips["file"][test])
print(f"method: {args.features} + {args.classifier}")
print(f"train clips: {train.sum()}  test clips: {test.sum()}  (test set fingerprint {common.FINGERPRINT})")

# %% Train
model = common.make_pipeline(CLASSIFIERS[args.classifier].make()).fit(X[train], y[train])

# %% Test
p_firm = model.predict_proba(X[test])[:, 1]
pred = (p_firm >= 0.5).astype(int)
correct = pred == y[test]
n_ok, n = int(correct.sum()), len(correct)

print(f"\nTEST accuracy: {n_ok}/{n} = {n_ok / n:.3f}  (guessing = 0.50)")
print(f"chance of doing this well by guessing (one-sided binomial p): "
      f"{binomtest(n_ok, n, 0.5, alternative='greater').pvalue:.2g}")
members = getattr(model[-1], "members_", [])
if members:
    Xs = model[:-1].transform(X[test])                        # the scaler fitted on the train clips
    print(f"single-seed accuracies: {', '.join(f'{m.score(Xs, y[test]):.3f}' for m in members)}")

for col in ["object", "person"]:
    groups = clips[col].values[test]
    print(f"\nby {col}: " + "  ".join(f"{g} {correct[groups == g].mean():.2f}" for g in sorted(set(groups))))

cm = confusion_matrix(y[test], pred, labels=[0, 1])
print("\nconfusion matrix (rows = true, cols = predicted)")
print(pd.DataFrame(cm, index=["true light", "true firm"], columns=["pred light", "pred firm"]))
