# CODE MAP of explore.py
#   Settings       which feature version (command-line argument, see common.FEATURES)
#   Load           cached features + data/split.csv; keep TRAIN clips only (test clips are never used)
#   Validate       leave-one-person-out over the train people: train on the others, score on the left-out one
#                  scaler and classifier are fitted inside each fold, on that fold's training clips only
#   Models         logistic regression (baseline) and a 1-hidden-layer MLP, from common.CLASSIFIERS
#   Report         validation accuracy per left-out person and object, mean, vs 50% guessing
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting
#
# One run = one tree node. The number for the tree is the mean validation accuracy, never the train accuracy.
#
# HOW TO RUN (from the Part3 folder, venv activated, after extract_features.py and make_split.py)
#   python explore.py                      # default: dinov2_cls_mean
#   python explore.py videomae             # any name in common.FEATURES

# %% Settings
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from common import CLASSIFIERS, FEATURES, SEEDS, load_features

parser = argparse.ArgumentParser()
parser.add_argument("features", nargs="?", default="dinov2_cls_mean", choices=FEATURES)
NAME = parser.parse_args().features

# classifier settings are in common.CLASSIFIERS (shared with FinalMethod.py)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# %% Load
X_all, meta = load_features(NAME)
train = (meta["set"] == "train").values
X = X_all[train]
person = meta["person"].values[train]
obj = meta["object"].values[train]
y = (meta["label"].values[train] == "firm").astype(int)   # 1 = firm, 0 = light
print(f"features: {NAME}  {X.shape[1]}-dim")
print(f"train clips: {train.sum()}  people: {', '.join(sorted(set(person)))}  "
      f"({(~train).sum()} test clips are not used here)")


# %% Validate
def leave_one_person_out(make_model):
    """Return a table of validation accuracy for each left-out person and object."""
    rows = []
    for p in sorted(set(person)):
        tr, va = person != p, person == p
        model = make_model().fit(X[tr], y[tr])
        correct = model.predict(X[va]) == y[va]
        rows.append({"left out": p, "all": correct.mean(),
                     **{o: correct[obj[va] == o].mean() for o in sorted(set(obj))},
                     "train acc": model.score(X[tr], y[tr])})
    return pd.DataFrame(rows).set_index("left out")


# %% Report
pd.set_option("display.float_format", "{:.2f}".format)

lr = leave_one_person_out(lambda: CLASSIFIERS["lr"](0))
print("\nLogistic regression (baseline), leave-one-person-out")
print(lr)

runs = [leave_one_person_out(lambda s=s: CLASSIFIERS["mlp"](s)) for s in SEEDS["mlp"]]
mlp_mean = sum(runs) / len(runs)
seed_accs = [r["all"].mean() for r in runs]
print(f"\nMLP (64,), mean of {len(runs)} seeds, leave-one-person-out")
print(mlp_mean)

print(f"\nMean validation accuracy, {NAME} (guessing = 0.50)")
print(f"  logistic regression : {lr['all'].mean():.3f}")
print(f"  MLP                 : {np.mean(seed_accs):.3f}  (seeds range {min(seed_accs):.3f}-{max(seed_accs):.3f})")
