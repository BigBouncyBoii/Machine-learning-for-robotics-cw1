# CODE MAP of validate_split.py   (colour method on the GROUP split: validation only, test clips never used)
#   Settings      paths, required test-set fingerprint, output file
#   Load split    read Part3/data/split.csv (never redrawn), check the fingerprint, keep the 120 train clips
#   Validation    leave-one-person-out on the train clips (4 folds: A, B, C, D, 30 clips each);
#                 every input x model from colour_method; imputer + scaler + classifier fitted inside each fold
#   Choose        lowest mean validation LOG LOSS among the MLP ensembles (logistic regression = baseline only)
#   Save          validation table -> group_split/validation.csv, choice -> group_split/choice.json
#   AI use        Claude Opus 5.5 (claude-opus-5-5), default setting
#
# HOW TO RUN (from the repo root)
#   .venv\Scripts\python "James F\colour_method\validate_split.py"
#   Then, once, when the method is final:  .venv\Scripts\python "James F\colour_method\final_test_split.py"
#
# Features: per-frame finger colour from colour_method (MediaPipe + skin mask, no labels used), cached for
# all 160 clips. Only the 120 train clips enter this file's learning.

# %% Settings
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import log_loss

import colour_method as cm

SPLIT = cm.PART3 / "split.csv"
FINGERPRINT = "3696c80d"
OUT = Path(__file__).resolve().parent / "group_split"


def fingerprint(split):
    test_files = split.loc[split["set"] == "test", "file"]
    return hashlib.md5("\n".join(sorted(test_files)).encode()).hexdigest()[:8]


# %% Load split
split = pd.read_csv(SPLIT)
fp = fingerprint(split)
print(f"test-set fingerprint {fp} (must be {FINGERPRINT})")
if fp != FINGERPRINT:
    raise SystemExit("wrong split: stop")

set_of = dict(zip(split["file"], split["set"]))
is_train = cm.df["file"].map(set_of).eq("train").values     # rows of colour_method's feature arrays
assert is_train.sum() == 120
tr_df = cm.df[is_train].reset_index(drop=True)              # the 40 test clips are dropped here, before anything
y = tr_df["y"].values
groups = tr_df["person"].values
X_of = {name: X[is_train] for name, X in cm.INPUTS.items()}
print(f"{len(tr_df)} train clips: " + ", ".join(f"{p} {n}" for p, n in tr_df["person"].value_counts().sort_index().items()))

# %% Validation
people = sorted(set(groups))
jobs = [(name, kind, p) for name in cm.INPUTS for kind in cm.MODELS for p in people]
probs = Parallel(n_jobs=-1)(
    delayed(cm.fit_predict)(kind, X_of[name][groups != p], y[groups != p], X_of[name][groups == p])
    for name, kind, p in jobs)
prob_of = dict(zip(jobs, probs))

rows = []
for name in cm.INPUTS:
    for kind in cm.MODELS:
        p_va = np.zeros(len(y))
        fold_acc, fold_loss = {}, {}
        for p in people:
            m = groups == p
            p_va[m] = prob_of[(name, kind, p)]
            fold_acc[p] = ((p_va[m] > 0.5) == y[m]).mean()
            fold_loss[p] = log_loss(y[m], p_va[m], labels=[0, 1])
        correct = (p_va > 0.5) == y
        rows.append({"input": name, "model": kind,
                     **{f"val_{p}": fold_acc[p] for p in people},
                     "mean_acc": np.mean(list(fold_acc.values())),
                     "mean_logloss": np.mean(list(fold_loss.values())),
                     "hard_acc": correct[tr_df["object"] == "hard"].mean(),
                     "soft_acc": correct[tr_df["object"] == "soft"].mean()})
val = pd.DataFrame(rows)

print("\nLeave-one-person-out validation on the 120 train clips (guessing = 50 %, log loss 0.693)")
print(f"{'input':<32}{'model':<22}" + "".join(f"{'val ' + p:>7}" for p in people)
      + f"{'mean':>7}{'logloss':>9}{'hard':>7}{'soft':>7}")
for _, r in val.iterrows():
    print(f"{r['input']:<32}{r['model']:<22}" + "".join(f"{100 * r[f'val_{p}']:>6.0f}%" for p in people)
          + f"{100 * r['mean_acc']:>6.1f}%{r['mean_logloss']:>9.3f}{100 * r['hard_acc']:>6.0f}%{100 * r['soft_acc']:>6.0f}%")

# %% Choose
cand = val[val["model"].isin(cm.SELECTABLE)]
best = cand.loc[cand["mean_logloss"].idxmin()]
print(f"\nchosen by mean validation LOG LOSS: '{best['input']}' + {best['model']}"
      f"  (val acc {100 * best['mean_acc']:.1f} %, log loss {best['mean_logloss']:.3f})")
print("the test clips have not been used.")

# %% Save
OUT.mkdir(exist_ok=True)
val.to_csv(OUT / "validation.csv", index=False, float_format="%.4f")
(OUT / "choice.json").write_text(json.dumps({
    "input": best["input"], "model": best["model"], "criterion": "mean validation log loss",
    "val_acc": round(float(best["mean_acc"]), 4), "val_logloss": round(float(best["mean_logloss"]), 4),
    "fingerprint": fp, "n_ensemble": cm.N_ENSEMBLE, "seeds": f"{cm.SEED}..{cm.SEED + cm.N_ENSEMBLE - 1}",
}, indent=2))
print(f"saved validation.csv and choice.json in {OUT}")
