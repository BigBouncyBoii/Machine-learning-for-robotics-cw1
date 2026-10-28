# CODE MAP of FinalMethod.py
#   Settings       feature file, final MLP settings (copied from the chosen explore.py node), seeds
#   Load           cached DINOv2 features + data/split.csv, split into TRAIN (128 clips) and TEST (32 clips)
#   Pool           16 frame vectors per clip -> 1 vector per clip (mean over frames of the CLS token)
#   Train          scaler + MLP fitted on ALL train clips (nothing is fitted on test clips)
#   Test           predict the test clips once; accuracy overall, per object and per person, vs 50% guessing
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting
#
# Run ONLY when exploring is finished. Running it and then changing the method means the test set
# is no longer an unseen test. If the final tree node changes, update the Settings/Pool sections to match.
#
# HOW TO RUN (from the Part3 folder, after extract_features.py and make_split.py)
#   ../.venv/bin/python FinalMethod.py

# %% Settings
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
FEATURES = HERE / "features" / "dinov2_small.npz"
SPLIT = HERE / "data" / "split.csv"

HIDDEN = (64,)        # same as explore.py
ALPHA = 1e-2
MAX_ITER = 500
SEEDS = range(5)      # 5 MLPs with different initial weights; their firm-probabilities are averaged

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# %% Load
feat = np.load(FEATURES)
split = pd.read_csv(SPLIT).set_index("file")
clip_set = split.loc[feat["clip"], "set"].values      # match by file name, not by row order
train, test = clip_set == "train", clip_set == "test"
y_all = (feat["label"] == "firm").astype(int)          # 1 = firm, 0 = light

# %% Pool
X_all = feat["cls"].mean(axis=1)                       # (n_clips, 384)
X_tr, y_tr = X_all[train], y_all[train]
X_te, y_te = X_all[test], y_all[test]
print(f"train clips: {train.sum()}  test clips: {test.sum()}")

# %% Train
models = [make_pipeline(StandardScaler(), MLPClassifier(
    hidden_layer_sizes=HIDDEN, alpha=ALPHA, max_iter=MAX_ITER, random_state=s)).fit(X_tr, y_tr)
    for s in SEEDS]

# %% Test
p_firm = np.mean([m.predict_proba(X_te)[:, 1] for m in models], axis=0)
pred = (p_firm >= 0.5).astype(int)
correct = pred == y_te
n_ok, n = int(correct.sum()), len(correct)

print(f"\nTEST accuracy: {n_ok}/{n} = {n_ok / n:.3f}  (guessing = 0.50)")
print(f"chance of doing this well by guessing (one-sided binomial p): {binomtest(n_ok, n, 0.5, alternative='greater').pvalue:.4f}")
print(f"single-seed accuracies: {', '.join(f'{m.score(X_te, y_te):.3f}' for m in models)}")

for col in ["object", "person"]:
    groups = feat[col][test]
    print(f"\nby {col}: " + "  ".join(f"{g} {correct[groups == g].mean():.2f}" for g in sorted(set(groups))))

cm = confusion_matrix(y_te, pred, labels=[0, 1])
print("\nconfusion matrix (rows = true, cols = predicted)")
print(pd.DataFrame(cm, index=["true light", "true firm"], columns=["pred light", "pred firm"]))
