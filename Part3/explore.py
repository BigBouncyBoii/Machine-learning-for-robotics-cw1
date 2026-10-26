# CODE MAP of explore.py
#   Settings       feature file, MLP size and regularisation, seeds
#   Load           cached DINOv2 features + data/split.csv; keep TRAIN clips only (the test person is never loaded)
#   Pool           16 frame vectors per clip -> 1 vector per clip (mean over frames of the CLS token)
#   Validate       leave-one-person-out over the train people: train on the others, score on the left-out one
#                  scaler and classifier are fitted inside each fold, on that fold's training clips only
#   Models         logistic regression (baseline) and a 1-hidden-layer MLP
#   Report         validation accuracy per left-out person and object, mean, vs 50% guessing
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting
#
# Tree node: "mean of all frame vectors -> MLP"
#
# HOW TO RUN (from the Part3 folder, after extract_features.py and make_split.py)
#   pip install scikit-learn
#   python explore.py

# %% Settings
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
FEATURES = HERE / "features" / "dinov2_small.npz"
SPLIT = HERE / "data" / "split.csv"

HIDDEN = (64,)        # one hidden layer of 64 ReLU units
ALPHA = 1e-2          # L2 weight penalty: ~90 training clips per fold, so the MLP overfits easily
MAX_ITER = 500
SEEDS = range(5)      # MLP result depends on the random initial weights, so average over 5 seeds

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# %% Load
feat = np.load(FEATURES)
split = pd.read_csv(SPLIT).set_index("file")
clip_set = split.loc[feat["clip"], "set"].values      # match by file name, not by row order
train = clip_set == "train"

F = feat["cls"][train]                    # (n_train, 16, 384)
person = feat["person"][train]
obj = feat["object"][train]
y = (feat["label"][train] == "firm").astype(int)   # 1 = firm, 0 = light
print(f"train clips: {train.sum()}  people: {', '.join(sorted(set(person)))}  "
      f"(test person {', '.join(sorted(set(feat['person'][~train])))} is not used here)")

# %% Pool
X = F.mean(axis=1)                        # (n_train, 384): average the 16 frames of each clip


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


# %% Models
def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))


def mlp(seed):
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=HIDDEN, alpha=ALPHA, max_iter=MAX_ITER, random_state=seed))


# %% Report
pd.set_option("display.float_format", "{:.2f}".format)

lr = leave_one_person_out(logreg)
print("\nLogistic regression (baseline), leave-one-person-out")
print(lr)

runs = [leave_one_person_out(lambda s=s: mlp(s)) for s in SEEDS]
mlp_mean = sum(runs) / len(runs)
seed_accs = [r["all"].mean() for r in runs]
print(f"\nMLP {HIDDEN}, alpha={ALPHA}, mean of {len(SEEDS)} seeds, leave-one-person-out")
print(mlp_mean)

print(f"\nMean validation accuracy (guessing = 0.50)")
print(f"  logistic regression : {lr['all'].mean():.3f}")
print(f"  MLP                 : {np.mean(seed_accs):.3f}  (seeds range {min(seed_accs):.3f}-{max(seed_accs):.3f})")
