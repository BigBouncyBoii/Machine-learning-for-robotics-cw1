# CODE MAP of final_test_split.py   (colour method on the GROUP split: the ONE final test)
#   Settings      paths, required test-set fingerprint
#   Load split    read Part3/data/split.csv, check the fingerprint
#   Load choice   group_split/choice.json written by validate_split.py (input + model chosen on validation)
#   Fit           refit the chosen pipeline (imputer + scaler + 10-MLP ensemble, seeds fixed) on all 120 train clips
#   Test          predict the 40 test clips once: accuracy, 95 % CI, p vs 50 %, per person / object, confusion matrix
#   Save          group_split/test_report.txt, test_predictions.csv
#   AI use        Claude Opus 5.5 (claude-opus-5-5), default setting
#
# HOW TO RUN (from the repo root), ONCE, after validate_split.py and only when the method is final:
#   .venv\Scripts\python "James F\colour_method\final_test_split.py"
#
# NOTE FOR THE REPORT  The classifier design (10-MLP ensemble, alpha grid 0.3-10, log-loss selection) was
# decided after an earlier person-held-out analysis (test_A..D.py / performance.py) in which these 40 test
# clips were also predicted. State this when reporting the test number.

# %% Settings
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix

import colour_method as cm

SPLIT = cm.PART3 / "split.csv"
FINGERPRINT = "3696c80d"
OUT = Path(__file__).resolve().parent / "group_split"

# %% Load split
split = pd.read_csv(SPLIT)
test_files = split.loc[split["set"] == "test", "file"]
fp = hashlib.md5("\n".join(sorted(test_files)).encode()).hexdigest()[:8]
print(f"test-set fingerprint {fp} (must be {FINGERPRINT})")
if fp != FINGERPRINT:
    raise SystemExit("wrong split: stop")
set_of = dict(zip(split["file"], split["set"]))
is_train = cm.df["file"].map(set_of).eq("train").values
is_test = cm.df["file"].map(set_of).eq("test").values
assert is_train.sum() == 120 and is_test.sum() == 40

# %% Load choice
choice = json.loads((OUT / "choice.json").read_text())
assert choice["fingerprint"] == fp
print(f"method (chosen on validation by {choice['criterion']}): '{choice['input']}' + {choice['model']}")

# %% Fit
X = cm.INPUTS[choice["input"]]
y = cm.df["y"].values
model = cm.make_model(choice["model"]).fit(X[is_train], y[is_train])

# %% Test
lines = []


def out(s=""):
    lines.append(s)


test = cm.df.loc[is_test, ["file", "person", "object", "label", "y"]].copy()
test["p_firm"] = model.predict_proba(X[is_test])[:, 1]
test["pred"] = (test["p_firm"] > 0.5).astype(int)
test["correct"] = test["pred"] == test["y"]
k, n = int(test["correct"].sum()), len(test)
ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
pval = binomtest(k, n, 0.5, alternative="greater").pvalue

out("FINAL TEST, colour method on the group split")
out(f"test-set fingerprint {fp}")
out(f"method: '{choice['input']}' + {choice['model']} ({choice['n_ensemble']} MLPs, seeds {choice['seeds']}),")
out(f"        chosen by {choice['criterion']} (val acc {100 * choice['val_acc']:.1f} %, log loss {choice['val_logloss']:.3f})")
out(f"trained on the 120 train clips, tested once on the 40 test clips")
out()
out(f"test accuracy  {100 * k / n:.1f} % ({k}/{n}), 95 % CI {100 * ci.low:.1f}-{100 * ci.high:.1f} %")
out(f"vs guessing    one-sided binomial p = {pval:.4f}")
out()
for col in ("person", "object", "label"):
    out(f"by {col}: " + ", ".join(f"{g} {100 * d['correct'].mean():.0f} % ({int(d['correct'].sum())}/{len(d)})"
                                  for g, d in test.groupby(col)))
c = confusion_matrix(test["y"], test["pred"], labels=[0, 1])
out()
out("confusion matrix      predicted light  predicted firm")
out(f"  true light ({c[0].sum()})        {c[0, 0]:>6}           {c[0, 1]:>6}")
out(f"  true firm  ({c[1].sum()})        {c[1, 0]:>6}           {c[1, 1]:>6}")
out()
out("NOTE: the classifier design was decided after an earlier person-held-out analysis in which these")
out("40 clips were also predicted (test_A..D.py / performance.py).")

# %% Save
report = "\n".join(lines)
print("\n" + report)
OUT.mkdir(exist_ok=True)
(OUT / "test_report.txt").write_text(report + "\n", encoding="utf-8")
test.to_csv(OUT / "test_predictions.csv", index=False, float_format="%.4f")
print(f"\nsaved test_report.txt and test_predictions.csv in {OUT}")
