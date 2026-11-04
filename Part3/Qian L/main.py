# CODE MAP of main.py   (Qian L, motion: one entry point for every tree node)
#   Settings      command line: which feature sets, which C values; --test; --figures
#   Load          clips + group split (common.load_clips, fingerprint 3696c80d) and the chosen feature sets
#   Validation    every (feature set, C): leave-one-person-out on the 120 TRAIN clips only
#   Save          rows merged into results/validation.csv (one row per node version, re-runs overwrite)
#   Test          --test, ONCE: the version chosen on validation, trained on all 120 train clips, predicts the 40 test clips
#   Figures       --figures: validation overview and train feature distributions (plots.py)
#   AI use        Claude Opus 5.5 (claude-opus-5-5), default setting (restructure); GPT-6.1 Sol (setting: medium); GPT-6 Astra (setting: light)
#
# HOW TO RUN (from anywhere, venv activated; clips missing from cache/ are decoded first, ~3 s each)
#   python "Part3/Qian L/main.py"                              # all three nodes, C = 0.1 and 1 (test clips not used)
#   python "Part3/Qian L/main.py" --features shaking --C 0.1   # one node, one version
#   python "Part3/Qian L/main.py" --figures
#   python "Part3/Qian L/main.py" --test                       # ONCE: validates all versions, then tests the chosen one
# The chosen version is picked by Qian L's rule: highest mean validation accuracy; ties -> fewer features, smaller C.
# It is not hard-coded: optical flow differs slightly between machines (video decoder), and so does the choice.

# %% Settings
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so the imports work from any folder

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix

import common
from classifiers import CLASSIFIERS, lr
from extractors import EXTRACTORS

SET_TO_EXTRACTOR = {s: e for e, m in EXTRACTORS.items() for s in m.FEATURE_SETS}

parser = argparse.ArgumentParser()
parser.add_argument("--features", nargs="+", help="feature sets (speed, shaking, both; default: all)")
parser.add_argument("--C", nargs="+", type=float, help=f"logistic-regression penalties (default {lr.C_VALUES})")
parser.add_argument("--test", action="store_true", help="ONCE: train on 120 train clips, score the 40 test clips")
parser.add_argument("--again", action="store_true", help="allow --test when a test report already exists")
parser.add_argument("--figures", action="store_true", help="save the validation overview and feature plots")
args = parser.parse_args()

warnings.filterwarnings("ignore", category=ConvergenceWarning)

for n in args.features or []:
    if n not in SET_TO_EXTRACTOR:
        raise SystemExit(f"unknown feature set '{n}'; choose from {list(SET_TO_EXTRACTOR)}")
feature_sets = args.features or list(SET_TO_EXTRACTOR)
c_values = args.C or lr.C_VALUES


def version_name(C):
    return f"lr C={C:g}"


# %% Load
clips = common.load_clips()
X_of = {}
for s in feature_sets:
    X_of.update(EXTRACTORS[SET_TO_EXTRACTOR[s]].load(clips, [s]))
tr = (clips["set"] == "train").values
print(f"test-set fingerprint {common.FINGERPRINT} ok; {tr.sum()} train clips, {(~tr).sum()} test clips")
for s in feature_sets:
    usable = np.isfinite(X_of[s]).all(axis=1)
    print(f"  {s}: usable motion in {usable[tr].sum()}/{tr.sum()} train clips "
          f"(the rest get the training majority label)")

# %% Validation
rows, oof = [], {}
for s in feature_sets:
    for C in c_values:
        r, p = common.validate(X_of[s], clips, lambda C=C: lr.make(C=C), return_oof=True)
        rows.append({"extractor": SET_TO_EXTRACTOR[s], "features": s, "classifier": version_name(C),
                     "n_features": X_of[s].shape[1], "C": C, **r})
        oof[(s, C)] = p
val = pd.DataFrame(rows)
people = sorted(clips["person"].unique())
print(f"\nLeave-one-person-out validation on the 120 train clips (guessing = 50 %); the test clips are not used")
print(f"{'features':<10}{'classifier':<12}" + "".join(f"{'val ' + p:>7}" for p in people)
      + f"{'mean':>7}{'logloss':>9}{'hard':>7}{'soft':>7}{'train':>7}")
for _, r in val.iterrows():
    print(f"{r['features']:<10}{r['classifier']:<12}" + "".join(f"{100 * r[f'val_{p}']:>6.1f}%" for p in people)
          + f"{100 * r['mean_acc']:>6.1f}%{r['mean_logloss']:>9.3f}{100 * r['hard_acc']:>6.0f}%"
          + f"{100 * r['soft_acc']:>6.0f}%{100 * r['train_acc']:>6.0f}%")
best = val.sort_values(["mean_acc", "n_features", "C"], ascending=[False, True, True]).iloc[0]
print(f"\nchosen (highest mean accuracy; ties -> fewer features, smaller C): "
      f"{best['features']} + {best['classifier']} ({100 * best['mean_acc']:.1f} %)")

# %% Save
common.RESULTS.mkdir(exist_ok=True)
path = common.RESULTS / "validation.csv"
merged = val
if path.exists():
    old = pd.read_csv(path)
    keep = ~old.set_index(["features", "classifier"]).index.isin(val.set_index(["features", "classifier"]).index)
    merged = pd.concat([old[keep], val], ignore_index=True)
order = {s: i for i, s in enumerate(SET_TO_EXTRACTOR)}
merged = merged.sort_values(["features", "classifier"], key=lambda c: c.map(order) if c.name == "features" else c)
merged.to_csv(path, index=False, float_format="%.4f")
print(f"saved {path.relative_to(common.HERE)}")

# %% Test
if args.test:
    s, C = best["features"], best["C"]                      # chosen on validation only (see Validation)
    report_path = common.RESULTS / "test_report.txt"
    if report_path.exists() and not args.again:
        raise SystemExit(f"{report_path.name} exists: the test has already been run (add --again to overwrite)")
    X, y = X_of[s], clips["y"].values
    model = common.make_pipeline(lr.make(C=C)).fit(X[tr], y[tr])
    test = clips.loc[~tr, ["file", "person", "object", "label", "y"]].copy()
    test["p_firm"] = model.predict_proba(X[~tr])[:, 1]
    test["model_used"] = np.isfinite(X[~tr]).all(axis=1)
    test["pred"] = (test["p_firm"] >= 0.5).astype(int)
    test["correct"] = test["pred"] == test["y"]
    k, n = int(test["correct"].sum()), len(test)
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    pval = binomtest(k, n, 0.5, alternative="greater").pvalue
    cm = confusion_matrix(test["y"], test["pred"], labels=[0, 1])
    lines = [f"FINAL TEST, Qian L motion: {s} + {version_name(C)}",
             f"test-set fingerprint {common.FINGERPRINT}; trained on the 120 train clips, tested on the 40 test clips",
             f"usable motion in {int(test['model_used'].sum())}/{n} test clips (the rest: training majority label)",
             "", f"test accuracy  {100 * k / n:.1f} % ({k}/{n}), 95 % CI {100 * ci.low:.1f}-{100 * ci.high:.1f} %",
             f"vs guessing    one-sided binomial p = {pval:.4f}", ""]
    lines += [f"by {col}: " + ", ".join(f"{g} {100 * d['correct'].mean():.0f} % ({int(d['correct'].sum())}/{len(d)})"
                                        for g, d in test.groupby(col)) for col in ("person", "object", "label")]
    lines += ["", "confusion matrix      predicted light  predicted firm",
              f"  true light ({cm[0].sum()})        {cm[0, 0]:>6}           {cm[0, 1]:>6}",
              f"  true firm  ({cm[1].sum()})        {cm[1, 0]:>6}           {cm[1, 1]:>6}",
              "", "NOTE: some of these clips were seen in Qian L's earlier project with another protocol",
              "(archive/results/historical_exposure.csv); do not describe the test set as never seen."]
    report = "\n".join(lines)
    print("\n" + report)
    report_path.write_text(report + "\n", encoding="utf-8")
    test.to_csv(common.RESULTS / "test_predictions.csv", index=False, float_format="%.4f")
    print(f"\nsaved {report_path.relative_to(common.HERE)} and test_predictions.csv")

# %% Figures
if args.figures:
    import plots
    plots.validation_overview(val, oof, clips)
    plots.feature_distributions(clips)
