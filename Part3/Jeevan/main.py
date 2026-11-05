# CODE MAP of main.py   (Jeevan, finger joint angles: one entry point for every tree node)
#   Settings      command line: which feature sets, which classifiers; --test; --figures
#   Load          clips + group split (common.load_clips, fingerprint 3696c80d) and the chosen feature sets
#   Validation    every (feature set, classifier): leave-one-person-out on the 120 TRAIN clips only
#   Save          rows merged into results/validation.csv (one row per node version, re-runs overwrite)
#   Test          --test, ONCE, one version: train on all 120 train clips, predict the 40 test clips
#   Figures       --figures: accuracy per held-out person and pooled validation confusion matrix (plots.py)
#   AI use        Claude Opus 5.5 (claude-opus-5-5), default setting (restructure); original code: Claude Opus 5.5 (claude-opus-5-5), default setting
#
# HOW TO RUN (from anywhere, venv activated; the first run builds cache/ from data/videos_480p, ~10 min)
#   python Part3/Jeevan/main.py                     # validate (test clips not used)
#   python Part3/Jeevan/main.py --figures           # + figures in results/figures/
#   python Part3/Jeevan/main.py --test              # ONCE: train on 120 train clips, score the 40 test clips

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
from classifiers import CLASSIFIERS
from extractors import EXTRACTORS

SET_TO_EXTRACTOR = {s: e for e, m in EXTRACTORS.items() for s in m.FEATURE_SETS}

parser = argparse.ArgumentParser()
parser.add_argument("--features", nargs="+", help="feature sets or extractor names (default: all)")
parser.add_argument("--classifier", nargs="+", choices=list(CLASSIFIERS), help="default: all")
parser.add_argument("--test", action="store_true", help="ONCE: train on 120 train clips, score the 40 test clips")
parser.add_argument("--again", action="store_true", help="allow --test when a test report already exists")
parser.add_argument("--figures", action="store_true", help="save per-person accuracy and confusion matrix")
args = parser.parse_args()

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def expand(names):
    out = []
    for n in names:
        if n in EXTRACTORS:
            out += list(EXTRACTORS[n].FEATURE_SETS)
        elif n in SET_TO_EXTRACTOR:
            out.append(n)
        else:
            raise SystemExit(f"unknown feature set '{n}'; choose from {list(EXTRACTORS) + list(SET_TO_EXTRACTOR)}")
    return out


feature_sets = expand(args.features or list(SET_TO_EXTRACTOR))
classifiers = args.classifier or list(CLASSIFIERS)
if (args.test or args.figures) and (len(feature_sets) != 1 or len(classifiers) != 1):
    raise SystemExit("--test and --figures use exactly one version: give one feature set and one classifier")

# %% Load
clips = common.load_clips()
X_of = {}
for e in dict.fromkeys(SET_TO_EXTRACTOR[s] for s in feature_sets):   # each extractor loads once
    X_of.update(EXTRACTORS[e].load(clips, [s for s in feature_sets if SET_TO_EXTRACTOR[s] == e]))
tr = (clips["set"] == "train").values
print(f"test-set fingerprint {common.FINGERPRINT} ok; {tr.sum()} train clips, {(~tr).sum()} test clips")
_, rate = EXTRACTORS["joint_angles"].load_cache(clips)
print(f"MediaPipe found the hand in a median of {100 * np.median(rate):.0f} % of frames; "
      f"in no frame for {(rate == 0).sum()} clips (those get zero angle features)")

# %% Validation
rows, oof = [], {}
for s in feature_sets:
    for c in classifiers:
        runs = []
        for seed in CLASSIFIERS[c].SEEDS:
            r, p = common.validate(X_of[s], clips, lambda: CLASSIFIERS[c].make(seed), return_oof=True)
            runs.append(r)
            oof[(s, c)] = p
        rows.append({"extractor": SET_TO_EXTRACTOR[s], "features": s, "classifier": c,
                     **{k: np.mean([r[k] for r in runs]) for k in runs[0]}})
val = pd.DataFrame(rows)
people = sorted(clips["person"].unique())
print(f"\nLeave-one-person-out validation on the 120 train clips (guessing = 50 %); the test clips are not used")
print(f"{'features':<16}{'classifier':<12}" + "".join(f"{'val ' + p:>7}" for p in people)
      + f"{'mean':>7}{'logloss':>9}{'hard':>7}{'soft':>7}{'train':>7}")
for _, r in val.iterrows():
    print(f"{r['features']:<16}{r['classifier']:<12}" + "".join(f"{100 * r[f'val_{p}']:>6.1f}%" for p in people)
          + f"{100 * r['mean_acc']:>6.1f}%{r['mean_logloss']:>9.3f}{100 * r['hard_acc']:>6.0f}%"
          + f"{100 * r['soft_acc']:>6.0f}%{100 * r['train_acc']:>6.0f}%")

# %% Save
common.RESULTS.mkdir(exist_ok=True)
path = common.RESULTS / "validation.csv"
merged = val
if path.exists():
    old = pd.read_csv(path)
    keep = ~old.set_index(["features", "classifier"]).index.isin(val.set_index(["features", "classifier"]).index)
    merged = pd.concat([old[keep], val], ignore_index=True)
merged.to_csv(path, index=False, float_format="%.4f")
print(f"saved {path.relative_to(common.HERE)}")

# %% Test
if args.test:
    (s,), (c,) = feature_sets, classifiers
    report_path = common.RESULTS / "test_report.txt"
    if report_path.exists() and not args.again:
        raise SystemExit(f"{report_path.name} exists: the test has already been run (add --again to overwrite)")
    X, y = X_of[s], clips["y"].values
    model = common.make_pipeline(CLASSIFIERS[c].make()).fit(X[tr], y[tr])
    test = clips.loc[~tr, ["file", "person", "object", "label", "y"]].copy()
    test["p_firm"] = model.predict_proba(X[~tr])[:, 1]
    test["pred"] = (test["p_firm"] > 0.5).astype(int)
    test["correct"] = test["pred"] == test["y"]
    k, n = int(test["correct"].sum()), len(test)
    pval = binomtest(k, n, 0.5, alternative="greater").pvalue
    cm = confusion_matrix(test["y"], test["pred"], labels=[0, 1])
    lines = [f"FINAL TEST, Jeevan joint angles: {s} + {c}",
             f"test-set fingerprint {common.FINGERPRINT}; trained on the 120 train clips, tested on the 40 test clips",
             "", f"test accuracy  {100 * k / n:.1f} % ({k}/{n}), one-sided binomial p vs 50 % = {pval:.4f}", ""]
    lines += [f"by {col}: " + ", ".join(f"{g} {100 * d['correct'].mean():.0f} % ({int(d['correct'].sum())}/{len(d)})"
                                        for g, d in test.groupby(col)) for col in ("person", "object", "label")]
    lines += ["", "confusion matrix      predicted light  predicted firm",
              f"  true light ({cm[0].sum()})        {cm[0, 0]:>6}           {cm[0, 1]:>6}",
              f"  true firm  ({cm[1].sum()})        {cm[1, 0]:>6}           {cm[1, 1]:>6}"]
    report = "\n".join(lines)
    print("\n" + report)
    report_path.write_text(report + "\n", encoding="utf-8")
    test.to_csv(common.RESULTS / "test_predictions.csv", index=False, float_format="%.4f")
    print(f"\nsaved {report_path.relative_to(common.HERE)} and test_predictions.csv")

# %% Figures
if args.figures:
    import plots
    (s,), (c,) = feature_sets, classifiers
    r = val.iloc[0]
    plots.per_person(people, [r[f"val_{p}"] for p in people], r["mean_acc"])
    plots.confusion(clips["y"].values[tr], (oof[(s, c)] > 0.5).astype(int))
