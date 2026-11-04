# CODE MAP of main.py   (James F, finger-colour branch: one entry point for every tree node)
#   Settings      command line: which feature sets, which classifiers, which L2 penalties; --test; --figures
#   Load          clips + group split (common.load_clips, fingerprint 3696c80d) and the chosen feature sets
#   Validation    every (feature set, classifier, alpha): leave-one-person-out on the 120 TRAIN clips only
#   Save          rows merged into results/validation.csv (one row per node version, re-runs overwrite)
#   Test          --test, ONCE, one version: train on all 120 train clips, predict the 40 test clips
#   Figures       --figures: validation bars and redness-over-time curves (plots.py)
#   AI use        Claude Opus 5.5 (claude-opus-5-5), GPT-6 Astra, GPT-6.1 Sol; all Medium setting
#
# HOW TO RUN (from anywhere, venv activated; first run builds the caches from data/videos_480p, ~15 min)
#   python "Part3/James F/main.py"                                    # validate every node (test clips not used)
#   python "Part3/James F/main.py" --features hand_colour --classifier lr        # node 1
#   python "Part3/James F/main.py" --features finger_patches --classifier lr mlp # node 2
#   python "Part3/James F/main.py" --features finger_patches --classifier mlp_ensemble --alpha 0.3 1 3 10   # node 3
#   python "Part3/James F/main.py" --test                             # ONCE: chosen version on the 40 test clips
#   --features takes feature-set names (e.g. nail_mid_a_curve) or extractor names (hand_colour, finger_patches)

# %% Settings
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so the imports work from any folder

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binomtest
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix

import common
from classifiers import CLASSIFIERS, TAKES_ALPHA
from extractors import EXTRACTORS

SET_TO_EXTRACTOR = {s: e for e, m in EXTRACTORS.items() for s in m.FEATURE_SETS}
DEFAULT_ALPHAS = {"mlp": [1.0], "mlp_ensemble": [0.3, 1.0, 3.0, 10.0]}
# chosen on validation (lowest mean validation log loss among the MLP ensembles); used by --test
CHOSEN = {"features": "nail_mid_lab_curves", "classifier": "mlp_ensemble", "alpha": 10.0}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--features", nargs="+", help="feature sets or extractor names (default: all)")
parser.add_argument("--classifier", nargs="+", choices=list(CLASSIFIERS), help="default: all")
parser.add_argument("--alpha", nargs="+", type=float, help="L2 penalties for mlp / mlp_ensemble")
parser.add_argument("--test", action="store_true", help="ONCE: train on 120 train clips, score the 40 test clips")
parser.add_argument("--again", action="store_true", help="allow --test when a test report already exists")
parser.add_argument("--figures", action="store_true", help="save validation bars and redness curves")
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


if args.test:
    feature_sets = expand(args.features or [CHOSEN["features"]])
    versions = [(c, a) for c in (args.classifier or [CHOSEN["classifier"]])
                for a in ((args.alpha or [CHOSEN["alpha"]]) if c in TAKES_ALPHA else [None])]
    if len(feature_sets) != 1 or len(versions) != 1:
        raise SystemExit("--test runs exactly one version: give one feature set, one classifier, one alpha")
else:
    feature_sets = expand(args.features or list(SET_TO_EXTRACTOR))
    versions = [(c, a) for c in (args.classifier or list(CLASSIFIERS))
                for a in ((args.alpha or DEFAULT_ALPHAS[c]) if c in TAKES_ALPHA else [None])]


def make_classifier(name, alpha):
    module = CLASSIFIERS[name]
    return (lambda: module.make(alpha=alpha)) if alpha is not None else module.make


def version_name(name, alpha):
    return name if alpha is None else f"{name} a={alpha:g}"


# %% Load
clips = common.load_clips()
X_of = {}
for e in dict.fromkeys(SET_TO_EXTRACTOR[s] for s in feature_sets):   # each extractor loads once
    X_of.update(EXTRACTORS[e].load(clips))
tr = (clips["set"] == "train").values
print(f"test-set fingerprint {common.FINGERPRINT} ok; {tr.sum()} train clips, {(~tr).sum()} test clips")

# %% Validation
if not args.test:
    jobs = [(s, c, a) for s in feature_sets for c, a in versions]
    print(f"\nLeave-one-person-out validation on the 120 train clips, {len(jobs)} versions "
          "(guessing = 50 %, log loss 0.693); the test clips are not used")
    res = Parallel(n_jobs=-1)(delayed(common.validate)(X_of[s], clips, make_classifier(c, a)) for s, c, a in jobs)
    val = pd.DataFrame([{"extractor": SET_TO_EXTRACTOR[s], "features": s, "classifier": version_name(c, a), **r}
                        for (s, c, a), r in zip(jobs, res)])
    people = sorted(clips["person"].unique())
    print(f"{'features':<22}{'classifier':<22}" + "".join(f"{'val ' + p:>7}" for p in people)
          + f"{'mean':>7}{'logloss':>9}{'hard':>7}{'soft':>7}")
    for _, r in val.iterrows():
        print(f"{r['features']:<22}{r['classifier']:<22}" + "".join(f"{100 * r[f'val_{p}']:>6.0f}%" for p in people)
              + f"{100 * r['mean_acc']:>6.1f}%{r['mean_logloss']:>9.3f}{100 * r['hard_acc']:>6.0f}%"
              + f"{100 * r['soft_acc']:>6.0f}%")
    best_acc = val.loc[val["mean_acc"].idxmax()]
    best_loss = val.loc[val["mean_logloss"].idxmin()]
    print(f"\nbest mean accuracy: {best_acc['features']} + {best_acc['classifier']} ({100 * best_acc['mean_acc']:.1f} %)")
    print(f"best mean log loss: {best_loss['features']} + {best_loss['classifier']} ({best_loss['mean_logloss']:.3f})")

    # %% Save
    common.RESULTS.mkdir(exist_ok=True)
    path = common.RESULTS / "validation.csv"
    if path.exists():
        old = pd.read_csv(path)
        keep = ~old.set_index(["features", "classifier"]).index.isin(val.set_index(["features", "classifier"]).index)
        val = pd.concat([old[keep], val], ignore_index=True)
    order = {s: i for i, s in enumerate(SET_TO_EXTRACTOR)}
    val = val.sort_values(["features", "classifier"], key=lambda c: c.map(order) if c.name == "features" else c)
    val.to_csv(path, index=False, float_format="%.4f")
    print(f"saved {path.relative_to(common.HERE)}")

# %% Test
if args.test:
    (s,), ((c, a),) = feature_sets, versions
    report_path = common.RESULTS / "test_report.txt"
    if report_path.exists() and not args.again:
        raise SystemExit(f"{report_path.name} exists: the test has already been run (add --again to overwrite)")
    X, y = X_of[s], clips["y"].values
    model = common.make_pipeline(make_classifier(c, a)()).fit(X[tr], y[tr])
    test = clips.loc[~tr, ["file", "person", "object", "label", "y"]].copy()
    test["p_firm"] = model.predict_proba(X[~tr])[:, 1]
    test["pred"] = (test["p_firm"] > 0.5).astype(int)
    test["correct"] = test["pred"] == test["y"]
    k, n = int(test["correct"].sum()), len(test)
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    pval = binomtest(k, n, 0.5, alternative="greater").pvalue
    cm = confusion_matrix(test["y"], test["pred"], labels=[0, 1])
    lines = [f"FINAL TEST, James F colour branch: {s} + {version_name(c, a)}",
             f"test-set fingerprint {common.FINGERPRINT}; trained on the 120 train clips, tested on the 40 test clips",
             "",
             f"test accuracy  {100 * k / n:.1f} % ({k}/{n}), 95 % CI {100 * ci.low:.1f}-{100 * ci.high:.1f} %",
             f"vs guessing    one-sided binomial p = {pval:.4f}",
             ""]
    lines += [f"by {col}: " + ", ".join(f"{g} {100 * d['correct'].mean():.0f} % ({int(d['correct'].sum())}/{len(d)})"
                                        for g, d in test.groupby(col)) for col in ("person", "object", "label")]
    lines += ["", "confusion matrix      predicted light  predicted firm",
              f"  true light ({cm[0].sum()})        {cm[0, 0]:>6}           {cm[0, 1]:>6}",
              f"  true firm  ({cm[1].sum()})        {cm[1, 0]:>6}           {cm[1, 1]:>6}",
              "", "NOTE: these 40 clips were also predicted in an earlier person-held-out analysis (archive/),",
              "before the group split was used; the classifier design was decided after it."]
    report = "\n".join(lines)
    print("\n" + report)
    common.RESULTS.mkdir(exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    test.to_csv(common.RESULTS / "test_predictions.csv", index=False, float_format="%.4f")
    print(f"\nsaved {report_path.relative_to(common.HERE)} and test_predictions.csv")

# %% Figures
if args.figures:
    import plots
    plots.validation_bars(pd.read_csv(common.RESULTS / "validation.csv"))
    plots.redness_curves(clips)
