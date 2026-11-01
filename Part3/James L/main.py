# CODE MAP of main.py   (James L, frozen pretrained encoders: one entry point for every tree node)
#   Settings      command line: which feature sets, which classifiers; --figures
#   Load          clips + group split (common.load_clips, fingerprint 3696c80d) and the chosen feature sets
#   Validation    every (feature set, classifier): leave-one-person-out on the 120 TRAIN clips only;
#                 classifiers with several seeds (mlp) are run once per seed and the results averaged
#   Save          rows merged into results/validation.csv (one row per node version, re-runs overwrite)
#   Figures       --figures: validation bars (plots.py)
#   AI use        Claude Opus 5.5 (claude-opus-5-5), default setting
#
# The final test of this branch is the group final method: ../FinalMethod.py (dinov2_middle + mlp_ensemble).
#
# HOW TO RUN (from anywhere, venv activated; the first run builds cache/ from data/videos_480p, ~15 min)
#   python "Part3/James L/main.py"                                   # validate every node (test clips not used)
#   python "Part3/James L/main.py" --features dinov2_middle          # one node, LR and MLP
#   python "Part3/James L/main.py" --features dinov2_cls_mean dinov2_resampled_8 --classifier mlp_ensemble
#   --features takes feature-set names (e.g. dinov2_middle) or extractor names (dinov2, videomae)

# %% Settings
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so the imports work from any folder

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

import common
from classifiers import CLASSIFIERS
from extractors import EXTRACTORS

SET_TO_EXTRACTOR = {s: e for e, m in EXTRACTORS.items() for s in m.FEATURE_SETS}

parser = argparse.ArgumentParser()
parser.add_argument("--features", nargs="+", help="feature sets or extractor names (default: all)")
parser.add_argument("--classifier", nargs="+", choices=list(CLASSIFIERS), default=["lr", "mlp"],
                    help="default: lr mlp (the two columns of the tree)")
parser.add_argument("--figures", action="store_true", help="save the validation bar chart")
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

# %% Load
clips = common.load_clips()
X_of = {}
for e in dict.fromkeys(SET_TO_EXTRACTOR[s] for s in feature_sets):   # each extractor loads once
    X_of.update(EXTRACTORS[e].load(clips, [s for s in feature_sets if SET_TO_EXTRACTOR[s] == e]))
tr = (clips["set"] == "train").values
print(f"test-set fingerprint {common.FINGERPRINT} ok; {tr.sum()} train clips, {(~tr).sum()} test clips")

# %% Validation
jobs = [(s, c, seed) for s in feature_sets for c in args.classifier for seed in CLASSIFIERS[c].SEEDS]
print(f"\nLeave-one-person-out validation on the 120 train clips (guessing = 50 %); the test clips are not used")
res = Parallel(n_jobs=-1)(delayed(common.validate)(X_of[s], clips, lambda c=c, seed=seed: CLASSIFIERS[c].make(seed))
                          for s, c, seed in jobs)
by_version = {}
for (s, c, _), r in zip(jobs, res):
    by_version.setdefault((s, c), []).append(r)
rows = []
for (s, c), runs in by_version.items():
    acc = [r["mean_acc"] for r in runs]
    rows.append({"extractor": SET_TO_EXTRACTOR[s], "features": s, "classifier": c,
                 **{k: np.mean([r[k] for r in runs]) for k in runs[0]},           # mean over seeds
                 "n_seeds": len(runs), "seed_min_acc": min(acc), "seed_max_acc": max(acc)})
val = pd.DataFrame(rows)
people = sorted(clips["person"].unique())
print(f"{'features':<20}{'classifier':<14}" + "".join(f"{'val ' + p:>7}" for p in people)
      + f"{'mean':>7}{'logloss':>9}{'hard':>7}{'soft':>7}{'train':>7}  seeds")
for _, r in val.iterrows():
    seeds = f"{100 * r['seed_min_acc']:.1f}-{100 * r['seed_max_acc']:.1f} %" if r["n_seeds"] > 1 else ""
    print(f"{r['features']:<20}{r['classifier']:<14}" + "".join(f"{100 * r[f'val_{p}']:>6.0f}%" for p in people)
          + f"{100 * r['mean_acc']:>6.1f}%{r['mean_logloss']:>9.3f}{100 * r['hard_acc']:>6.0f}%"
          + f"{100 * r['soft_acc']:>6.0f}%{100 * r['train_acc']:>6.0f}%  {seeds}")
best = val.loc[val["mean_acc"].idxmax()]
print(f"\nbest mean accuracy: {best['features']} + {best['classifier']} ({100 * best['mean_acc']:.1f} %)")

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

# %% Figures
if args.figures:
    import plots
    plots.validation_bars(val)
