# James F: finger colour (blanching)

**Idea:** squeezing hard pushes blood out of the fingertips and nail beds, so they go paler and less red.
In CIELAB, a* is the red–green axis, so a firm grip should lower a* from rest to hold.
MediaPipe only finds *where* the fingers are. The light/firm decision is made only from colour.

Every number here is **mean leave-one-person-out validation accuracy on the 120 train clips** of the group split
(`Part3/data/split.csv`, fingerprint `3696c80d`). The 40 test clips are never used for validation. Guessing = 50 %.

## Layout

| Path | What it is |
|---|---|
| `main.py` | Entry point. Pick feature sets + classifiers on the command line; validates them, or runs the one final test (`--test`) |
| `common.py` | Paths, clips + split (fingerprint check), skin mask, MediaPipe tracker, imputer + scaler pipeline, `validate()` |
| `extractors/hand_colour.py` | Node 1: whole-hand and nail colour, rest window vs hold window (6 feature sets) |
| `extractors/finger_patches.py` | Nodes 2–3: 10 finger patches (nail + middle segment), colour change over 10 time bins (4 feature sets) |
| `classifiers/lr.py` | Logistic regression, C = 1 (the baseline in every node) |
| `classifiers/mlp.py` | One MLP, 16 hidden units, alpha 1 |
| `classifiers/mlp_ensemble.py` | 10 MLPs (lbfgs) with different starting weights, probabilities averaged |
| `plots.py` | Validation bars and redness-over-time curves (`main.py --figures`) |
| `results/validation.csv` | One row per (feature set, classifier) version: accuracy per held-out person, mean, log loss, hard / soft |
| `results/figures/` | Figures for the report |
| `cache/` | Per-frame colours of all 160 clips (no labels), so re-runs take seconds |
| `models/hand_landmarker.task` | MediaPipe hand model (pretrained) |
| `archive/` | Earlier scripts that held out a whole person (they trained on group test clips). **Not submitted, not in the tree.** |
| `Task1_early/` | Early Part 1 work, not Part 3. **Not submitted.** |

## Tree nodes

| Node | Command | Result |
|---|---|---|
| 1. Hand / nail colour, rest vs hold, LR | `python main.py --features hand_colour --classifier lr` | best `hand_lab_change` 60.0 %; nail only 53–58 % |
| 2. Finger patches over time, LR vs one MLP | `python main.py --features finger_patches --classifier lr mlp` | `nail_mid_a_curve` + LR **70.0 %**; one MLP 47–69 % |
| 3. 10-MLP ensemble, alpha 0.3–10 | `python main.py --features finger_patches --classifier mlp_ensemble` | 44–70 %; lowest log loss `nail_mid_lab_curves`, alpha 10 (65.8 %, 0.577) |

`python main.py` with no options runs all of them (about 3 min on 8 CPU cores).
The full table is in `results/validation.csv`.

**Chosen version** (lowest validation log loss among the ensembles): `nail_mid_lab_curves` + `mlp_ensemble`, alpha 10.
`python main.py --test` runs it once on the 40 test clips → `results/test_report.txt`. It refuses to run again
unless `--again` is given. **Test: not run yet.** **Caveat:** these 40 clips were already predicted in the archived person-held-out analysis.

## How to run

From anywhere, with the repo venv activated:

```bash
pip install -r "Part3/James F/requirements.txt"
python "Part3/James F/main.py"                                          # validate every node
python "Part3/James F/main.py" --features nail_mid_a_curve --classifier lr --figures
python "Part3/James F/main.py" --test                                   # ONCE
```

If `cache/*.npz` is missing, the first run rebuilds it from `Part3/data/videos_480p/` (~15 min on CPU).
On Linux/WSL, MediaPipe also needs `sudo apt install libgles2`.

MLP numbers can differ by one clip between Windows and Linux (floating point in the lbfgs solver).
The numbers above are from Linux; the logistic regression numbers are identical on both.

AI use: Claude Opus 5.5 (`claude-opus-5-5`), GPT-6 Astra and GPT-6.1 Sol, all at the Medium setting.
