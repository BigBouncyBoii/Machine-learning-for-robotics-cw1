# Qian L: motion (finger-closing speed and hand shaking)

**Idea:** a firm grip may close the fingers faster, and make the hand shake more under strain.
This uses optical flow only: no landmarks, no MediaPipe. The labels are the participant's intended grip, not measured force.

```
frame (long edge 320) ─► fixed hand/object region + YCrCb skin mask ─► Farneback optical flow between adjacent frames
clip  ─► speed features (10–45 % of clip) and/or shaking features (45–85 %) ─► StandardScaler ─► LogisticRegression
```

Every number here is **mean leave-one-person-out validation accuracy on the 120 train clips** of the group split
(`Part3/data/split.csv`, fingerprint `3696c80d`). The 40 test clips are never used for validation. Guessing = 50 %.

## Layout

| Path | What it is |
|---|---|
| `main.py` | Entry point: LOPO validation of every node and C → `results/validation.csv`; `--figures`; `--test` (once) |
| `common.py` | Paths, clips + split (fingerprint check), scaler pipeline, `validate()` (same as the other members) |
| `extractors/motion.py` | Shared: optical flow of every frame pair, cache, the window rules. Not a node itself |
| `extractors/speed.py` | Node: finger-closing speed. `WINDOW = (0.10, 0.45)`, 4 features |
| `extractors/shaking.py` | Node: hand shaking. `WINDOW = (0.45, 0.85)`, 4 features |
| `extractors/both.py` | Node: the 8 features together |
| `classifiers/lr.py` | Logistic regression, C ∈ {0.1, 1}; a clip with no usable motion gets the training majority label |
| `plots.py` | Validation overview, raw feature distributions (train clips only) |
| `cache/motion_series.json.gz` | Per-frame optical-flow samples of all 160 clips (no labels), so re-runs take seconds |
| `results/` | `validation.csv`, `figures/` (and `test_report.txt` once the test is run) |
| `archive/` | The original pipeline (`explore.py`, `features.py`, `verify.py`, `FinalMethod.py`, …) and its `results/`. **Not submitted.** |

## Tree nodes

`python main.py` runs all of them (seconds, from the cache). One node: `python main.py --features shaking`.

| Node | C = 0.1 | C = 1 | Hard / soft (best C) |
|---|---|---|---|
| Speed (early window) | 55.0 % | 55.0 % | 53 / 57 % |
| **Shaking (later window)** | 59.2 % | **60.8 %** | 67 / 55 % |
| Both | 58.3 % | 60.0 % | 63 / 57 % |

**Chosen** by Qian L's rule (highest mean accuracy; ties → fewer features, then smaller C): **shaking, C = 1**.
`python main.py --test` validates, applies that rule, and tests the chosen version once → `results/test_report.txt`.
**Test: not run yet.**

**These numbers differ from Qian L's original run** (`archive/results/summary.csv`: speed 56.7, shaking 62.5 with C = 0.1,
both 59.2 %). The code is the same: the original `archive/explore.py`, run on this machine with the same cache, gives exactly the
numbers above. The difference is the video decoder (Windows vs Linux), which changes the optical flow slightly. With ~30 clips
per held-out person, one or two clips move each number by 3 %, and the best C flips; the conclusion (shaking is the best of the
three, ~60 %, just above guessing) does not change. Report one machine's numbers consistently.

**Limitations:**
- The skin mask also picks up pale wall/table; the fixed windows can include approach, lift and release.
- **Some test clips were seen in Qian L's earlier project** under another protocol (`archive/results/historical_exposure.csv`:
  30 of the 40 test clips have artifacts there). Do not describe the test set as never seen for this branch.

## How to run

From anywhere, with the repo venv activated:

```bash
pip install -r "Part3/Qian L/requirements.txt"
python "Part3/Qian L/main.py" --figures     # validate all nodes + figures
python "Part3/Qian L/main.py" --test        # ONCE
```

If a clip is missing from `cache/`, it is decoded first (~3 s per clip).

AI use: original code GPT-6 (Codex), **version and setting: Qian L to fill in** (also in the code maps marked `<Qian L to fill in>`);
restructure by Claude Opus 5.5 (`claude-opus-5-5`), default setting.
