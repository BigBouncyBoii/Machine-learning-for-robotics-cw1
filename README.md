# Machine-learning-for-robotics-cw1

ELEC0144 coursework (assignment Draft 37).

## Layout

| Folder   | Contents |
|----------|----------|
| `Brief/` | Assignment PDF, data notes (`DATA_README.md`), self-assessment template |
| `Part1/` | Task 1: fruit detection, brightness experiments, supplied results and MATLAB analysis; see [Part1 instructions](Part1/README.md) |
| `Part2/` | Task 2: regression network (`Task2.py`, `RegressionData.csv`, `InitialWeights.*`) |
| `Part3/` | Open Exploration (Section C): `FinalMethod.py` (group final method, test 95.0 %, see `final_test_result.txt`), shared `data/`, `make_split.py`, `SPLIT_PROMPT.md`, `node_exploration.md` (summary of every method) |
| `Part3/James L/` | Frozen pretrained encoders (DINOv2, VideoMAE) + LR / MLP |
| `Part3/James F/` | Finger-colour (blanching) method; also James F's early Task 1 work |
| `Part3/Qian L/` | Optical-flow speed and shaking features |
| `Part3/Jeevan/` | MediaPipe finger joint angles |

## Setup (each person does this once)

```bash
git clone https://github.com/BigBouncyBoii/Machine-learning-for-robotics-cw1.git
cd Machine-learning-for-robotics-cw1
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`.venv/` is git-ignored, so everyone keeps their own. If you add a package, put it in `requirements.txt`.

## Running

Scripts load their data with relative paths, so run them from inside their folder:

```bash
cd Part2
python Task2.py
```

## Part 1: fruit detection

```bash
python Part1/Task1.py
python Part1/WhatIf_Task1.py
```

The main demo defaults to `darkFactor=0.8`, `mirror=True`.
Part 1 includes its input image, Python helpers, experiment results and a MATLAB
four-panel analysis script. See [Part1/README.md](Part1/README.md) for the actual
file layout, apple-stage image generation and MATLAB instructions.

## Part 3: the MLP classifier

The MLP is scikit-learn's `MLPClassifier`, defined in `Part3/James L/common.py` (`CLASSIFIERS["mlp"]`):

```python
make_pipeline(StandardScaler(), MLPClassifier(
    hidden_layer_sizes=(64,), alpha=1e-2, max_iter=500, random_state=seed))
```

- `StandardScaler` rescales each feature to zero mean, unit variance first.
- 1 hidden layer of 64 ReLU units, L2 penalty `alpha = 0.01`, Adam optimiser, up to 500 iterations.
- Input: one feature vector per clip from a frozen encoder (`extract_features.py`), not raw video.

**It is an ensemble of 5 MLPs, not one.** An MLP's result depends on its random initial weights, so `Part3/FinalMethod.py` (the group final method: `dinov2_middle` features) trains 5 copies with seeds 0–4 (`SEEDS["mlp"]`). All 5 have the same architecture and the same training clips; only the starting weights differ.

1. Train: each of the 5 MLPs is fitted on the 120 train clips.
2. Test: each MLP gives a firm-probability for every test clip, and the 5 probabilities are averaged.
3. A clip is predicted **firm** if the average is ≥ 0.5, otherwise **light**.

Averaging smooths out the seed-to-seed variation, so the reported accuracy doesn't depend on one lucky or unlucky initialisation. The script also prints each single seed's test accuracy to show how much they vary. The logistic regression baseline (`"lr"`) has no randomness, so it uses a single model (`SEEDS["lr"] = [0]`).

## Workflow

1. `git pull` before you start.
2. Work on a branch: `git checkout -b part1-yourname`.
3. Commit small changes, then `git push -u origin part1-yourname`.
4. Open a pull request on GitHub and merge it into `main`.

Try to stick to your own Part folder to avoid merge conflicts.
