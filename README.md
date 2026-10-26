# Machine-learning-for-robotics-cw1

ELEC0144 coursework (assignment Draft 37).

## Layout

| Folder   | Contents |
|----------|----------|
| `Brief/` | Assignment PDF, data notes (`DATA_README.md`), self-assessment template |
| `Part1/` | Task 1: image processing on `Fruits.jpg` |
| `Part2/` | Task 2: regression network (`Task2.py`, `RegressionData.csv`, `InitialWeights.*`) |
| `Part3/` | Open Exploration (Section C) |

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

## Workflow

1. `git pull` before you start.
2. Work on a branch: `git checkout -b part1-yourname`.
3. Commit small changes, then `git push -u origin part1-yourname`.
4. Open a pull request on GitHub and merge it into `main`.

Try to stick to your own Part folder to avoid merge conflicts.
