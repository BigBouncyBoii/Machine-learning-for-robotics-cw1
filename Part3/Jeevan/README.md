# Jeevan: finger joint angles (hand pose)

**Idea:** a firm grip bends the fingers more and holds them more steadily, which should show up in the hand pose.

```
every 2nd frame ─► MediaPipe hand landmarker (21 points) ─► 15 joint angles (3 joints × 5 fingers)
clip ─► mean, std, min of each angle (45) + hand-detection rate = 46 features ─► StandardScaler ─► LogisticRegression(C=1)
```

Every number here is **mean leave-one-person-out validation accuracy on the 120 train clips** of the group split
(`Part3/data/split.csv`, fingerprint `3696c80d`). The 40 test clips are never used for validation. Guessing = 50 %.

## Layout

| Path | What it is |
|---|---|
| `main.py` | Entry point: LOPO validation → `results/validation.csv`; `--figures`; `--test` (once) |
| `common.py` | Paths, clips + split (fingerprint check), scaler pipeline, `validate()` |
| `extractors/joint_angles.py` | MediaPipe landmarks → 15 joint angles per frame → 46 features per clip |
| `classifiers/lr.py` | Logistic regression, C = 1 |
| `plots.py` | Accuracy per held-out person, pooled validation confusion matrix |
| `cache/joint_angles.npz` | The 46 features of all 160 clips (no labels), so re-runs take seconds |
| `results/` | `validation.csv`, `figures/` (and `test_report.txt` once the test is run) |
| `archive/` | The original `FingerAnglesJeevan.py` and its figures. **Not submitted.** |

## What I explored

- **Joint angles:** for each finger, the knuckle (MCP/CMC), middle (PIP/MCP) and upper (DIP/IP) joint.
  Each angle is at landmark b, between b→a and b→c. 180° = straight.
- **Per-clip summary:** mean bend, how much it varies (std), the tightest bend (min), and the fraction of frames
  where a hand was found. Clips with no hand detected get zero angle features.

| Held out | A | B | C | D | **Mean** |
|---|---|---|---|---|---|
| Validation | 56.7 % | 63.3 % | 56.7 % | 60.0 % | **59.2 %** |

Training accuracy is 86 %. The hand was found in a median of only 42 % of frames, and in 0 frames for 7 of the
160 clips. The object hides much of the hand, which limits how good the angle features can be.
**Test:** not run yet (`--test`, once, when the method is final).

## How to run

From anywhere, with the repo venv activated:

```bash
pip install -r Part3/Jeevan/requirements.txt
python Part3/Jeevan/main.py --figures     # validation on the 120 train clips + figures
python Part3/Jeevan/main.py --test        # ONCE: final test on the 40 test clips
```

If `cache/joint_angles.npz` is missing, the first run downloads the MediaPipe hand model to `~/.cache/mediapipe/`
and processes all 160 clips from `Part3/data/videos_480p/` (~10 min on CPU).
On Linux/WSL, MediaPipe also needs `sudo apt install libgles2`.

AI use: restructure by Claude Opus 5.5 (`claude-opus-5-5`), default setting. Original code: **Jeevan to fill in**
(model, version, setting), also in the code maps marked `<Jeevan: ...>`.
