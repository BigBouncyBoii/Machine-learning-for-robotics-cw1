# Jeevan: finger joint angles (hand pose)

**Idea:** a firm grip bends the fingers more and holds them more steadily, which should show up in the hand pose.

```
every 2nd frame ─► MediaPipe hand landmarker (21 points) ─► 15 joint angles (3 joints × 5 fingers)
clip ─► mean, std, min of each angle (45) + hand-detection rate = 46 features ─► StandardScaler ─► LogisticRegression(C=1)
```

## What I explored

- **Joint angles:**
  - For each finger: knuckle (MCP/CMC), middle (PIP/MCP) and upper (DIP/IP) joint.
  - Each angle is at landmark b, between b→a and b→c. 180° = straight.
- **Per-clip summary:**
  - Mean bend, how much it varies (std), and the tightest bend (min).
  - The fraction of frames where a hand was found is also a feature.
  - Clips with no hand detected get zero angle features.
- **Evaluation** (group protocol, `Part3/data/split.csv`, fingerprint `3696c80d`):
  - MediaPipe features are extracted for all 160 clips. That step has no labels and fits nothing.
  - The scaler + logistic regression only ever train on the **120 train clips**.
  - Validation is leave-one-person-out on those 120: train on 3 people (90 clips), validate on the 4th (30).
  - `--test` (run ONCE, when the method is final) trains on all 120 and scores the 40 test clips.

**Results** (leave-one-person-out on the 120 train clips, guessing = 50 %):

| Held out | A | B | C | D | **Mean** |
|---|---|---|---|---|---|
| Validation | 56.7 % | 63.3 % | 56.7 % | 60.0 % | **59.2 %** |

The hand was found in a median of only 42 % of frames, and in 0 frames for 7 of the 160 clips. The object hides much of the hand,
which limits how good the angle features can be. **Test:** not run yet (`--test`, once, when the method is final).

## Files

| File | What it does |
|---|---|
| `FingerAnglesJeevan.py` | Feature extraction (cached), LOPO validation on train clips, optional final test, figures |
| `angle_features_all_videos_image_mode.npz` | Feature cache, created on first run |
| `grip_success_rates.png` | Validation accuracy per held-out person |
| `grip_confusion_matrix.png` | Pooled validation confusion matrix |

## How to run

From the repo root, with the venv activated. The script works from any folder.

```bash
pip install -r Part3/Jeevan/requirements.txt
python Part3/Jeevan/FingerAnglesJeevan.py           # validation on the 120 train clips
python Part3/Jeevan/FingerAnglesJeevan.py --test    # ONCE: + final test on the 40 test clips
```

- The first run downloads the MediaPipe hand model to `~/.cache/mediapipe/` and processes all 160 clips from `Part3/data/videos_480p/`.
- Later runs reload the `.npz` cache. Delete it to recompute.

On Linux/WSL, MediaPipe also needs a system library: `sudo apt install libgles2` (otherwise you get `libGLESv2.so.2: cannot open shared object file`).
