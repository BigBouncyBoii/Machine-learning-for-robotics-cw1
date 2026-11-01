# Part 3: Open Exploration: can a camera tell a light grip from a firm grip?

Classify a short video of a hand gripping an object as a **light** or **firm** grip, from the video alone.

**Group final method:** James L's DINOv2 middle frames + MLP ensemble, run by `FinalMethod.py`.
**Test: 38/40 = 95.0 %** (guessing 50 %, one-sided binomial p = 7.5e-10), see `final_test_result.txt`.

Every command below is run from the repo root with the venv activated (`pip install -r <member>/requirements.txt`).

---

## Data and split

| | |
|---|---|
| Recordings | 160 clips = 4 people (A–D) × 2 objects (hard: phone, soft: plush toy) × 2 labels (light, firm) × 10 grips |
| Files | `data/labels.csv` (one row per clip), `data/videos_480p/` (480p copies; not submitted) |
| Split | `make_split.py` (seed 42) → `data/split.csv`: **120 train / 40 test**, fixed before exploring, never redrawn |
| Test set | 10 clips per person (5 light, 5 firm), 20 per object, 20 per label: 25 % of every member's recordings |
| Fingerprint | md5 of the sorted test-file list = `3696c80d`; every script checks it and stops if it differs |

Whole clips go to one set, so frames of one grip are never split between sets. The label is only in the file name,
never in the image. Feature extraction (frozen encoders, MediaPipe, optical flow) uses no labels, so it runs on all
160 clips; **everything that is fitted** (scaler, imputer, classifier, hyperparameters) sees only training clips.

**Validation** (every number in the tree): leave-one-person-out on the 120 train clips. Train on 3 people (90 clips),
validate on the 4th (30 clips), once per person; the number is the mean of the four. The 40 test clips are never used.
Guessing = 50 % (light and firm are equally many in every fold).

---

## Exploration tree

Draft: the main attempts and decisions, one line each. Initials: JL = James L, JF = James F, QL = Qian L, JE = Jeevan.
Pick 5–15 nodes for the report; every number is a validation accuracy from the command in the next table.

```
Question: can video tell a light grip from a firm grip?
├─ DINOv2 (frozen), mean of 16 frames (JL): pretrained image model, too few clips to train one
│  │  val 63.8 % MLP, 62.5 % LR → baseline, MLP kept
│  ├─ VideoMAE (JL): a video model, does seeing the motion help?
│  │     val 57.8 % MLP, 65.8 % LR → inconsistent (+4 clips with LR, mostly person C; −7 with MLP);
│  │     it takes all 16 frames together, so it can't be limited to part of the grip → dropped
│  └─ Fewer frames: 1 / 4 / 8 of the 16 (JL): does averaging all 16 dilute the grip?
│     │  val 70.0 / 66.7 / 64.5 % MLP → the single middle frame is best
│     └─ Grip phase (JL): start (frames 0–2) vs middle third (frames 5–10)
│        ├─ start: val 49.8 % → guessing: the scene does not give the label away
│        └─ middle: val 74.3 % MLP (75.0 % as the 5-MLP ensemble), 72.5 % LR → CHOSEN
├─ Finger colour, rest vs hold window, LR (JF): blanching makes squeezed fingertips paler; val 60.0 % → refined
│  └─ Per-finger colour curves over time (JF): nails + middle segments, 10 time bins; val 70.0 % (LR) → kept
│     └─ 10-MLP ensemble (JF): a non-linear model on the same curves; val 65.8 % (alpha 10) → no better than LR
├─ Hand motion, optical flow (QL): a firm grip may close faster or shake more
│  ├─ closing speed, early window: val 55.0 % → dropped
│  ├─ shaking, later window: val 60.8 % → best of the motion nodes
│  └─ speed + shaking: val 60.0 % → no gain
└─ Finger joint angles, MediaPipe (JE): a firm grip bends the fingers more; val 59.2 %; hand found in 42 % of frames → dropped
Final method on the test set: DINOv2 middle frames + MLP ensemble, 38/40 = 95.0 %.
```

**James L's nodes, notes for the viva:**
- Every James L node was run with **both** LR and MLP (`main.py` default), so each line gives both numbers;
  MLP is the main classifier, and LR is shown so no result is hidden (e.g. VideoMAE + LR).
- **Why 6 frames?** 16 frames are sampled evenly over the whole clip, so frame *i* is at *i*/15 of the clip:
  frames 5–10 are 33–67 %, the **middle third**, when the hand squeezes and holds. The window was picked in
  advance as "the hold phase", **not tuned**: only start (0–2) and middle (5–10) were tried.
- 8 vs 16 frames is a side check, not a step: 8 re-sampled from the video scores 62.0 % vs 63.8 % for 16 (MLP),
  so the number of frames is not what matters; *which* frames is.

### Node → code

| Node | Member | Command (from the repo root) | Validation |
|---|---|---|---|
| DINOv2 baseline, 16 frames | JL | `python "Part3/James L/main.py" --features dinov2` | 62.5 % LR, 63.8 % MLP |
| VideoMAE | JL | `python "Part3/James L/main.py" --features videomae` | 65.8 % LR, 57.8 % MLP |
| Number of frames | JL | `python "Part3/James L/main.py" --features dinov2_frames` | 1 frame 70.0 %, 4: 66.7 %, 8: 64.5 % (MLP) |
| Grip phase | JL | `python "Part3/James L/main.py" --features dinov2_phase --classifier lr mlp mlp_ensemble` | start 49.8 %, middle 74.3 % (MLP), 75.0 % (ensemble) |
| Rest vs hold colour | JF | `python "Part3/James F/main.py" --features hand_colour --classifier lr` | best 60.0 % (`hand_lab_change`) |
| Finger colour curves | JF | `python "Part3/James F/main.py" --features finger_patches --classifier lr mlp` | 70.0 % (`nail_mid_a_curve`, LR) |
| MLP ensemble | JF | `python "Part3/James F/main.py" --features finger_patches --classifier mlp_ensemble` | 65.8 % (chosen by log loss) |
| Speed / shaking / both | QL | `python "Part3/Qian L/main.py"` | 55.0 / 60.8 / 60.0 % |
| Joint angles | JE | `python Part3/Jeevan/main.py` | 59.2 % |
| **Final method, test** | JL | `python Part3/FinalMethod.py` | **test 38/40 = 95.0 %** |

Full tables (per held-out person, log loss, hard / soft, training accuracy): `<member>/results/validation.csv`.
Figures: `<member>/results/figures/`, made with `main.py --figures`.

---

## Members' branches

All four folders have the same layout:

| Path | What it is |
|---|---|
| `main.py` | Entry point: pick feature sets and classifiers on the command line; LOPO validation → `results/validation.csv` |
| `common.py` | `load_clips()` (labels + split + fingerprint check) and `validate()`: **the same in every folder** |
| `extractors/` | One file per idea (tree node): turns each clip into one feature vector |
| `classifiers/` | One file per classifier |
| `plots.py` | Figures for the report (`main.py --figures`) |
| `cache/` | Label-free features of all 160 clips, so re-runs take seconds and need no video |
| `results/` | `validation.csv`, `figures/`, and `test_report.txt` once that member's test is run |
| `archive/` | Earlier versions of the code. **Not submitted, not in the tree** |

### James L: frozen pretrained encoders (`James L/`)

160 clips are far too few to train a vision model, so a frozen pretrained encoder turns each clip into one vector,
and only a small classifier is trained on top.

```
clip ─► 16 evenly spaced frames ─► frozen encoder ─► pool to one vector ─► StandardScaler ─► LR or MLP ─► light / firm
```

- `extractors/dinov2.py`: DINOv2 ViT-S/14, each frame on its own (CLS vector, 384-d); baseline = mean of all 16 frames.
- `extractors/dinov2_frames.py`: which frames are averaged, written out (`[8]`, `[0, 4, 8, 12]`, …).
- `extractors/dinov2_phase.py`: start = frames `[0, 1, 2]`, middle = frames `[5, …, 10]` of 16.
- `extractors/videomae.py`: VideoMAE ViT-B, all 16 frames together (sees motion).
- Classifiers: `lr` (C = 0.1), `mlp` (64 units, mean of 5 seeds' accuracies), `mlp_ensemble` (the 5 MLPs as one model).

| Feature set | LR | MLP (mean of 5) | MLP ensemble |
|---|---|---|---|
| `dinov2_cls_mean` | 62.5 % | 63.8 % | 64.2 % |
| `dinov2_patch_mean` | 54.2 % | 62.8 % | 60.0 % |
| `videomae` | 65.8 % | 57.8 % | 58.3 % |
| `dinov2_1frame` / `4frames` / `8frames` | 68.3 / 69.2 / 63.3 % | 70.0 / 66.7 / 64.5 % | 69.2 / 67.5 / 65.0 % |
| `dinov2_resampled_8` (vs 16 = `cls_mean`) | 62.5 % | 62.0 % | 61.7 % |
| `dinov2_start` | 50.0 % | 49.8 % | 49.2 % |
| **`dinov2_middle`** | 72.5 % | **74.3 %** | **75.0 %** |

- **The signal is in the hold phase.** The start frames score exactly guessing; averaging in rest and release frames dilutes it.
- **Every model reaches 100 % training accuracy**: 384–768 features and 90 clips per fold, so they memorise.
- **Soft object much easier** (middle, ensemble: soft 90 %, hard 60 %): the toy visibly deforms.
- **Person B is the hardest fold** (43–60 % in every version).
- `FinalMethod.py` trains the **ensemble** (validation 75.0 %); 74.3 % is the mean of 5 single MLPs. Say which one you quote.

### James F: finger colour / blanching (`James F/`)

Squeezing hard pushes blood out of the fingertips and nail beds, so they go paler (CIELAB a* = redness drops).
MediaPipe only says *where* the fingers are; the decision is made only from colour, relative to the white wall.

- `extractors/hand_colour.py`: whole-hand and nail colour, rest window vs hold window (6 feature sets).
- `extractors/finger_patches.py`: 10 patches (nail + middle segment of each finger), colour change from rest in 10 time bins.
- Classifiers: `lr`, `mlp` (one MLP), `mlp_ensemble` (10 MLPs, alpha 0.3–10).

| Node | Best version | Validation |
|---|---|---|
| Rest vs hold, LR | `hand_lab_change` | 60.0 % (nail only 53–58 %) |
| Colour curves, LR | `nail_mid_a_curve` | **70.0 %** |
| Colour curves, one MLP | `nail_mid_lab_curves` | 69.2 % |
| Colour curves, 10-MLP ensemble | `nail_mid_lab_curves`, alpha 10 (lowest log loss 0.577) | 65.8 % |

- **Time course beats a single window**: one number per finger at the hold scores 47.5 %, the curve 70.0 %.
- The ensemble is not better than LR here; it was chosen for the lowest validation log loss, not the highest accuracy.
- Earlier versions held out a whole person and trained on group test clips; they are in `archive/`, not in the tree.

### Qian L: motion, speed and shaking (`Qian L/`)

A firm grip may close the fingers faster, and make the hand shake more under strain. Optical flow only: no landmarks.

- `extractors/motion.py`: Farneback optical flow between adjacent frames, skin pixels in a fixed hand/object region.
- `extractors/speed.py`: `WINDOW = (0.10, 0.45)`: local and inward finger speed (4 features).
- `extractors/shaking.py`: `WINDOW = (0.45, 0.85)`: hand velocity relative to the background, jitter around its trend (4 features).
- `extractors/both.py`: the 8 together. Classifier: LR, C ∈ {0.1, 1}; a clip with no usable motion gets the majority label.

| Node | C = 0.1 | C = 1 |
|---|---|---|
| Speed | 55.0 % | 55.0 % |
| **Shaking** | 59.2 % | **60.8 %** |
| Both | 58.3 % | 60.0 % |

- Low-dimensional features overfit less (training accuracy 59–71 %) but carry less signal.
- Unlike the appearance methods, the **hard object is easier** (shaking: hard 67 %, soft 55 %). Person C is the hardest fold.
- Qian L's own Windows run gave 56.7 / 62.5 / 59.2 % (`archive/results/`): the video decoder changes the optical flow
  slightly. Same conclusion; quote one machine's numbers.
- Some test clips were seen in Qian L's earlier project (`archive/results/historical_exposure.csv`).

### Jeevan: finger joint angles (`Jeevan/`)

A firm grip bends the fingers more and holds them more steadily.

- `extractors/joint_angles.py`: MediaPipe on every 2nd frame, 15 joint angles (3 joints × 5 fingers); per clip the mean,
  std and min of each angle + the hand-detection rate = 46 features. Classifier: LR, C = 1.

| Held out | A | B | C | D | **Mean** |
|---|---|---|---|---|---|
| Validation | 56.7 % | 63.3 % | 56.7 % | 60.0 % | **59.2 %** |

- MediaPipe finds the hand in a median of only **42 %** of frames, and in none for 7 clips: the object hides the fingers.
- Training accuracy 86 %. Hard and soft are about equal (58 / 60 %).

---

## Cross-method comparison

- **Every method beats guessing, but only modestly on unseen people (55–75 %).** The best node is DINOv2 on the middle
  frames (75.0 % as the ensemble); the best hand-crafted feature is the finger colour curve (70.0 %).
- **Timing matters.** DINOv2's start frames score exactly 50 %, and the colour method needs the rest frames as a
  reference: the information is in the squeeze / hold phase, not in the scene.
- **Appearance methods find the soft object easier** (DINOv2, colour): the toy deforms visibly. The motion method finds
  the hard object easier.
- **Overfitting vs signal:** 384–768-d encoder features reach 100 % training accuracy; low-dimensional physical features
  (motion 4–8-d, joint angles 46-d) overfit less but carry less signal.
- **The final test (95.0 %) is much higher than validation (75.0 %)** because the test people also appear in training,
  while validation always holds out an unseen person. Validation is the honest estimate for a new person.

---

## Files in `Part3/`

| File | What it is |
|---|---|
| `FinalMethod.py` | The group final method on the 40 test clips (uses James L's code and cache) |
| `final_test_result.txt` | Its output (run once) |
| `make_split.py` | Made `data/split.csv` once; never run again |
| `data/labels.csv`, `data/split.csv` | The clip list and the fixed split |
| `What_to_submit.md` | Checklist from the brief |
| `SPLIT_PROMPT.md` | The protocol prompt every member gave their AI tool (not submitted) |
