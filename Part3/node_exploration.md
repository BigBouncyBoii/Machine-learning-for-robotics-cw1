# Part 3 Node Exploration: What Each Method Did

**Task:** classify a short video of a hand gripping an object as a **light** or **firm** grip, from the video alone.

**Group final method:** `Part3/FinalMethod.py` = James L's `dinov2_middle` + MLP. Validation 74.3 %. **Test 38/40 = 95.0 %** (95 % CI 83–99 %, p = 7.5e-10; run once, output in `final_test_result.txt`).

**Shared data (`Part3/data/`):** 160 clips = 4 people (A–D) × 2 objects (hard phone, soft plush toy) × 2 labels × 10 clips.
`make_split.py` (seed 42) fixed `data/split.csv`: 120 train / 40 test, stratified so that every person, object and label is 25 % test. Test fingerprint `3696c80d`. `SPLIT_PROMPT.md` sets the protocol everyone follows: leave-one-person-out (LOPO) validation on the 120 train clips, and the 40 test clips used once at the end.

Guessing = 50 % everywhere. Unless stated otherwise, every number below is **mean LOPO validation accuracy on the 120 train clips**. It is not a test score.

| Person | Folder | Idea | What it looks at | Validation (chosen method) | Test (40 clips) |
|---|---|---|---|---|---|
| James L | `James L/` | Frozen pretrained vision encoder + small classifier | Whole image (DINOv2) or whole video (VideoMAE) | **74.3 %** (DINOv2, middle 6 frames, MLP) | **95.0 %** (38/40), group final |
| James F | `James F/` | Finger blanching: a firm grip pushes blood out of the fingertips | Colour of nail and finger patches over time | 65.8 % (nail + mid Lab, 10-MLP); best LR 70.0 % | Not run |
| Qian L | `Qian L/` | Motion: finger-closing speed and hand shaking | Optical flow in a skin-coloured region | 62.5 % (shaking proxy, LR) | Not run |
| Jeevan | `Jeevan/` | Hand pose: a firm grip bends the fingers more | 15 finger joint angles from MediaPipe | 59.2 % (46 angle features, LR) | Not run |

---

## James L: frozen pretrained encoders (`James L/`)

**Idea.** 160 clips are far too few to train a vision model from scratch. A frozen pretrained encoder turns each clip into one feature vector, and only a small classifier is trained on top. Features are extracted once and cached (`features/*.npz`), so each node runs in seconds on a CPU.

```
clip ─► 16 evenly spaced frames ─► frozen encoder ─► pool to one vector ─► StandardScaler ─► LR or MLP ─► light / firm
```

**Files**
- `extract_features.py`: runs an encoder (`encoders/dinov2.py` or `encoders/videomae.py`) on all 160 clips. No labels are used, so test clips can be encoded too.
- `common.py`: frame sampling, the list of feature versions (`FEATURES`), and the two classifiers (`CLASSIFIERS`).
- `explore.py <feature>`: one run = one tree node. Runs LOPO validation for LR and MLP.
- `frames_node.py`: re-samples N frames from the video (not a subset of the 16) to test the frame count.
- `../FinalMethod.py` (in `Part3/`): the **group final method**. It trains on all 120 train clips and predicts the 40 test clips once.
- `predict_clip.py`: interactive demo that classifies any video path from the raw video.
- `report.md`: draft report text for this branch.

**Encoders**
- **DINOv2 ViT-S/14** (`facebook/dinov2-small`): an image model. Each of the 16 frames is encoded on its own (CLS + mean patch vector, 384-d), then the frames are averaged. Frame order is lost.
- **VideoMAE ViT-B** (`MCG-NJU/videomae-base-finetuned-kinetics`): a video model. All 16 frames are encoded together (centre 480×480 crop), giving one 768-d clip vector. It can, in principle, see the squeeze happen.

**Classifiers** (both fitted inside each fold)
- LR: `LogisticRegression(C=0.1)`, strong L2.
- MLP: 1 hidden layer of 64 ReLU units, `alpha=0.01`, with 5 seeds whose firm-probabilities are averaged.

**Nodes** (re-run from the cached features with `explore.py` / `frames_node.py`)

| Node (`common.FEATURES` name) | Question | LR | MLP |
|---|---|---|---|
| `dinov2_cls_mean`: mean CLS of 16 frames | Baseline: does a frozen image model see the grip at all? | 62.5 % | 63.8 % |
| `dinov2_patch_mean`: mean patch vector instead of CLS | Is local detail better than the global token? | 54.2 % | 62.8 % |
| `videomae`: clip vector | Does the motion of the squeeze add signal? | 65.8 % | 57.8 % |
| `dinov2_1frame`: middle frame only | Do we need many frames? | 68.3 % | 70.0 % |
| `dinov2_4frames`: frames 0, 4, 8, 12 | | 69.2 % | 66.7 % |
| `dinov2_8frames`: every 2nd of the 16 | | 63.3 % | 64.5 % |
| `frames_node.py 8 16`: 8 vs 16 frames re-sampled from the video (MLP) | | | 61.7 % vs 64.2 % |
| `dinov2_start`: first 3 frames (hand not gripping yet) | Control: where is the signal? | 50.0 % | 49.8 % |
| `dinov2_middle`: middle 6 frames (squeeze / hold) | | **72.5 %** | **74.3 %** |

**Findings**
- **The signal is in the hold phase.** The first frames, before the grip, score exactly guessing level. This is also a good sanity check: nothing in the scene, such as the person or the camera, leaks the label. Averaging in the rest and release frames dilutes the signal, which is why 1 middle frame beats all 16.
- **VideoMAE vs DINOv2 is not a clear win** (2–3 clips out of 120). We cannot claim that motion helps.
- **Every model reaches 100 % training accuracy.** With 384–768 features and about 90 clips per fold, the models memorise the training clips. The MLP overfits more than the strongly regularised LR, especially on VideoMAE.
- **The soft object is easier than the hard one**, because the deformation of the toy is visible.
- **Person B is the hardest fold**, and below 50 % with DINOv2. B's light grips look like the others' firm grips.
- **Group final method: `dinov2_middle` + MLP** (best validation score of all members' nodes). `Part3/FinalMethod.py` runs it by default.
- **Final test (run once): 38/40 = 95.0 %** (95 % CI 83–99 %, p = 7.5e-10). Hard 95 %, soft 95 %; A 90, B 90, C 100, D 100 %. Light 20/20 correct, firm 18/20.
- Test is far above validation because the test clips come from people who are also in training, while validation always uses an unseen person. Person B: 48 % unseen vs 90 % seen.

---

## James F: finger colour / blanching (`James F/`)

**Idea.** Squeezing hard pushes blood out of the fingertips and nail beds, so they go paler and less red (blanching). In CIELAB, a* is the red–green axis, so a firm grip should lower a* from rest to hold. MediaPipe only says *where* the fingers are. The light/firm decision is made only from colour.

**Pipeline** (`main.py` → `extractors/`, `classifiers/`, shared code in `common.py`)
1. MediaPipe hand landmarker (`models/hand_landmarker.task`) finds the 21 hand landmarks in each frame.
2. Ten discs per frame: the nail and the middle segment of each finger.
3. A YCrCb skin mask keeps only skin pixels in each disc. The patch colour is converted to CIELAB, relative to the white wall at the top of the frame. This cancels the camera's auto white balance.
4. Each clip is cut into 10 time bins. The first 2 bins (hand reaching in) are the rest reference, so the input is the colour change from rest per bin. Per-frame colours are cached in `cache/`.
5. Imputer + scaler + classifier, all fitted inside each fold. The classifiers are LR (C = 1), one MLP (16 units), and a 10-MLP ensemble.

**Tree nodes** (LOPO on the 120 train clips; full table in `results/validation.csv`)

| Node | Result |
|---|---|
| 1. Hand / nail colour, rest vs hold window, LR (`extractors/hand_colour.py`) | Best `hand_lab_change` 60.0 %; nail only 53–58 % |
| 2. Finger patches over time, LR vs one MLP (`extractors/finger_patches.py`) | `nail_mid_a_curve` + LR **70.0 %**; one MLP 47–69 % |
| 3. 10-MLP ensemble, alpha 0.3–10 | 44–70 %; lowest log loss `nail_mid_lab_curves`, alpha 10 (**65.8 %**, log loss 0.577) |

**Chosen:** `nail_mid_lab_curves` + `mlp_ensemble`, alpha 10. It was selected by validation **log loss**, not accuracy. Validation 65.8 %.
**Test:** not run (`python main.py --test` writes `results/test_report.txt`).

**Report caveat.** Earlier scripts, now in `archive/` and not part of the tree, held out one whole person at a time. Pooled over 160 clips, they scored 63.1 % (95 % CI 55–70 %). They used all 160 clips, so the 40 group test clips were already predicted. State this if James F's test number is reported.

---

## Qian L: motion, speed and shaking (`Qian L/`)

**Idea.** A firm grip may close the fingers faster and make the hand shake more under strain. Qian L measures these with optical flow, using no landmarks and no MediaPipe.

**Pipeline** (`features.py`)
- Every frame is resized to a long edge of 320 px. A fixed hand/object region (x 0.20–0.80, y 0.30–0.78) and a background region (top band) are set. A fixed YCrCb threshold selects skin-coloured pixels. This is a coarse heuristic, not a hand segmenter.
- Farneback optical flow is computed between adjacent frames.
- **Speed proxy** (early window, 10–45 % of the clip), 4 features: mean and 90th-percentile local speed, and inward speed toward the region centroid, after removing the median (bulk) motion.
- **Shaking proxy** (later window, 45–85 %), 4 features: the hand motion minus the background motion gives a velocity signal. Features are its detrended position RMS, velocity jitter RMS, acceleration RMS and direction-reversal rate.
- **Both**: the 8 features concatenated.
- Classifier: `StandardScaler` + `LogisticRegression` (C ∈ {0.1, 1}) in a pipeline, `LeaveOneGroupOut` by person on the 120 train clips. Selection is by mean validation accuracy, and ties prefer fewer features, then smaller C.

**Results** (`results/summary.csv`)

| Trial | Features | C | Val | Hard | Soft |
|---|---|---|---|---|---|
| Speed | 4 | 1.0 | 56.7 % | 56.7 % | 56.7 % |
| **Shaking** | 4 | 0.1 | **62.5 %** | 65.0 % | 60.0 % |
| Both | 8 | 0.1 | 59.2 % | 60.0 % | 58.3 % |

**Proposed method:** shaking + LR (C = 0.1). It is not frozen yet, and the test has not been run.

**Engineering:**
- `verify.py` recomputes all out-of-fold predictions and checks the fingerprint and the fallback rule.
- `FinalMethod.py --freeze` / `--run-test` refuses to run the test twice or with changed code.
- `historical_exposure.csv` records that some current test clips were seen in an older protocol.

---

## Jeevan: finger joint angles (`Jeevan/FingerAnglesJeevan.py`)

**Idea.** A firm grip bends the fingers more and holds them more steadily, which should be visible in the hand pose.

**Pipeline**
- MediaPipe hand landmarker (image mode, 1 hand) on every 2nd frame.
- **15 joint angles** per frame: 3 joints (knuckle, middle, upper) × 5 fingers. Each angle is at landmark b between the segments b→a and b→c (180° = straight).
- Per clip: the mean, std and min of each angle (45 features) + the hand-detection rate = **46 features**. Clips with no hand detected get zero angle features.
- `StandardScaler` + `LogisticRegression(C=1.0)`.
- Evaluation: leave-one-person-out on the 120 train clips of `split.csv`. MediaPipe features are extracted for all 160 clips (no labels), but the classifier never sees test clips. `--test` runs the one final test. The script saves `grip_success_rates.png` and `grip_confusion_matrix.png`.

**Status / caveats**
- **Validation: 59.2 %** (A 56.7, B 63.3, C 56.7, D 60.0 %). Above guessing, but the lowest of the four approaches.
- MediaPipe finds the hand in a median of only 42 % of frames, and in none for 7 clips. The gripped object hides the fingers, which limits pose features.
- B is *not* the hardest person here (63.3 %), unlike the appearance- and colour-based methods.
- Earlier versions trained and scored on all 160 clips, including the 40 test clips. This is now fixed to the group protocol, so its validation number is directly comparable with the others.
- The `AI use` line in the header is still a placeholder.

---

## Results split by held-out person and object

Each person's **chosen** method, all on the same protocol: leave-one-person-out on the 120 train clips of `split.csv`. Each fold validates on 30 clips (15 hard, 15 soft), and guessing = 50 %. These are validation numbers. No test clips are used.

| Method (person) | Val A | Val B | Val C | Val D | **Mean** | Hard | Soft |
|---|---|---|---|---|---|---|---|
| DINOv2 middle frames + MLP (James L) | 86.0 % | 48.0 % | 69.0 % | 95.0 % | **74.3 %** | 59.5 % | 89.0 % |
| Nail + mid Lab curves + 10-MLP (James F) | 76.7 % | 60.0 % | 56.7 % | 70.0 % | **65.8 %** | 60.0 % | 71.7 % |
| Shaking proxy + LR (Qian L) | 70.0 % | 60.0 % | 53.3 % | 66.7 % | **62.5 %** | 65.0 % | 60.0 % |
| Finger joint angles + LR (Jeevan) | 56.7 % | 63.3 % | 56.7 % | 60.0 % | **59.2 %** | 58.3 % | 60.0 % |

Sources:
- James L: `James L/explore.py dinov2_middle`. MLP accuracies are averaged over 5 seeds, so they are not whole clip counts.
- James F: `James F/results/validation.csv`.
- Qian L: `Qian L/results/folds.csv` and `summary.csv`.
- Jeevan: `Jeevan/FingerAnglesJeevan.py`.

What the split shows:
- **James L's lead comes from the soft object and people A and D.** On the hard object (59.5 %) it is no better than the others. On person B it is below guessing (48 %; hard object only 11 %).
- **The hand-crafted methods are flatter across people.** Qian L and Jeevan have no fold below 53 %, but no fold above 70 % either.
- **No method is reliable on the hard object** (58–65 %). A rigid phone barely deforms, so most of the signal has to come from the hand itself.
- **Person C is hard for everyone** (53–69 %). Person B splits the methods: worst for James L, best for Jeevan.

## Cross-method comparison

- **Every method beats guessing, but only modestly (≈ 60–74 %).** The best single node is James L's DINOv2 on the middle frames (74.3 %). The best hand-crafted feature is James F's nail + mid a* curve (70.0 %).
- **Common pattern: the soft object is easier than the hard one** (James L, James F). Deformation and contact are more visible on the plush toy.
- **Person B is consistently the hardest held-out person** (James L, James F). B's grip style differs from the others'.
- **Timing matters.** James L's start-frames control scores exactly 50 %, and James F uses the first bins as a rest reference. Both show that the information is in the squeeze/hold phase.
- **Overfitting:** high-dimensional encoder features (384–768-d) reach 100 % training accuracy. Low-dimensional, physically motivated features (Qian L: 4-d, 62.5 %; Jeevan: 46-d, 59.2 %) overfit less but carry less signal.
- **Group headline: 95.0 % test (38/40) vs 74.3 % validation.** The test uses people seen in training; validation uses a new person. Report both.
- **Still open:** James F, Qian L and Jeevan may each run their own final test **once**. James F's and Qian L's test clips were seen in earlier work, so disclose that if you report their test numbers.
