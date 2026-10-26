# Part 3 Plan: Open Exploration (Section C, 19 marks)

**Question:** from camera images alone, can a program tell a **light** grip from a **firm** grip?

**Approach:** a frozen pretrained vision model turns each grip video into a feature vector. A small classifier (an MLP, with logistic regression as the baseline) is then trained on those vectors to predict `light` / `firm`.

```
grip video ──► sample frames ──► pretrained encoder (frozen) ──► per-frame vectors
           ──► pool over time (mean / diff) ──► one vector per grip ──► MLP ──► light / firm
```

Why this approach: we only have about 100–200 grips, which is far too few to train a vision model from scratch. A pretrained encoder already knows what hands and objects look like, so we only train the small head. Features are extracted once and cached, which makes every later experiment take seconds on a CPU (the brief says no GPU is needed).

---

## 1. Recording (do this first, before any exploring)

| Item | Decision |
|------|----------|
| Objects | At least 2, and at least one must keep its shape when squeezed. **TODO pick:** e.g. sponge/stress ball (soft) + plastic bottle/mug (hard) |
| People | Every team member |
| Grips | For each person and object: **≥10 light + ≥10 firm**, in random order (use a pre-generated random sheet) |
| Label | Written on the sheet **before** each grip, never visible to the camera |
| Camera | **TODO pick:** one fixed phone camera, side-on to the hand. Same position, angle and lighting for every session (mark the spot with tape) |
| Clip | One short clip per grip, about 3 s: hand resting → squeeze → hold → release |
| Privacy | No faces, nobody outside the team |

**File naming:** `<initials>_<object>_<label>_<nn>.mp4`, e.g. `JL_sponge_firm_07.mp4`.
The filename holds the label, but the method only ever sees pixels, so this is fine (rule 3).

**Minimum dataset (4 people):** 4 × 2 objects × 20 grips = 160 grips.

Videos stay out of git (they are large and must not be submitted). Keep them in `Part3/data/videos/`, which is git-ignored, and share them via OneDrive/Drive. Keep them until marks are released, because the assessor may ask us to open one.
Commit `Part3/data/labels.csv` (`file, person, object, label`) so everyone works from the same list.

## 2. Test set (fix this BEFORE exploring, then don't touch it)

- Put ~25% aside. **Proposal:** hold out **all recordings of one member** (with 4 members this is 25%). It is a clean split and also tests whether the method works on a new person.
  - Alternative: 25% of each person × object × label, chosen with a fixed seed.
- Save the split to `Part3/data/split.csv` and commit it. That file is the record of which recordings are in the test set (the report must say this).
- Whole grips only: all frames of one clip go to the same set.
- Nothing is fit on the test set: no classifier weights, no thresholds, no feature normalisation / scaler.
- Validation on the remaining people: **leave-one-person-out** (train on 2, validate on 1, rotate). The tree reports the mean validation accuracy.

## 3. Pipeline / code

| File | Does |
|------|------|
| `make_sheet.py` | Generates the random light/firm order sheet for each person and object |
| `extract_features.py` | Video → sampled frames → encoder → saves `features/<model>.npz` (one vector per frame per grip) |
| `explore.py` | Loads cached features, runs one tree attempt, prints validation accuracy |
| `FinalMethod.py` | **Required deliverable.** Trains the chosen method on train+val, runs it on the test set, prints accuracy vs 50% guessing |

Every file starts with a code map, and its last line names the AI model used (see `Part2/Task2.py`).

**Encoder candidates** (all run on CPU):
- DINOv2 ViT-S/14: general image features, good on fine detail (finger/skin deformation)
- CLIP ViT-B/32: alternative image features
- VideoMAE / X-CLIP: a true video encoder, which may capture the squeeze motion

**Classifier:** start with logistic regression (the baseline), then a small MLP (e.g. 1 hidden layer of 64, dropout, early stopping on validation). With about 120 training grips, the MLP may overfit. If it does, that is a valid tree result.

Extra packages needed (add to `requirements.txt` when we start): `torch`, `transformers` (or `timm`), `opencv-python`, `scikit-learn`.

## 4. Exploration tree (plan, which will change with results)

At least 5 and at most about 15 nodes, at least one per member, each with **why / val result / decision / initials**.

```
Question: can video tell a light grip from a firm grip?
├─ Middle frame → DINOv2 → logistic regression (??): simplest baseline
├─ Mean of all frame vectors → LR (??): does the whole clip help?
│  └─ Same features → MLP (??): does a nonlinear head help?
├─ Squeeze-frame vector minus rest-frame vector (??): firm grip = bigger visual change?
├─ Crop to the hand (MediaPipe Hands) before encoding (??): does background hurt?
├─ Video encoder (VideoMAE) instead of image encoder (??): does motion matter?
└─ One model per object vs pooled (??): does the soft object carry all the signal?
Final method on the test set: ??%
```

Assign `??` to members once we split work. Later nodes should follow from earlier results ("step by step" is marked).

## 5. Extension (3 marks, pick one later)

- **B. Deeper test (probably easier):** ablations (mask the hand, mask the object, drop the frame-difference) and/or new conditions (a new object, other lighting).
- **A. Better reference:** put the object on a kitchen scale (display out of shot) and check whether the MLP's firm-score follows the reading.

Without an extension, this section caps at 16/19.

## 6. Report (≤4 pages, six headings)

Research question · Hypothesis (state what result would prove us wrong, e.g. "test accuracy not clearly above 50%") · Method (tree + one line per member + figure of input/output) · Validation (split, counts per person/object/label, measure, vs guessing) · Results · Discussion.
Every figure caption names the code file that produced it.

## 7. Order of work

1. Choose objects and camera setup, generate sheets, record everyone
2. Write `labels.csv`, fix and commit `split.csv`
3. `extract_features.py` and cache features
4. Baseline LR, then MLP, then the other tree nodes (split between members)
5. Choose final method, write `FinalMethod.py`, run on the test set **once we're done exploring**
6. Extension, report, viva prep (everyone must be able to explain their own node)
