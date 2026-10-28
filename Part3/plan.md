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

- Put 25% aside (75/25, as the brief asks). **Decision:** part of every member's recordings, drawn at random with a fixed seed (`make_split.py`, seed 42). Each person × object × label group gives 2 or 3 of its 10 clips, arranged so that every person (10), object (20) and label (20) has exactly 25% in test. This gives 120 train / 40 test, with test balanced 20 light / 20 firm.
  - Rejected alternative: hold out all recordings of one member (25%). It tests generalisation to a new person, but costs a quarter of the training data. Leave-one-person-out validation (below) still measures that.
- Save the split to `Part3/data/split.csv` and commit it. That file is the record of which recordings are in the test set (the report must say this).
- Whole grips only: all frames of one clip go to the same set.
- Nothing is fit on the test set: no classifier weights, no thresholds, no feature normalisation / scaler.
- Validation on the 120 train clips: **leave-one-person-out** (train on 3 people, validate on the 4th, rotate). The tree reports the mean validation accuracy.

## 3. Pipeline / code

```
Part3/
├─ plan.md, report.md
├─ data/
│  ├─ labels.csv            file, person, object, label (one row per grip)
│  ├─ split.csv             + set: train / test (fixed, committed)
│  └─ videos_480p/A..D/     480p copies of the clips
├─ features/                cached encoder outputs, one .npz per encoder (git-ignored)
│  ├─ dinov2_small.npz
│  └─ videomae_base.npz
├─ encoders/                one file per frozen pretrained encoder
│  ├─ dinov2.py             image encoder: each frame on its own
│  └─ videomae.py           video encoder: the 16 frames together
├─ common.py                shared paths, frame sampling, feature versions (FEATURES), loading
├─ make_split.py            makes data/split.csv (run once)
├─ extract_features.py      python extract_features.py dinov2 | videomae
├─ explore.py               python explore.py <feature version>  -> one tree node's validation accuracy
├─ predict_clip.py          demo: classify one video from the raw file
└─ FinalMethod.py           required deliverable: final method on the test set
```

| File | Does |
|------|------|
| `make_split.py` | Random 25% of every member's clips → test (fixed seed), written to `data/split.csv` |
| `extract_features.py` | Video → 16 evenly spaced frames → frozen encoder → `features/<encoder>.npz`. Run once per encoder |
| `common.py` | `FEATURES`: every feature version the tree compares (encoder + how it is pooled to one vector per clip). A pooling-only node is one new line here |
| `explore.py` | Loads one feature version, leave-one-person-out validation of LR and MLP on the train clips, prints validation accuracy |
| `predict_clip.py` | Trains on cached train features, then classifies any video path from scratch (same sampling/encoder code) |
| `FinalMethod.py` | **Required deliverable.** Trains the chosen method on all train clips, runs it on the test set, prints accuracy vs 50% guessing |

Every file starts with a code map, and its last line names the AI model used (see `Part2/Task2.py`).

**Encoders** (all frozen, run on CPU):

| Encoder | Type | What it sees | Output per clip | Status |
|---------|------|--------------|-----------------|--------|
| DINOv2 ViT-S/14 (`facebook/dinov2-small`) | Image | Each of the 16 frames on its own (whole frame resized to 224×392) | 16 × 384 (CLS) and 16 × 384 (patch mean); pooled later | Extracted, baseline done |
| VideoMAE ViT-B (`MCG-NJU/videomae-base-finetuned-kinetics`) | Video | All 16 frames together, in order (centre 480×480 square → 224×224) | 768 (clip vector) and 8 × 768 (one per 2-frame time step) | Extracted, LR val 65.8% / MLP 57.8% |
| X-CLIP (`microsoft/xclip-base-patch32`) | Video + text | 8 frames together | Video vector, or zero-shot: compare with "a hand gripping firmly" vs "holding lightly" | Candidate |

**Why VideoMAE next.** DINOv2 encodes every frame on its own, and averaging the 16 frame vectors throws away their order, so the classifier cannot see the squeeze happen. VideoMAE was trained to fill in hidden parts of videos, so its tokens (2 frames × 16 × 16 pixels) describe motion as well as appearance. If VideoMAE beats DINOv2 on the same split, folds and classifiers, the motion of the squeeze carries signal. If not, a single frame already holds what can be seen. Risk: 768 numbers from 120 clips overfits more than DINOv2's 384, so the LR baseline (strong L2) matters as much as the MLP.

**Classifier:** start with logistic regression (the baseline), then a small MLP (1 hidden layer of 64, L2 penalty). With 120 training grips, the MLP may overfit. If it does, that is a valid tree result.

## 4. Exploration tree (plan, which will change with results)

At least 5 and at most about 15 nodes, at least one per member, each with **why / val result / decision / initials**. Every number is a mean leave-one-person-out validation accuracy on the 120 train clips (guessing = 50%).

```
Question: can video tell a light grip from a firm grip?
├─ DINOv2, image encoder (frozen)
│  ├─ Middle frame → LR (??): simplest baseline
│  ├─ Mean of 16 frames' CLS → LR (JL): does the whole clip help? val 62.5%
│  │  └─ Same features → MLP (JL): does a nonlinear head help? val 63.8% (train 100%: overfits) → kept as baseline
│  ├─ Patch mean instead of CLS (??): does local detail beat the global summary?
│  ├─ Squeeze frame − rest frame (??): firm grip = bigger visual change?
│  └─ Crop to the hand (MediaPipe Hands) (??): does background hurt?
├─ VideoMAE, video encoder (JL): does the motion of the squeeze matter? LR val 65.8%, MLP 57.8% (train 100%) → LR slightly above DINOv2, within noise
├─ X-CLIP zero-shot (??): does a video-language model already know grip force?
├─ One model per object (??): soft object val ~0.75 vs hard ~0.5, does the soft object carry the signal?
└─ Combine the best branches (all): do they make different mistakes? (weights/threshold chosen on validation only)
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
