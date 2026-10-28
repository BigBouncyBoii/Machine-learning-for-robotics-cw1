# Part 3 Report Notes: Open Exploration

Draft text for the report and the viva walk-through, written as we go. Numbers marked `??` are not measured yet.
All validation numbers come from `explore.py`. Train/test counts come from `make_split.py`.

## Data and split (report: Validation)

**Recordings.** 160 grip clips: 4 people (A–D) × 2 objects (hard: phone, soft: plush toy) × 2 labels (light, firm) × 10 grips. One fixed camera, side-on. Clips are 60–269 frames long (mean 104, about 2–9 s at 30 fps), so 16 evenly spaced frames are a different step size per clip but always span the whole grip. Labels come from the file name (`data/labels.csv`) and never appear in the image (rule 3).

**Test set (`make_split.py`, seed 42, saved in `data/split.csv`).** Fixed before exploring. 25% of every member's recordings go to test, as the brief asks. The split is **stratified**: the script fixes how many clips each group gives to test, then draws which ones at random.

| | Test (40 clips) | Train (120 clips) |
|---|---|---|
| Each person (A–D) | 10 (5 light, 5 firm) | 30 (15 light, 15 firm) |
| Each object (hard/soft) | 20 | 60 |
| Each label (light/firm) | 20 | 60 |

Each person × object × label group has 10 clips and gives 2 or 3 of them to test. Which groups give 3 is arranged so that every person, object and label has exactly 25% in test.

Why stratified rather than a plain random draw:
- **Guessing stays at exactly 50%.** With 20 light / 20 firm in test, always answering "firm" scores 50%. A plain random draw could give e.g. 26/14, where a constant answer already scores 65%.
- **No person or object dominates the score.** For example, a test set with mostly soft-object clips (the easier object) would flatter the method.

Whole clips are assigned to one set, so the frames of one grip are never split between train and test (rule 2). Nothing is fitted on the test clips: no classifier weights, no scaler, no threshold.

**Validation: leave-one-person-out, inside the 120 train clips (`explore.py`).**

| Round | Train on (90 clips) | Validate on (30 clips) |
|---|---|---|
| 1 | B, C, D | A |
| 2 | A, C, D | B |
| 3 | A, B, D | C |
| 4 | A, B, C | D |

Each round fits a fresh scaler and classifier on its 90 clips, then predicts the left-out person's 30 clips. The tree number is the **mean validation accuracy over the 4 rounds**. Each left-out person has 15 light / 15 firm clips, so guessing is 50% in every round.

- **Why leave out a whole person?** Clips of the same person share the hand, sleeve and squeezing style. If the validation clips came from people already in training, the model could score by recognising the person rather than the grip. Leaving a person out asks: does it work on someone it has never seen?
- **Why validation and not the test set?** We compare many versions and keep the best. Choosing that way is itself a form of fitting, so it is done on validation clips. The test set only measures the final choice.
- **The 4 round models are thrown away.** Validation chooses the method. The final model is then trained once on all 120 train clips (all 4 people) and run on the 40 test clips (`FinalMethod.py`).
- **The test is easier than validation.** The test clips are new grips by people who are in training, while validation always uses an unseen person. We expect test accuracy to be somewhat above validation accuracy.

Measure: accuracy (fraction of clips labelled correctly), compared with 50% guessing. Training accuracy is printed only as an overfitting check and never appears in the tree.

## Pipeline (report: Method)

**Idea.** 160 grips are far too few to train a vision model from scratch. A frozen pretrained encoder turns each clip into numbers that describe what it shows, and only a small classifier is trained on top. Encoder outputs are computed once and cached (`extract_features.py`), so each tree node runs in seconds on a CPU.

```
clip ──► 16 evenly spaced frames ──► frozen encoder ──► pool to one vector per clip ──► LR or MLP ──► light / firm
         (common.sample_frames)       (encoders/)        (common.FEATURES)              (explore.py)
```

**Frame sampling (both encoders).** The whole clip is decoded and 16 frames are taken at evenly spaced positions from first to last (`np.linspace`). This is not random, so the same clip always gives the same frames and the same features. The 16 frames cover rest → squeeze → hold → release.

**Encoders compared: 16 images vs 1 video.** Both get the same 16 frames. The difference is how they look at them:

```
DINOv2:   frame 1 ─► DINOv2 ─► vector 1  ┐
          frame 2 ─► DINOv2 ─► vector 2  ├─► average the 16 vectors ─► 1 clip vector (384)
          ...                            │   (order is lost: shuffled frames give the same vector)
          frame 16 ─► DINOv2 ─► vector 16 ┘

VideoMAE: frames 1–16 together ─► VideoMAE ─► 1 clip vector (768)
          (the model relates any part of any frame to any other, so it can see the grip change)
```

- **DINOv2 treats the clip as 16 separate photos.** Each frame is encoded on its own, and the frames are never overlaid or blended into one image. Only the 16 output vectors are averaged afterwards. It can see *how the grip looks* (finger position, how squashed the toy is), but not *how it changes*.
- **VideoMAE treats the 16 frames as one video.** Its tokens are 2 frames × 16 × 16 pixel blocks, and attention connects tokens across all frames. So it can, in principle, see the squeeze happen.


| | DINOv2 ViT-S/14 | VideoMAE ViT-B |
|---|---|---|
| Hugging Face model | `facebook/dinov2-small` | `MCG-NJU/videomae-base-finetuned-kinetics` |
| Type | **Image**: each frame encoded on its own (16 separate forward passes, run as one batch) | **Video**: the 16 frames encoded together, in order (one forward pass) |
| Tokens | Per frame: 16 × 28 = 448 patches of 14×14 px | Per clip: 8 × 14 × 14 = 1,568 blocks of 2 frames × 16×16 px |
| Pretraining | Self-supervised on 142M images | Fill in hidden parts of videos, then fine-tuned on 400 human actions (Kinetics) |
| Input | Whole frame resized to 224×392, no crop | Centre 480×480 square (the hand and object are always in the middle band) → 224×224 |
| Output kept | Per frame: CLS vector + mean of patch vectors (16 × 384 each) | Per clip: 768-number vector; also 8 × 768 (one per 2-frame time step) |
| Pooled to one vector | Mean of the 16 frames' CLS vectors (384) | The clip vector (768) |
| Feature file | `features/dinov2_small.npz` | `features/videomae_base.npz` |
| Encoder code | `encoders/dinov2.py` | `encoders/videomae.py` |

**Why VideoMAE.** If VideoMAE beats DINOv2 on the same split, folds and classifiers, the motion of the squeeze carries signal beyond what single frames show.

**Caveat:** the comparison changes two things at once, the encoder and the framing (whole frame vs centre square). The crop only removes empty wall and table, so it probably does not matter. A clean check would be DINOv2 on the same centre crop.

**Classifiers compared (both fitted after a StandardScaler, inside each fold):**

| | Logistic regression (baseline) | MLP |
|---|---|---|
| Model | Linear: weighted sum of the features → firm-probability | 1 hidden layer of 64 ReLU units → firm-probability |
| Regularisation | L2, C = 0.1 (strong) | L2, alpha = 0.01 |
| Weights fitted | 385 (DINOv2) / 769 (VideoMAE) | ~24,700 (DINOv2) / ~49,300 (VideoMAE) |
| Randomness | None | Depends on initial weights, so averaged over 5 seeds |

## My nodes (JL)

All numbers are mean leave-one-person-out validation accuracy on the 120 train clips (guessing = 50%).

| Node | Why | LR val | MLP val | Train acc | Decision |
|---|---|---|---|---|---|
| DINOv2, mean of 16 frames' CLS | Baseline: does a frozen image model see the grip at all? | 62.5% | 63.8% | 100% | Kept as baseline. Above guessing, but overfits |
| VideoMAE clip vector | Does the motion of the squeeze matter? | **65.8%** | 57.8% | 100% | LR best so far, but within noise of DINOv2 |

Per left-out person and object (MLP for DINOv2, LR for VideoMAE):

| Left out | DINOv2 all | hard | soft | VideoMAE all | hard | soft |
|---|---|---|---|---|---|---|
| A | 0.65 | 0.72 | 0.59 | 0.53 | 0.60 | 0.47 |
| B | 0.37 | 0.12 | 0.63 | 0.60 | 0.40 | 0.80 |
| C | 0.69 | 0.53 | 0.85 | 0.80 | 0.87 | 0.73 |
| D | 0.83 | 0.71 | 0.96 | 0.70 | 0.53 | 0.87 |

What the results show:
- **Both encoders are above guessing, but not by much.** 62–66% across 120 clips.
- **VideoMAE vs DINOv2 is not a clear win.** 65.8% vs 63.8% is 2–3 clips out of 120 (each clip ≈ 0.8%), and the rounds vary from 53% to 83%. We cannot claim that motion helps.
- **The MLP does not beat LR, and does worse on VideoMAE.** Every model reaches 100% training accuracy: with ~90 clips per round and 384–768 features, it memorises the training clips. The MLP has more weights, so it overfits more, especially on VideoMAE's 768 features. The strongly regularised LR copes better.
- **The soft object is easier than the hard one** (soft up to 0.96, hard often near or below 50%). The visible deformation of the soft toy may carry most of the signal → next node: one model per object.
- **Person B is below guessing with DINOv2** (0.37; hard object 0.12). Being that far below chance means the model systematically gets B backwards: B's light grips look like the others' firm grips. People squeeze differently. VideoMAE is less affected (0.60).

Next nodes that follow from these results:
- **PCA before the classifier.** Fewer features to reduce the overfitting.
- **VideoMAE late − early time step** (uses the saved 8 × 768). Does the change over the clip carry the signal?
- **One model per object.** Is the signal mostly in the soft object?

## Final method (`FinalMethod.py`)

**Current state:** DINOv2, mean of the 16 frames' CLS → StandardScaler → MLP (64 hidden units), 5 seeds with their firm-probabilities averaged. It is trained on all 120 train clips and run once on the 40 test clips. It prints test accuracy vs 50%, a one-sided binomial p-value (the chance of doing this well by guessing), accuracy per object and per person, and a confusion matrix. Test accuracy: ?? (not run yet).

**To decide:** on validation, VideoMAE + LR (65.8%) is now slightly ahead of DINOv2 + MLP (63.8%). If we choose it, `FinalMethod.py` must be updated to match before it is run.

`FinalMethod.py` is **not interactive**. It runs on the fixed test set and prints the result, which is what the brief asks for.

The interactive demo is a separate file, `predict_clip.py`: it trains on the cached train features, lists the test clips, then classifies any video path you give it (`--features videomae` to switch encoder). It re-runs the same sampling and encoder code from the raw video. For a train clip we checked that this matches the cached features exactly (difference 0.0).

**Figure idea (input/output):** one clip → 16 thumbnails → encoder → one vector → light/firm with firm-probability. Caption: *produced by `predict_clip.py`*.

## Viva one-liners

- **Why frozen and not fine-tuned?** 160 clips would badly overfit a 22M (DINOv2) or 86M (VideoMAE) parameter network. Freezing means we only fit the classifier.
- **Why 16 evenly spaced frames, not random?** They cover every phase of the grip and give the same features every run. VideoMAE also needs exactly 16.
- **Why extract test-clip features before training?** The encoder is frozen and processes each clip on its own with fixed ImageNet normalisation, so nothing is learned from the test clips. Only the scaler and classifier learn, and they are fitted on train clips only.
- **Why does training accuracy reach 100% while validation is ~64%?** Far more features than training clips per round, so the classifier can memorise them. That is why only validation numbers go in the tree.
- **Why is the test score expected to be higher than validation?** Test clips are new grips by people seen in training, while validation always uses an unseen person.
