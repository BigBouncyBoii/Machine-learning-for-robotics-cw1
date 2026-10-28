# Part 3 Report Notes: Open Exploration

Draft text for the report (Method section) and the viva walk-through. Numbers marked `??` are filled in once `explore.py` has run.

## Feature extraction (`extract_features.py`)

**Idea.** We have only 160 grips, far too few to train a vision model from scratch. Instead, a frozen pretrained image model (DINOv2 ViT-S/14) turns each video frame into a 384-number vector that describes what is in it. Only a small classifier is trained on top of these vectors. The features are computed once and cached, so every later attempt in the tree runs in seconds on a CPU.

**Steps, per grip clip:**

| Step | What we do | Why |
|------|------------|-----|
| Input | 480p copy of the clip (480×854, portrait, ~3 s, ~90 frames) | Smaller files to share; enough detail for a 224-px-wide model input |
| Sample | 16 evenly spaced frames over the whole clip | Covers rest → squeeze → hold → release, so later nodes can compare phases |
| Resize | Whole frame to 224×392, no crop, ImageNet normalisation | Keeps the full hand and object; both sides are multiples of DINOv2's 14-px patch |
| Encode | Frozen DINOv2-small, forward pass only | Pretrained on 142M images, good at fine detail such as skin and object deformation |
| Output | Per frame: CLS vector + mean of patch vectors (384 each) | Two summaries of the frame: global (CLS) and averaged local detail (patches) |
| Save | `features/dinov2_small.npz`: 160 clips × 16 frames × 384, with person/object/label | Frames are kept separate, so pooling over time is chosen later as a tree node |

**What the method never sees:** the label. It is read from `labels.csv` only to train and score the classifier and never enters the image (rule 3). All 16 frames of a clip stay together in one set (rule 2).

**Figure idea (input/output):** one clip → 16 thumbnails → 16 × 384 feature matrix → one pooled vector → light/firm. Caption: *produced by `extract_features.py`*.

## How it feeds the tree

The extractor is the shared front end. The tree nodes differ in what happens **after** it:

```
Question: can video tell a light grip from a firm grip?
└─ DINOv2 features, frozen (extract_features.py)
   ├─ Middle frame CLS → logistic regression (??): simplest baseline; val ??% → ??
   ├─ Mean over 16 frames → LR (??): does the whole clip help? val ??% → ??
   │  └─ Same features → small MLP (??): does a nonlinear head help? val ??% → ??
   ├─ Squeeze frame − rest frame (??): firm grip = bigger visual change? val ??% → ??
   └─ Patch mean instead of CLS (??): does local detail beat the global summary? val ??% → ??
Final method on the test set: ??% (guessing = 50%)
```

## Viva one-liners

- **Why frozen and not fine-tuned?** 160 clips would overfit a 22M-parameter network. Freezing it means we only fit a few hundred weights.
- **Why 16 frames?** Enough to see each phase of the grip, and few enough to run on a CPU. Pooling across frames is a tree decision, not fixed here.
- **Why no crop?** The first attempt keeps everything. "Crop to the hand" is a separate node that tests whether the background hurts.
- **Why both CLS and patch mean?** It costs nothing extra to save both, and comparing them is one tree node.
