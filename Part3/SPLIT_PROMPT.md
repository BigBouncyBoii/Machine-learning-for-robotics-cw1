# Standard train / validation / test protocol for Part 3

Paste everything inside the box below into your LLM at the start of a session (or into its project
instructions). It makes every group member's method use the **same test clips** and the **same validation
scheme**, so all numbers in the report can be compared directly.

---

```text
You are helping me with Part 3 of the ELEC0144 coursework (classifying video clips of a hand gripping an
object as "light" or "firm" grip). Every member of my group must use the SAME data split and validation
scheme so our results can be compared. Follow these rules exactly in any code you write for me.

DATA
- Repo layout: Part3/data/labels.csv lists all 160 clips (columns: file, person, object, label).
  4 people (A, B, C, D) x 2 objects (hard, soft) x 2 labels (light, firm) x 10 clips each.
- Videos: Part3/data/videos/ (original .MOV) and Part3/data/videos_480p/ (same names, .mp4).
- THE SPLIT IS FIXED in Part3/data/split.csv (columns: file, person, object, label, set).
  "set" is "train" (120 clips) or "test" (40 clips). The test set has exactly 10 clips per person
  (5 light, 5 firm), 20 per object and 20 per label.
- Always READ split.csv. Never redraw, reshuffle, re-seed or edit the split, and never run make_split.py again.
- Sanity check to print in every script: the test-set fingerprint
      hashlib.md5("\n".join(sorted(test_files)).encode()).hexdigest()[:8]
  where test_files are the "file" values with set == "test". It must be 3696c80d.
  If it is not, stop: the wrong split is being used.

TEST SET (40 clips) — touch it ONCE
- Never fit anything on test clips: no scaler, PCA, imputer, threshold, feature selection, classifier,
  hyperparameter choice, early stopping, or "looking at the plots to decide".
- Test clips are used only in one final script, after the method (features + model + hyperparameters)
  has been fully chosen on validation. Report that number once. If you change the method after seeing the
  test result, say so in the report.

VALIDATION — leave-one-person-out on the 120 TRAIN clips only
- 4 folds: for each person P in A, B, C, D: train on the train clips of the other 3 people,
  validate on the train clips of P (30 clips). Use sklearn LeaveOneGroupOut with groups = person,
  or an equivalent loop.
- Everything that learns from data (scaler, imputer, PCA, classifier, any feature normalisation) is fitted
  INSIDE each fold on that fold's training clips only (use an sklearn Pipeline).
- All clips/frames of one clip always stay together in the same set (never split frames of a clip).
- Report per fold accuracy, mean validation accuracy, and per-object (hard / soft) accuracy.
  Guessing = 50 %.
- Choose features, model and hyperparameters by MEAN VALIDATION accuracy (or validation log loss, but
  state which). Never by training accuracy, never by test accuracy.

FINAL TEST
- Refit the chosen method on ALL 120 train clips, predict the 40 test clips once.
- Report: test accuracy (n correct / 40), 95 % binomial CI, one-sided binomial p-value vs 50 %,
  accuracy per person and per object, confusion matrix, and the test-set fingerprint.
- If the model is random (e.g. an MLP), fix the seeds; if you average several seeds, say how many.

GOOD PRACTICE
- Cache slow feature extraction (per clip, no labels used) to a file so experiments re-run quickly.
  Feature extraction that does not use labels (e.g. a pretrained network, MediaPipe) may be run on all
  160 clips; anything fitted with labels may not see the test clips.
- Put a short header in each script: what it does, how to run it, and which AI model was used.
- If I ask for something that breaks these rules (e.g. a different split, tuning on test), point it out
  before writing the code.
```

---

## Why this protocol

- The **fixed test set** (`split.csv`, fingerprint `3696c80d`) means everyone's final number is measured
  on the same 40 clips, so methods can be ranked fairly.
- **Leave-one-person-out validation** checks that a method works on a person it was not trained on,
  instead of just recognising each person's hand or background.
- A whole-person hold-out (e.g. James F's nested `test_A.py`–`test_D.py`) is a stricter extra
  experiment and can still go in the report. But the headline test number should come from `split.csv`
  so it can be compared with everyone else's.
