# James L: frozen pretrained encoders + small classifier

**Question:** can a frozen, pretrained vision model tell a light grip from a firm grip with no fine-tuning?
We have only 160 clips, so the encoder stays frozen. Only a scaler + LR/MLP is trained on its output.

```
clip ─► 16 evenly spaced frames ─► frozen encoder ─► pool to one vector ─► StandardScaler ─► LR or MLP ─► light / firm
```

Every number here is **mean leave-one-person-out validation accuracy on the 120 train clips** of the group split
(`Part3/data/split.csv`, fingerprint `3696c80d`). The 40 test clips are never used for validation. Guessing = 50 %.

## Layout

| Path | What it is |
|---|---|
| `main.py` | Entry point. Pick feature sets + classifiers on the command line; LOPO validation → `results/validation.csv` |
| `common.py` | Paths, clips + split (fingerprint check), frame sampling, scaler pipeline, `validate()` |
| `extractors/dinov2.py` | DINOv2 ViT-S/14 image encoder (frozen), cache, baseline node (mean of all 16 frames) |
| `extractors/dinov2_frames.py` | Node: how many frames. `frames_used()` lists exactly which frames each version averages |
| `extractors/dinov2_phase.py` | Node: which phase of the grip. `frames_used()`: start = frames 0–2, middle = frames 5–10 of 16 |
| `extractors/videomae.py` | VideoMAE ViT-B video encoder (frozen), one 768-d vector per clip |
| `classifiers/lr.py` | Logistic regression, C = 0.1 |
| `classifiers/mlp.py` | One MLP, 64 ReLU units, alpha 0.01; validated with 5 seeds, **mean of their accuracies** |
| `classifiers/mlp_ensemble.py` | The same 5 MLPs as one model, **probabilities averaged** (what `../FinalMethod.py` trains) |
| `plots.py` | Validation bar chart (`main.py --figures`) |
| `predict_clip.py` | Demo: classify any video file from the raw video. Not needed for any reported number |
| `cache/` | Encoder outputs for all 160 clips (no labels). Git-ignored; built by the first run (~15 min CPU) |
| `results/validation.csv` | One row per (feature set, classifier): accuracy per held-out person, mean, log loss, hard / soft, train accuracy |
| `report.md` | Draft report text for this branch |
| `archive/` | The old scripts (`explore.py`, `frames_node.py`, `extract_features.py`, `encoders/`). **Not submitted.** |

## Tree nodes

| Node (file) | Feature set | LR | MLP (mean of 5) | MLP ensemble |
|---|---|---|---|---|
| Baseline: does a frozen image model see the grip? (`dinov2.py`) | `dinov2_cls_mean` | 62.5 % | 63.8 % | 64.2 % |
| | `dinov2_patch_mean` (local detail) | 54.2 % | 62.8 % | 60.0 % |
| Does the motion of the squeeze add signal? (`videomae.py`) | `videomae` | 65.8 % | 57.8 % | 58.3 % |
| How many frames? (`dinov2_frames.py`) | `dinov2_1frame` / `4frames` / `8frames` | 68.3 / 69.2 / 63.3 % | 70.0 / 66.7 / 64.5 % | 69.2 / 67.5 / 65.0 % |
| | `dinov2_resampled_8` (vs 16 = `dinov2_cls_mean`) | 62.5 % | 62.0 % | 61.7 % |
| Which phase of the grip? (`dinov2_phase.py`) | `dinov2_start` (control, frames 0–2) | 50.0 % | 49.8 % | 49.2 % |
| | **`dinov2_middle`** (frames 5–10) | 72.5 % | **74.3 %** | 75.0 % |

One command runs every row: `python main.py --classifier lr mlp mlp_ensemble` (about 20 s from the cache).
One node: `python main.py --features dinov2_phase` (an extractor name runs all its feature sets). Every model reaches 100 % training accuracy (`train_acc`).

**Group final method:** `dinov2_middle` + `mlp_ensemble`, run by `../FinalMethod.py`.
**Test (run once): 38/40 = 95.0 %** (p = 7.5e-10; hard 95 %, soft 95 %). See `../final_test_result.txt`.
The test is much higher than validation because the test people are also in training, while validation always uses an unseen person.

## How to run

From anywhere, with the repo venv activated:

```bash
pip install -r "Part3/James L/requirements.txt"
python "Part3/James L/main.py"                          # all nodes, LR + MLP (first run builds cache/)
python "Part3/James L/main.py" --features dinov2_middle --figures
python Part3/FinalMethod.py                              # group final method (ALREADY RUN)
python "Part3/James L/predict_clip.py"                  # demo
```

AI use: Claude Opus 5.5 (`claude-opus-5-5`), default setting.
