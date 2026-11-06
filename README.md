# ELEC0144 Coursework 1: Image Processing and Neural Networks (Group 7)

Code for the ELEC0144 assignment (brief: `Brief/2026-Assignment-Draft37.pdf`). The work has three parts:

| Part | Task | What the code does | Main files |
|---|---|---|---|
| **Part 1** | Section A, Task 1: locate fruits | Finds the apple, capsicum and lemon in `Fruits.jpg` with from-scratch image processing; draws each bounding box and both centroids (box midpoint and moment method); works on darkened and mirrored images | `Task1.py`, `WhatIf_Task1.py` |
| **Part 2** | Section B, Task 2: 1-3-1 network | A 1-3-1 tanh network written from scratch in NumPy, trained on `RegressionData.csv` with SGD and then with ADAM | `Task2.py`, `WhatIf_Task2.py` |
| **Part 3** | Section C, Open Exploration | Can a camera tell a light grip from a firm grip? Four members' methods, the group final method and an ablation extension | `FinalMethod.py` |

Each code file starts with a **code map** that lists its `# %%` sections. The last line of the map names the AI model used for that file.

---

## Repository layout

```
.
├── README.md                  this file
├── requirements.txt           Python packages for all parts
├── Brief/                     assignment PDF, data notes, self-assessment template
├── Part1/                     Task 1: fruit detection
│   ├── Fruits.jpg                     input image
│   ├── Task1.py                       main demo (darkFactor = 0.8, mirror = True)
│   ├── Task1_functions.py             helpers: colour filter, scatter filter, largest region, box, centroids
│   ├── WhatIf_Task1.py                What-if: darkFactor 1.00 → 0.10 in steps of 0.05
│   ├── Task1_Apple_DiffStage_Image.py report figure: the apple mask after each filtering stage
│   ├── Task1_Whatif_DataAnalysis.m    MATLAB four-panel plot of the What-if CSV (report Figure 2)
│   ├── Task1_Whatif_Results/          What-if output: CSV, one image per dark factor, overview figure
│   └── Task1_Apple_Stage_Results/     apple-stage images
├── Part2/                     Task 2: 1-3-1 network, SGD and ADAM
│   ├── Task2.py                       normal run
│   ├── WhatIf_Task2.py                What-if: tanh slope (1 − a²) replaced by 1
│   ├── RegressionData.csv, InitialWeights.npz / .mat   given data and starting weights
│   ├── results_normal/                figures from Task2.py
│   └── results_forgotten_derivative/  figures from WhatIf_Task2.py
└── Part3/                     Open Exploration (full details: Part3/README.md)
    ├── FinalMethod.py                 group final method on the 40 test clips
    ├── final_test_result.txt          its output (38/40 = 95.0 %)
    ├── make_split.py                  made the fixed 120 train / 40 test split (run once)
    ├── data/labels.csv, data/split.csv  clip list and the fixed split (videos not included)
    ├── final_method_figure.py         report figure: final method input and output
    ├── method_diagram.py              report figure: frames → DINOv2 → MLP ensemble → P(firm)
    ├── figures/                       output of the two figure scripts
    ├── James L/                       frozen pretrained encoders (DINOv2, VideoMAE) + LR / MLP
    ├── James F/                       finger colour (blanching) with MediaPipe
    ├── Qian L/                        optical flow: finger-closing speed and hand shaking
    ├── Jeevan/                        MediaPipe finger joint angles
    └── Extension/                     Extension B: hand-masking ablation of the final method
```

---

## Setup

Tested with Python 3.14. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Parts 1 and 2 only need `numpy`, `pandas` and `matplotlib`. Part 3 also needs scikit-learn, SciPy, PyTorch, transformers, OpenCV and MediaPipe. On Linux/WSL, MediaPipe also needs `sudo apt install libgles2`.

Every script finds its files relative to its own location, so you can run every command below from the repository root.

---

## Part 1: locate fruits in an image

**Method.** Each fruit goes through three filters, all written from scratch in NumPy:
1. **Colour filter**: per-fruit rules on RGB ratios, for example `R > 1.5*G`. Ratios do not change when the whole image is darkened.
2. **Scatter filter**: a pixel is kept only if enough of its neighbourhood window is also in the mask.
3. **Largest region**: an 8-connected flood fill keeps only the biggest region.

From the final mask, the code takes the bounding box and its midpoint, the moment centroid (M10/M00, M01/M00), and the distance between the two centroids. There are no hard-coded coordinates and no library region functions.

```bash
python Part1/Task1.py                        # viva demo: darkFactor = 0.8, mirror = True
python Part1/WhatIf_Task1.py                 # What-if sweep, writes Part1/Task1_Whatif_Results/
python Part1/Task1_Apple_DiffStage_Image.py  # apple filtering stages, writes Part1/Task1_Apple_Stage_Results/
```

- `Task1.py`: `darkFactor` and `mirror` are at the top of the file. It shows the image with every box and both centroids, and prints both centroids of each fruit and the distance between them.
- `WhatIf_Task1.py`: uses the same settings as `Task1.py` (not retuned) with `mirror = False`, and runs 19 dark factors from 1.00 to 0.10. It saves an annotated image per factor, `fruit_detection_data_mirror_False.csv` and a 4×5 overview figure (report Figure 1). It prints a line for each factor.
- `Task1_Whatif_DataAnalysis.m` (MATLAB): reads the CSV and plots box width, box height, centroid distance and box area against the dark factor (report Figure 2).

More detail: `Part1/README.md`.

---

## Part 2: 1-3-1 network with SGD and ADAM

**Network.** 1 input → 3 tanh hidden nodes → 1 linear output, with E = ½(d − y)². Backpropagation is written by hand and follows the derivation in the report appendix. Settings are fixed by the brief:
- SGD: η = 0.1.
- ADAM: α = 0.005, β₁ = 0.9, β₂ = 0.999, ε = 1e-8.
- Both: 3000 epochs, an update after every sample, reshuffle every epoch, the same seed at the start of each run, start from `InitialWeights.npz`.

```bash
python Part2/Task2.py          # normal run
python Part2/WhatIf_Task2.py   # forgotten derivative
```

| Script | Prints | Figures |
|---|---|---|
| `Task2.py` | final training error for SGD and ADAM; the first epoch each goes below 0.001 | `results_normal/Task2_fit.png` (fit on `xtest` with the training data), `results_normal/Task2_learning_curves.png` (both learning curves) |
| `WhatIf_Task2.py` | the same, with `1 - a1**2` in the hidden delta replaced by `1` | `results_forgotten_derivative/` |

Expected output:

| Run | SGD final error | ADAM final error | First below 0.001 (SGD / ADAM) |
|---|---|---|---|
| `Task2.py` | 0.000399 | 0.000575 | epoch 200 / epoch 150 |
| `WhatIf_Task2.py` | 0.027419 | 0.026762 | not reached / not reached |

Each script trains for 3000 epochs twice, so it takes a little while.

---

## Part 3: Open Exploration: light vs firm grip from video

**Data.** 160 clips = 4 people (A–D) × 2 objects (hard: phone, soft: plush toy) × 2 labels (light, firm) × 10 grips.
- The split is fixed in `data/split.csv`: 120 train and 40 test clips. The test set has 10 per person (5 light, 5 firm), 20 per object and 20 per label. `make_split.py` made it once, before any exploring.
- Every script checks the test-set fingerprint `3696c80d` and stops if the split has changed.
- The videos are **not included** (the brief says not to submit them). Every method reads its features from the committed `cache/` folders, so the results can be reproduced without the videos.

**Validation.** Every number in the exploration tree is a leave-one-person-out result on the 120 training clips: train on 3 people, validate on the 4th, and take the mean over the four folds. The test clips are never used for training, scaling or choosing a method.

**Final method** (`FinalMethod.py`):
1. Take 16 evenly spaced frames per clip and keep the middle 6 (frames 5–10, the hold phase).
2. Run each frame through frozen DINOv2 ViT-S/14 to get a 384-d CLS vector, then average the 6 vectors.
3. Apply a StandardScaler, then 5 MLPs (64 ReLU units, seeds 0–4), all fitted on the 120 training clips.
4. Average the 5 values of P(firm). The clip is classified as firm if the average is at least 0.5.

Validation 75.0 %. **Test 38/40 = 95.0 %** (guessing 50 %, one-sided binomial p = 7.5e-10).

```bash
python Part3/FinalMethod.py              # final method on the test set (output as in final_test_result.txt)
python Part3/final_method_figure.py      # report figure: input and output -> Part3/figures/final_method_io.png
python Part3/method_diagram.py           # report figure: pipeline -> Part3/figures/method_pipeline.png
```

**Members' branches.** Each member's folder has the same layout: `main.py` (command-line entry point, runs leave-one-person-out validation and writes `results/validation.csv`), `common.py`, `extractors/` (one file per tree node), `classifiers/`, `plots.py`, `cache/` and `results/`.

| Member | Idea | Best validation | Command |
|---|---|---|---|
| James L (JL) | Frozen DINOv2 / VideoMAE features + LR / MLP; frame choice | 75.0 % (DINOv2, middle 6 frames, 5-MLP ensemble) → **final method** | `python "Part3/James L/main.py" --classifier lr mlp mlp_ensemble` |
| James F (JF) | Fingertip and nail colour (blanching) over time | 70.0 % (colour curves, LR) | `python "Part3/James F/main.py"` |
| Qian L (QL) | Optical flow: finger-closing speed, hand shaking | 60.8 % (shaking) | `python "Part3/Qian L/main.py"` |
| Jeevan (JG) | MediaPipe finger joint angles | 59.2 % | `python Part3/Jeevan/main.py` |

Add `--figures` to any `main.py` to redraw that member's figures. The full exploration tree, the command for each node and the per-person results are in **`Part3/README.md`**. Each member's own README explains that member's code.

**Extension B: masking ablation** (`Part3/Extension/`). A YCrCb skin threshold finds the hand. The ablation then removes the hand or keeps only the hand, and runs each version two ways: with the MLP ensemble retrained on the masked training clips, and with the unchanged final model on the masked test clips.

```bash
python Part3/Extension/mask_ablation.py   # -> Extension/results/mask_ablation.csv (masked features cached)
python Part3/Extension/plot_ablation.py   # -> Extension/figures/mask_ablation_results.png
python Part3/Extension/plot_table.py      # -> Extension/figures/mask_ablation_table.png
```

| Input | Retrained (test) | Final model unchanged (test) |
|---|---|---|
| Original | 95.0 % | 95.0 % |
| Hand removed | 87.5 % | 72.5 % |
| Hand only | 92.5 % | 57.5 % |

---

## Report figures → code

| Report figure | Produced by |
|---|---|
| Fig. 1, What-if darkening overview | `Part1/WhatIf_Task1.py` |
| Fig. 2, box width / height / distance / area vs dark factor | `Part1/Task1_Whatif_DataAnalysis.m` |
| Fig. 3, apple filtering stages | `Part1/Task1_Apple_DiffStage_Image.py` |
| Fig. 4 (a, b), normal fit and learning curves | `Part2/Task2.py` |
| Fig. 4 (c, d), forgotten-derivative curves and fit | `Part2/WhatIf_Task2.py` |
| Fig. 6, final method pipeline | `Part3/method_diagram.py` |
| Fig. 7a, P(firm) of all 40 test clips | `Part3/final_method_figure.py` |
| Fig. 7b, test confusion matrix | numbers printed by `Part3/FinalMethod.py` |
| Fig. 8 / Table 2, ablation masks and results | `Part3/Extension/mask_ablation.py`, `plot_table.py` |
