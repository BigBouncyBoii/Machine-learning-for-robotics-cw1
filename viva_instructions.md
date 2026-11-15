# Viva Day Instructions (Group 7)

## Before the viva starts
1. **Test everything on the laptop you'll actually use**, from a fresh terminal. There's no repair time, and a file that doesn't run caps every question for that task at 1/2.
2. **Open these files in the editor** (the brief says all files are open when the viva starts):
   - `Part1/Task1.py`, `Part1/Task1_functions.py`
   - `Part2/Task2.py`
   - the report PDF, including the derivation appendix
   - `Part3/README.md` (the exploration tree) and `Part3/final_test_result.txt`
3. Keep the raw videos on the laptop (`Part3/data/videos_480p/` or the originals). The assessor may ask you to open a recording.
4. Open a terminal at the repo root with the venv activated:

```bash
cd ~/Year3UCL/ELEC0144
source .venv/bin/activate
```

To run the submitted copy instead, `cd Group7_ELEC0144_CW1/Codebase` first. The code is identical, so the commands and output are the same.

---

## Timeline (at most 20 minutes)

| Time | Part | What happens |
|---|---|---|
| 0–6 min | **Task 1** | Run `Task1.py`, then one question each |
| by minute 7, about 6 min | **Task 2** | Run `Task2.py`, then one question each |
| by minute 13, about 7 min | **Open Exploration** | 2-minute walkthrough of the tree and final result, then one question each |

About 1¼ minutes per question. Students answer in an order drawn by lot. Give the specific answer first, then the reason.

---

## Task 1

```bash
python Part1/Task1.py
```

- It takes about 3 s, then opens the image. **Close the figure window to let the script finish.**
- **The assessor checks:** one box on each of the three fruits covering more than half of what is visible, nothing else boxed, and both centroids inside every box. The settings are already darkFactor 0.8 with mirroring.
- **On the image:** `o` = bounding-box midpoint, `x` = moment centroid. Box colours: apple red, capsicum cyan, lemon yellow.
- **Expected output:**

```
apple:    bounding-box midpoint = (664.00, 499.00); moment centroid = (664.21, 500.89); distance = 1.90 pixels
capsicum: bounding-box midpoint = (869.50, 420.50); moment centroid = (872.18, 425.76); distance = 5.90 pixels
lemon:    bounding-box midpoint = (378.00, 260.00); moment centroid = (382.72, 253.84); distance = 7.76 pixels
```

---

## Task 2

```bash
python Part2/Task2.py
```

- It takes **about 20 s with no output** while it trains SGD and then ADAM. Say "it's training" instead of waiting in silence.
- It then opens **two figures**: the fit on `xtest` and the learning curves. **Close both to let the script finish.**
- **The assessor checks:** the fit follows the data, both learning curves come down and flatten, and the final error is printed.
- **Expected output:**

```
SGD  final training error: 0.000399
ADAM final training error: 0.000575
SGD  first below 0.001: epoch 200
ADAM first below 0.001: epoch 150
```

---

## Open Exploration (Task 3)

- **Nothing needs to run.** The brief says this code is "read, not re-run".
- Start with the **2-minute team walkthrough** of the tree (`Part3/README.md` or the report's Figure 5) and the final result: **validation 75.0 %, test 38/40 = 95.0 %**.
- **Only if they ask to run it:**

```bash
python Part3/FinalMethod.py      # about 8 s, uses the cached features, prints 38/40 = 95.0 %
```

Backup if it fails: show `Part3/final_test_result.txt`.

---

## If something goes wrong
- **`ModuleNotFoundError`:** the venv isn't activated. Run `source .venv/bin/activate`.
- **No window appears:** check that the figure didn't open behind the editor. The printed numbers still appear in the terminal.
- **The script seems stuck:** a figure window is probably still open. Close it.
