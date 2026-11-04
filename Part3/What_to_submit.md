# What to submit: Part 3 (Open Exploration)

Source: `Brief/2026-Assignment-Draft37.pdf`, pages 6–11. Page numbers below refer to the brief.

The Open Exploration is worth **19 of 35 marks**: viva 4, report 12, "going further" extension 3.
Without an extension, the most this section can score is **16 of 19**.

---

## 1. One zip on Moodle (whole assignment, p. 10)

- `.zip` only, not `.rar`. **One member** submits it for the whole team.
- It holds the report PDF, all the code and the self-assessment form.
- **No videos.** The code must still reproduce the reported outputs from saved data: cached features, `labels.csv` and `split.csv`.

## 2. Report: the Part 3 section (p. 8, p. 10)

- **At most 4 pages** for Part 3, inside the 6-page report (the two What-ifs take up to 1 page each).
- Single spacing, **11 pt Arial**. Pages over the limit are not read.
- The cover page, references and derivation appendix don't count towards the limit.
- **Every figure caption names the code file that made it.** Figures need captions, axis labels and legends.

It must use exactly these **six headings**:

| Heading | What goes in it | Marked under |
|---|---|---|
| Research question | The question, and why it's worth answering | Part 3 |
| Hypothesis | The result we expect, **and which result would show we're wrong** | Part 3 |
| Method | The **exploration tree**, with **one line per member** under it saying what they did. The final method, with a **figure of its input and output** | Part 1 |
| Validation | How the recordings were split into train / validation / test; **how many recordings per person, object and label**; the measure (accuracy); how we compare with guessing | Part 2 |
| Results | The final method on the test set, and the extension if we do one | Part 2 |
| Discussion | What the results show and **don't** show, what failed and why (using tree results), and the next step | Part 3 |

### The tree (p. 7)
- **5 to about 15 nodes**, one line each, with **at least one node by every member**.
- Each node gives: what was tried and why, its **validation** result (never training accuracy), the decision (kept / dropped / chosen), and the member's initials.
- Later nodes should follow from earlier results ("step by step").
- Finish with the final method's **test** result, compared with 50% guessing.

### Report marking: 12 marks, three parts of 4 (p. 9)

| Part | 1 mark each | Scores 0 if |
|---|---|---|
| 1. Exploration (Method) | Each attempt explained · step by step (≥ 5 attempts) · final method explained with input/output figure · **code matches the report** | No tree, no method, or the code can't produce the results |
| 2. Final method (Validation, Results) | `FinalMethod` runs on the test set and **we say which recordings are in it** · data counts per person/object/label · compared with guessing · clear tables and figures | **The test set was used for training, frames of one grip were split between sets, or the label is visible in the input** |
| 3. Question and conclusions | Clear question, and a hypothesis that could be proven wrong · conclusions not overstated · failures explained from tree results · six headings, page limit, a next step that follows from the results | No research question, or no conclusions |

## 3. Code (p. 11)

- **`FinalMethod.py`** is a separate file that runs the final method on the test set and prints the result. It is **read, not re-run**.
- Also include **the code that produced every reported result**: every node in the tree, every figure.
- **Every file starts with a code map.** It uses the same section names as the `# %%` headings, and its **last line names the AI model, version and setting** (or "none").
- Use relative paths and documented commands. Code style and comments are not marked.

## 4. Going further: one extension, 3 marks (p. 7, p. 9)

Only one extension is marked:
- **A. A better reference.** Compare the method's firm score with a real measurement, such as a kitchen scale (keep the display out of view) or the EgoTactile dataset.
- **B. A deeper test.** Test at least two new conditions (a new object, camera, angle or lighting), or run ablations (mask the hand or the object, remove one part of the method).

How it's scored: 1 = done and result reported; 2 = also, test set not used for training and compared with the main result; 3 = also explains *why* the result differs. A negative result can still get 3.

## 5. Viva: Part 3 is 4 marks (p. 8–10)

- A **~2 minute** walk through the tree and final result (not marked on its own).
- Then **one question per student**, often about a node they did. Each answer is scored 2 / 1 / 0, and the average is multiplied by 2.
- **Keep the recordings until marks are released.** The assessor may ask us to open one.
- No AI help during the viva.

## 6. Other items for the whole team (p. 11)

- **Self-assessment form** (1 page, `SelfAssessment_Template.docx`). Each member writes 1–2 sentences on what they did. The team answers: what could improve in the solution, and in how we worked.
- **References** in IEEE style, including AI-generated code or text that was taken over unchanged.
- **Peer review on Moodle**: compulsory. If you skip it, your IPAC is capped at 70%.

---

## Where we are (checked 2026-11-01)

| Item | Status |
|---|---|
| Recordings: 4 people × 2 objects (hard + soft) × 10 light + 10 firm = 160 clips | Done (`data/labels.csv`) |
| Test set put aside before exploring (40 clips, 25%, whole clips) | Done: `data/split.csv`, fingerprint `3696c80d`, made by `make_split.py` |
| Nodes by every member | James L, James F, Qian L, Jeevan: see `README.md` (Members' branches) |
| Final method chosen on validation | James L's DINOv2 middle frames + MLP (val 74.3%) in `Part3/FinalMethod.py` |
| Final test run once, result vs 50% | Done: 38/40 = 95.0 % vs 50 % guessing (p = 7.5e-10), `final_test_result.txt` |
| Tree drawn (5–15 one-line nodes, with initials) | **Draft** in `README.md` (Exploration tree): pick the final 5–15 nodes |
| Input/output figure of the final method | **Still to do / confirm** |
| Data-count table (per person / object / label) | **Still to do** (10 per person/object/label; train 30 / test 10 per person) |
| Code maps with an AI-use line in every file | `Jeevan/FingerAnglesJeevan.py` still has a placeholder AI-use line |
| Extension (A or B) | Done: B, mask ablation (hand removed / hand only × retrained / test only), `Extension/mask_ablation.py`, `Extension/results/mask_ablation.csv`, figures in `Extension/figures/` |
| Videos left out of the zip | **Watch out:** `data/videos_480p/*.mp4` (160 clips) are *tracked in git* so the team can share them. Don't zip the repo as-is; leave `data/videos*/` and the raw `ELEC0144_Video*.zip` out |
| Saved data the code needs, in the zip | `Part3/*/features/` (James L's cached features, which `FinalMethod.py` needs) is git-ignored, so add it by hand. Also include `James F/cache/*.npz`, `data/labels.csv` and `data/split.csv` |

### Caveats to mention in the report
- **Test clips seen earlier:** some of the 40 test clips were used in older protocols, in James F's person-held-out runs and in Qian L's old split (`Qian L/results/historical_exposure.csv`). Say so, and don't call the test set "never seen".
- **Qian L's shaking result depends on the environment:** re-run on a different machine (different OpenCV build), it gave 60.8% instead of 62.5%. Quote the committed number and say where it came from.
- **James F chose by log loss:** the chosen input (nail + mid Lab curves, MLP) isn't the one with the highest validation accuracy (the a* curve with LR got 70.0%). The selection rule was validation log loss over the MLPs.

---

## Pre-submission checklist (checked 2026-11-04)

### Must fix
- [ ] **AI-use lines with placeholders.** Each one needs the model, version and setting:
  - `Jeevan/archive/FingerAnglesJeevan.py:9`: `<name your AI model, version and setting here>`
  - `Jeevan/main.py`, `Jeevan/plots.py`, `Jeevan/extractors/joint_angles.py`: `<Jeevan: model, version, setting>`
  - `Qian L/classifiers/lr.py`, `Qian L/plots.py`, `Qian L/extractors/{both,motion,shaking,speed}.py`: `<Qian L to fill in>`
- [ ] **No code map or AI-use line:** `James F/Task1_early/Function_A.py`, `Function_B.py`, `Main.py`, `Main_Repeated.py`. Add a header, or leave the folder out if nothing reported uses it.
- [ ] **Part 1 AI lines say only "GPT6 Astra".** Add a version and setting, or write "version/setting unavailable".
- [ ] **`SPLIT_PROMPT.md` and `tree.mmd` are deleted in the working tree but not committed.** Check this was intended. `tree.mmd` is the source of `tree.png`.

### Leave out of the zip
| What | Size | Why |
|---|---|---|
| `ELEC0144_video{A,B,C,D}.zip` | 3.2 GB | Raw videos aren't allowed |
| `data/videos/` and `data/videos_480p/` | 3.1 GB + 44 MB | No videos |
| `Extension/videos/*.mp4` | 4 MB | No videos |
| `.venv/`, all `__pycache__/` | | Environment files |
| `Qian L/archive/cache_speed_shaking_v2/` | | Old caches keyed by absolute paths |
| `What_to_submit.md` (this file) | | Internal checklist |
| `Assignment_Files_Draft37*`, `Brief/` | | The assignment brief itself |

### Make sure these are in the zip
- [ ] **`James L/cache/`**: git-ignored, but `FinalMethod.py` and `Extension/mask_ablation.py` need it. A fresh `git clone` will **not** have it; zip from a working copy that does, or copy it in by hand.
- [ ] `data/labels.csv`, `data/split.csv`
- [ ] `Extension/cache/*.npz`, `James F/cache/`, `Jeevan/cache/`, `Qian L/cache/` (tracked in git)
- [ ] `FinalMethod.py` and `final_test_result.txt`

### Already fine
- Every other Part 3 `.py` file has a code map ending in an AI-use line.
- No absolute paths in the code.
- The Extension features are cached, so `mask_ablation.py` rebuilds the table without the videos.
