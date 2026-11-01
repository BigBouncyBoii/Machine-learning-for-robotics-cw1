# Qian L: motion (finger-closing speed and hand shaking)

**Idea:** a firm grip may close the fingers faster, and make the hand shake more under strain.
This uses optical flow only: no landmarks, no MediaPipe. The labels are the participant's intended grip, not measured force.

```
frame (long edge 320) ─► fixed hand/object ROI + YCrCb skin mask ─► Farneback optical flow between adjacent frames
clip  ─► speed features (10–45 % of clip) and/or shaking features (45–85 %) ─► StandardScaler ─► LogisticRegression
```

## What I explored

Three trials. Each compares C ∈ {0.1, 1}, using leave-one-person-out validation on the 120 train clips (`LeaveOneGroupOut`).

- **Speed proxy** (4 features): mean and 90th-percentile local speed, and inward speed toward the region centroid. Bulk motion is removed first.
- **Shaking proxy** (4 features): hand motion minus background motion. Features are detrended position RMS, velocity jitter RMS, acceleration RMS and reversal rate.
- **Both:** the 8 features together.

| Trial | C | Validation | Hard | Soft |
|---|---|---|---|---|
| Speed | 1.0 | 56.7 % | 56.7 % | 56.7 % |
| **Shaking** | 0.1 | **62.5 %** | 65.0 % | 60.0 % |
| Both | 0.1 | 59.2 % | 60.0 % | 58.3 % |

**Proposed method:** shaking + LR (C = 0.1). Selected by mean validation accuracy; ties prefer fewer features, then smaller C.
It is not frozen yet, and the 40-clip test has **not** been run (status 2026-11-01).

**Limitations:**
- The skin mask also picks up pale wall/table.
- The fixed windows can include approach, lift and release.
- Some current test clips were seen under an older protocol (`results/historical_exposure.csv`).

## Files

| File | What it does |
|---|---|
| `protocol.py` | Loads the shared split, checks fingerprint `3696c80d`, helper functions |
| `features.py` | Optical-flow speed and shaking features, with a per-clip cache (`cache_speed_shaking_v2/`, git-ignored) |
| `explore.py` | LOPO validation of the three trials → `results/summary.csv`, `tuning.csv`, `folds.csv`, `proposed_method.json` |
| `verify.py` | Recomputes out-of-fold predictions and checks the fingerprint and fallback rules |
| `plot_results.py` | Figures → `results/figures/` |
| `run_validation_and_plot.py` | Runs explore + verify + plot in one go |
| `FinalMethod.py` | `--freeze` the chosen method, then `--run-test` once |
| `DETAILS.md` | Full method description, failure policy and limitations |
| `PROTOCOL.md` | The shared group rules |

## How to run

From the repo root, with the venv activated. The scripts work from any folder.

```bash
pip install -r "Part3/Qian L/requirements.txt"

python "Part3/Qian L/run_validation_and_plot.py"    # validate + verify + plots (first run extracts 120 train clips)

# only when the method is final:
python "Part3/Qian L/FinalMethod.py" --freeze
python "Part3/Qian L/FinalMethod.py" --run-test       # ONCE; repeated tests are refused
```

`--freeze` refuses to run if the code has changed since `explore.py` last ran. Re-run validation after any edit.
