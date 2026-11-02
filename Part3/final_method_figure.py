# CODE MAP of final_method_figure.py   (report figure: input and output of the final method)
#   Model      the final method exactly as FinalMethod.py: dinov2_middle + 5-MLP ensemble, fitted on the 120 train clips
#   Examples   3 test clips: light hard, firm soft, light soft; each the typical (median P(firm)) correct clip
#   Figure     (a) input = the 6 middle frames the method sees, output = averaged P(firm) and the decision
#              (b) output for all 40 test clips: one bar per clip = P(firm), sorted within each true label
#   AI use     Claude Opus 5.5 (claude-opus-5-5), default setting
#
# Nothing is fitted on the test clips; this only re-runs the final method to draw its input and output.
# HOW TO RUN (venv activated, from any folder):  python Part3/final_method_figure.py
# Output: Part3/figures/final_method_io.png and .pdf

# %% Model
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "James L"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.exceptions import ConvergenceWarning

import common
from classifiers import mlp_ensemble
from extractors import dinov2_phase

warnings.filterwarnings("ignore", category=ConvergenceWarning)

FEATURES = "dinov2_middle"
MIDDLE = dinov2_phase.frames_used(FEATURES)                  # [5, 6, 7, 8, 9, 10] of 16
LIGHT, FIRM = "#2a78d6", "#eb6834"                           # validated categorical pair (CVD-safe)
INK, MUTED = "#0b0b0b", "#52514e"
OUT = Path(__file__).resolve().parent / "figures"

clips = common.load_clips()                                  # stops if the test-set fingerprint is wrong
X = dinov2_phase.load(clips, [FEATURES])[FEATURES]
train = (clips["set"] == "train").values
test = (clips["set"] == "test").values
y = clips["y"].values

model = common.make_pipeline(mlp_ensemble.make()).fit(X[train], y[train])
t = clips[test].copy()
t["p_firm"] = model.predict_proba(X[test])[:, 1]
t["pred"] = (t["p_firm"] >= 0.5).astype(int)
t["correct"] = t["pred"] == t["y"]
print(f"test accuracy {t['correct'].sum()}/{len(t)} (must match final_test_result.txt: 38/40)")

# %% Examples
def typical(label, obj):
    """Correct clip of this label and object whose P(firm) is closest to the median of those clips."""
    c = t[(t["label"] == label) & (t["object"] == obj) & t["correct"]]
    return c.loc[(c["p_firm"] - c["p_firm"].median()).abs().idxmin()]


examples = [typical("light", "hard"), typical("firm", "soft"), typical("light", "soft")]

# %% Figure
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig = plt.figure(figsize=(7.0, 8.2))
gs = GridSpec(3, 8, figure=fig, top=0.93, bottom=0.40, width_ratios=[1] * 6 + [0.25, 1.9],
              hspace=0.12, wspace=0.05)
gs_b = GridSpec(1, 1, figure=fig, top=0.31, bottom=0.08, left=0.14, right=0.97)

for r, ex in enumerate(examples):
    frames, idx, n_total = common.sample_frames(ex["video"])
    for c, i in enumerate(MIDDLE):
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(frames[i])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if r == 0:
            ax.set_title(f"frame {i}/15", fontsize=7, color=MUTED)
    ax = fig.add_subplot(gs[r, 7])
    ax.axis("off")
    pred = "firm" if ex["pred"] else "light"
    ok = "correct" if ex["correct"] else "WRONG"
    ax.text(0, 0.78, f"true: {ex['label']}", fontsize=8, color=INK, transform=ax.transAxes)
    ax.text(0, 0.58, f"person {ex['person']}, {ex['object']} object", fontsize=7, color=MUTED,
            transform=ax.transAxes)
    ax.text(0, 0.34, f"P(firm) = {ex['p_firm']:.2f}", fontsize=9, fontweight="bold", color=INK,
            transform=ax.transAxes)
    ax.text(0, 0.14, f"→ {pred} ({ok})", fontsize=8, color=INK, transform=ax.transAxes)

fig.text(0.02, 0.975, "(a) Input: the 6 middle frames (of 16 evenly spaced) → DINOv2 → 5 MLPs → output",
         fontsize=9, fontweight="bold", color=INK, va="top")

ax = fig.add_subplot(gs_b[0, 0])
x0 = 0
for label, colour in [("light", LIGHT), ("firm", FIRM)]:
    g = t[t["label"] == label].sort_values("p_firm")             # one bar per clip, sorted within each label
    x = np.arange(x0, x0 + len(g))
    ok = g["correct"].values
    ax.bar(x[ok], g["p_firm"][ok], width=0.8, color=colour, zorder=3, label=f"true {label}, correct")
    if (~ok).any():
        ax.bar(x[~ok], g["p_firm"][~ok], width=0.8, color="white", edgecolor=colour, hatch="////",
               linewidth=1.2, zorder=3, label=f"true {label}, wrong")
        for xi, pi in zip(x[~ok], g["p_firm"][~ok]):
            ax.text(xi, pi + 0.03, "✗", ha="center", va="bottom", fontsize=9, color=INK)
    ax.text(x.mean(), -0.13, f"true {label} (n = {len(g)})", ha="center", va="top", fontsize=8, color=INK)
    x0 += len(g) + 2                                             # gap between the two groups
ax.axhline(0.5, color=INK, ls="--", lw=1, zorder=4)
ax.text(-0.5, 0.52, "threshold 0.5: above = predicted firm, below = predicted light", fontsize=7, color=MUTED, ha="left", va="bottom")
ax.set_xlim(-1, x0 - 1.5); ax.set_ylim(0, 1.05)
ax.set_xticks([])
ax.set_ylabel("P(firm)\n(average of 5 MLPs)")
ax.grid(axis="y", color="#e6e5e1", lw=0.6, zorder=0)
for sp in ["top", "right", "bottom"]:
    ax.spines[sp].set_visible(False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=3, frameon=False, fontsize=7)
ax.set_title(f"(b) Output for all 40 test clips, one bar per clip: {t['correct'].sum()}/40 correct", fontsize=9,
             fontweight="bold", color=INK, loc="left", pad=10)

OUT.mkdir(exist_ok=True)
for ext in ["png", "pdf"]:
    fig.savefig(OUT / f"final_method_io.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print(f"saved {OUT / 'final_method_io.png'} and .pdf")
print("examples:", [(e["file"], e["label"], round(e["p_firm"], 2)) for e in examples])
