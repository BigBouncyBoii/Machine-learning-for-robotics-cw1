# CODE MAP of Extension/plot_ablation.py   (figure of the mask ablation, to sit beside figures/mask_examples.png)
#   Load       results/mask_ablation.csv, written by mask_ablation.py (run that first)
#   Panel (a)  test accuracy per input: retrained on masked clips vs the unchanged final model (test only)
#   Panel (b)  retrained models only: test accuracy on the hard vs the soft object
#   Error bars 95% Wilson interval of a proportion (n = 40 test clips in (a), 20 per object in (b))
#   Output     figures/mask_ablation_results.png (same height as mask_examples.png)
#   AI use     Claude Opus 5.5 (claude-opus-5-5), default setting

# %% Load
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
t = pd.read_csv(HERE / "results" / "mask_ablation.csv")
N_TEST, N_OBJ = 40, 20

INPUTS = ["original (final method)", "hand_removed", "hand_only"]     # same order as the mask figure columns
NAMES = ["original", "hand removed", "hand only"]
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def get(inp, mode, col):
    r = t[(t["input"] == inp) & (t["mode"] == mode)]
    return float(r[col].iloc[0])


def ci(acc, n):
    """95% Wilson interval of a proportion -> (distance below, distance above) acc."""
    z, k = 1.96, round(acc * n)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return acc - (centre - half), (centre + half) - acc


def style(ax, title):
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, 0.25)], fontsize=8, color=INK2)
    ax.set_ylabel("test accuracy", fontsize=8, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(axis="both", length=0)
    ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)), label="guessing (50%)")


def bars(ax, x, vals, n, colour, label, hatch=None):
    err = np.array([ci(v, n) for v in vals]).T
    ax.bar(x, vals, width=0.36, color=colour, edgecolor="white", linewidth=2, hatch=hatch, label=label)
    ax.errorbar(x, vals, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5)
    for xi, v, e in zip(x, vals, err[1]):
        ax.text(xi, v + e + 0.015, f"{100 * v:g}%", ha="center", va="bottom", fontsize=7.5, color=INK)


plt.rcParams["hatch.color"] = "white"
plt.rcParams["hatch.linewidth"] = 1.0
fig, (a, b) = plt.subplots(2, 1, figsize=(5.7, 6.4))
x = np.arange(3)

# %% Panel (a)
retrained = [get(INPUTS[0], "-", "test")] + [get(i, "retrained", "test") for i in INPUTS[1:]]
test_only = [get(i, "test only", "test") for i in INPUTS[1:]]
bars(a, x - 0.19, retrained, N_TEST, BLUE, "retrained on masked train clips")
bars(a, x[1:] + 0.19, test_only, N_TEST, ORANGE, "final model unchanged (test only)", hatch="////")
a.set_xticks(x, NAMES, fontsize=8.5, color=INK)
style(a, "(a) Masked test clips, n = 40")
a.legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0, -0.36), ncol=2)

# %% Panel (b)
hard = [get(INPUTS[0], "-", "hard")] + [get(i, "retrained", "hard") for i in INPUTS[1:]]
soft = [get(INPUTS[0], "-", "soft")] + [get(i, "retrained", "soft") for i in INPUTS[1:]]
bars(b, x - 0.19, hard, N_OBJ, AQUA, "hard object (phone)")
bars(b, x + 0.19, soft, N_OBJ, YELLOW, "soft object (toy)", hatch="////")
b.set_xticks(x, NAMES, fontsize=8.5, color=INK)
style(b, "(b) Retrained models, by object, n = 20 each")
b.legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0, -0.36), ncol=2)

# %% Output
fig.suptitle("Mask ablation results (Extension/plot_ablation.py)", fontsize=10)
fig.text(0.5, 0.005, "Error bars: 95% Wilson interval. Test clips were never used for training.",
         ha="center", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.02, 1, 1), h_pad=2.5)
out = HERE / "figures" / "mask_ablation_results.png"
fig.savefig(out, dpi=200)
print(f"figure -> {out}")
