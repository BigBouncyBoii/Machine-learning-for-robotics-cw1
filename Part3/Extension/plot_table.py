# CODE MAP of Extension/plot_table.py   (results/mask_ablation.csv drawn as a table for the report)
#   Load     results/mask_ablation.csv, written by mask_ablation.py (run that first)
#   Format   percentages, change vs the final method's test accuracy, p-value vs guessing
#   Output   figures/mask_ablation_table.png
#   AI use   Claude Opus 5.5 (claude-opus-5-5), default setting

# %% Load
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
t = pd.read_csv(HERE / "results" / "mask_ablation.csv")
INK, INK2, RULE = "#0b0b0b", "#52514e", "#b9b8b2"

# %% Format
final = float(t.loc[t["mode"] == "-", "test"].iloc[0])
NAMES = {"original (final method)": "Original (final method)", "hand_removed": "Hand removed",
         "hand_only": "Hand only"}
MODES = {"-": "Original frames", "retrained": "Retrained", "test only": "Final model, test only"}


def pct(v):
    return "-" if pd.isna(v) else f"{round(100 * v, 1):g}%"


def p_text(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


header = ["Input", "Training", "Val\n(LOPO)", "Test\n(n = 40)", "Change vs\nfinal", "Hard\nobject",
          "Soft\nobject", "p vs\nguessing"]
rows = []
for _, r in t.iterrows():
    change = "-" if r["mode"] == "-" else f"{round(100 * (r['test'] - final), 1):+g} pts"
    rows.append([NAMES[r["input"]], MODES[r["mode"]], pct(r["val"]), pct(r["test"]), change,
                 pct(r["hard"]), pct(r["soft"]), p_text(r["p_vs_guess"])])

# %% Output
fig, ax = plt.subplots(figsize=(8.6, 2.6))
ax.axis("off")
tab = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center",
               colWidths=[0.24, 0.2, 0.08, 0.09, 0.1, 0.08, 0.08, 0.09])
tab.auto_set_font_size(False)
tab.set_fontsize(8.5)
tab.scale(1, 1.75)
for (i, j), cell in tab.get_celld().items():
    cell.set_edgecolor(RULE)
    cell.visible_edges = "B"                                     # horizontal rules only
    cell.get_text().set_color(INK)
    if i == 0:
        cell.set_text_props(weight="bold")
        cell.set_height(cell.get_height() * 1.5)
    if j in (0, 1):
        cell.get_text().set_ha("left")
        cell.PAD = 0.03
    if i > 0 and j == 7 and t["p_vs_guess"].iloc[i - 1] >= 0.05:
        cell.get_text().set_weight("bold")                      # not better than guessing
ax.set_title("Mask ablation results (Extension/plot_table.py)", fontsize=10, color=INK, pad=4)
fig.text(0.5, 0.03,
         "Val = leave-one-person-out on the 120 train clips. Test = the 40 held-out clips (never trained on). "
         "\nTest only = the unchanged final model given masked clips. "
         "p = one-sided binomial test against 50% guessing; bold p >= 0.05 = not better than guessing.",
         ha="center", fontsize=7, color=INK2)
out = HERE / "figures" / "mask_ablation_table.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"figure -> {out}")
