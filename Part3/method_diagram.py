# CODE MAP of method_diagram.py   (report figure for Method: input -> DINOv2 -> MLP -> output, 3 test clips)
#   Model      the final method exactly as FinalMethod.py: dinov2_middle + 5-MLP ensemble, fitted on the 120 train clips
#   Examples   light hard, firm soft, light soft; each the correct test clip with the median P(firm) of its group
#   Diagram    per row: 6 middle frames -> DINOv2 (ViT schematic on the clip's own frame) -> mean
#              -> MLP node diagram (x5) -> P(firm) and the decision
#   AI use     Claude Opus 5.5 (claude-opus-5-5), default setting
#
# Nothing is fitted on the test clips; they are only predicted.
# HOW TO RUN (venv activated, from any folder):  python Part3/method_diagram.py
# Output: Part3/figures/method_pipeline.png and .pdf

# %% Model
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "James L"))

import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.exceptions import ConvergenceWarning

import common
from classifiers import mlp_ensemble
from extractors import dinov2_phase

warnings.filterwarnings("ignore", category=ConvergenceWarning)

FEATURES = "dinov2_middle"
MIDDLE = dinov2_phase.frames_used(FEATURES)                  # [5, 6, 7, 8, 9, 10] of 16
INK, MUTED, BLOCK = "#0b0b0b", "#52514e", "#eef1f5"
OUT = Path(__file__).resolve().parent / "figures"

clips = common.load_clips()                                  # stops if the test-set fingerprint is wrong
X = dinov2_phase.load(clips, [FEATURES])[FEATURES]
train = (clips["set"] == "train").values
test = (clips["set"] == "test").values
model = common.make_pipeline(mlp_ensemble.make()).fit(X[train], clips["y"].values[train])
t = clips[test].copy()
t["p_firm"] = model.predict_proba(X[test])[:, 1]
t["correct"] = (t["p_firm"] >= 0.5).astype(int) == t["y"]


# %% Examples
def typical(label, obj):
    """Correct clip of this label and object whose P(firm) is closest to the median of those clips."""
    c = t[(t["label"] == label) & (t["object"] == obj) & t["correct"]]
    return c.loc[(c["p_firm"] - c["p_firm"].median()).abs().idxmin()]


examples = [typical("light", "hard"), typical("firm", "soft"), typical("light", "soft")]

# %% Diagram
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
ACCENT, EDGE = "#2a78d6", "#b9b8b3"
fig = plt.figure(figsize=(11.0, 5.6))
gs = GridSpec(3, 7, figure=fig, width_ratios=[1] * 6 + [13], hspace=0.12, wspace=0.05,
              top=0.92, bottom=0.02, left=0.01, right=0.99)


def panel(ax, x0, x1, title, stack=1):
    """Rounded panel with a title; stack > 1 draws offset copies behind it (an ensemble)."""
    for k in reversed(range(stack)):
        ax.add_patch(FancyBboxPatch((x0 + 0.7 * k, 0.9 + 0.35 * k), x1 - x0, 8.0,
                                    boxstyle="round,pad=0.1,rounding_size=0.6",
                                    fc=BLOCK if k == 0 else "#f7f8fa", ec=MUTED if k == 0 else EDGE,
                                    lw=1 if k == 0 else 0.7, zorder=1 - k * 0.1))
    ax.text((x0 + x1) / 2, 8.15, title, ha="center", va="center", fontsize=10, fontweight="bold",
            color=INK, zorder=5)


def arrow(ax, x0, x1, y=5, lw=1.2):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=10, lw=lw, color=INK,
                                 zorder=6))


def dinov2(ax, frame):
    """ViT schematic: the frame cut into 14-px patches -> tokens + [CLS] -> 12 transformer layers -> CLS 384-d."""
    panel(ax, 5.5, 41.5, "DINOv2 ViT-S/14 (frozen)")
    img = cv2.resize(frame, (224, 392))                     # the size the encoder is given: 16 x 28 patches
    th = ax.inset_axes([6.5, 1.4, 6.0, 6.2], transform=ax.transData)
    th.imshow(img)
    for g in range(0, 225, 14):
        th.axvline(g - 0.5, color="white", lw=0.25, alpha=0.8)
    for g in range(0, 393, 14):
        th.axhline(g - 0.5, color="white", lw=0.25, alpha=0.8)
    th.set_xlim(-0.5, 223.5); th.set_ylim(391.5, -0.5); th.axis("off")
    arrow(ax, 13.2, 16.3, lw=0.9)
    ys = [7.0, 6.1, 5.2, 4.3, 3.4]
    ax.scatter([18.0] * 5, ys, marker="s", s=28, color=[ACCENT] + ["#c9ced6"] * 4, edgecolor=MUTED,
               linewidth=0.5, zorder=5)
    ax.text(18.0, 2.6, "⋮", ha="center", va="center", fontsize=7, color=MUTED)
    ax.text(16.6, 7.0, "CLS", ha="right", va="center", fontsize=7.5, color=ACCENT)
    ax.text(18.0, 1.6, "448 patch tokens", ha="center", va="center", fontsize=7.5, color=MUTED)
    arrow(ax, 19.6, 22.6, lw=0.9)
    ax.add_patch(FancyBboxPatch((23.0, 2.6), 10.5, 4.6, boxstyle="round,pad=0.1,rounding_size=0.4",
                                fc="white", ec=MUTED, lw=0.8, zorder=4))
    ax.text(28.25, 4.9, "Transformer\n× 12 layers", ha="center", va="center", fontsize=8.5, color=INK, zorder=5)
    arrow(ax, 34.0, 37.0, lw=0.9)
    ax.scatter([38.8], [5.0], marker="s", s=60, color=ACCENT, edgecolor=MUTED, linewidth=0.5, zorder=5)
    ax.text(38.8, 3.4, "CLS\n384-d", ha="center", va="center", fontsize=7.5, color=INK)


def mlp(ax):
    """Node diagram of one MLP (384 -> 64 ReLU -> 1 sigmoid), drawn as a stack of 5 (the ensemble)."""
    panel(ax, 48.8, 72.4, "MLP × 5 (averaged)", stack=3)
    cols = {52.3: [6.9, 6.0, 5.1, 3.6, 2.7], 60.3: [6.5, 5.6, 4.7, 3.2], 68.0: [4.8]}
    gaps = {52.3: 4.35, 60.3: 3.95}                          # where the "⋮" goes (too many nodes to draw)
    xs = list(cols)
    for xa, xb in zip(xs, xs[1:]):
        for ya in cols[xa]:
            for yb in cols[xb]:
                ax.plot([xa, xb], [ya, yb], color=EDGE, lw=0.35, zorder=3)
    for x, ys in cols.items():
        colour = ACCENT if x == xs[-1] else "white"
        ax.scatter([x] * len(ys), ys, s=34, color=colour, edgecolor=INK, linewidth=0.6, zorder=4)
        if x in gaps:
            ax.text(x, gaps[x], "⋮", ha="center", va="center", fontsize=7, color=INK, zorder=4)
    for x, lab in zip(xs, ["384\ninput", "64\nReLU", "1\nsigmoid"]):
        ax.text(x, 1.55, lab, ha="center", va="center", fontsize=7.5, color=MUTED, zorder=5)


for r, ex in enumerate(examples):
    frames, _, _ = common.sample_frames(ex["video"])
    for c, i in enumerate(MIDDLE):
        a = fig.add_subplot(gs[r, c])
        a.imshow(frames[i]); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        if r == 0:
            a.set_title(f"frame\n{i}/15", fontsize=8.5, color=MUTED)
    ax = fig.add_subplot(gs[r, 6])
    ax.set_xlim(0, 100); ax.set_ylim(0, 10); ax.axis("off")
    arrow(ax, 0.5, 5.0)
    dinov2(ax, frames[MIDDLE[len(MIDDLE) // 2]])
    arrow(ax, 42.3, 48.2)
    ax.text(45.2, 6.3, "mean of\n6 frames", ha="center", va="center", fontsize=7, color=MUTED)
    mlp(ax)
    arrow(ax, 74.5, 77.8)
    pred = "firm" if ex["p_firm"] >= 0.5 else "light"
    ax.text(78.5, 6.9, f"P(firm) = {ex['p_firm']:.2f}", fontsize=15, fontweight="bold", color=INK, va="center")
    ax.text(78.5, 4.3, f"→ {pred}", fontsize=14, color=INK, va="center")
    ax.text(78.5, 1.9, f"true: {ex['label']},\n{ex['object']} object", fontsize=10.5, color=MUTED, va="center")

OUT.mkdir(exist_ok=True)
for ext in ["png", "pdf"]:
    fig.savefig(OUT / f"method_pipeline.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
print(f"saved {OUT / 'method_pipeline.png'} and .pdf")
print("examples:", [(e["file"], e["label"], round(float(e["p_firm"]), 2)) for e in examples])
