# CODE MAP of plots.py   (figures for the report; called by main.py --figures)
#   Overview        validation_overview(): mean accuracy per node, accuracy per held-out person, accuracy per object
#   Distributions   feature_distributions(): one speed and one shaking feature, light vs firm, TRAIN clips only
#   AI use          GPT-6 (Codex), version/setting <Qian L to fill in> (original figures);
#                   Claude Opus 5.5 (claude-opus-5-5), default setting (restructure)

# %% Overview
import matplotlib.pyplot as plt
import numpy as np

import common
from extractors import both

FIG_DIR = common.RESULTS / "figures"
COLORS = {"speed": "#2878B5", "shaking": "#C97723", "both": "#478B65"}


def validation_overview(val, oof, clips):
    """Best C of each node: (A) mean accuracy, (B) per held-out person, (C) per object. Dashed = guessing."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    best = val.sort_values(["mean_acc", "C"], ascending=[False, True]).drop_duplicates("features")
    best = best.set_index("features").loc[[s for s in COLORS if s in set(best["features"])]]
    tr = (clips["set"] == "train").values
    y, obj = clips["y"].values[tr], clips["object"].values[tr]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for i, (s, r) in enumerate(best.iterrows()):
        axes[0].bar(i, 100 * r["mean_acc"], color=COLORS[s], width=0.6)
        axes[0].text(i, 100 * r["mean_acc"] + 2, f"{100 * r['mean_acc']:.1f} %\n(C = {r['C']:g})", ha="center")
        axes[1].plot(list("ABCD"), [100 * r[f"val_{p}"] for p in "ABCD"], marker="o", color=COLORS[s], label=s)
        correct = (oof[(s, r["C"])] > 0.5) == y
        axes[2].bar(np.arange(2) + (i - 1) * 0.25, [100 * correct[obj == o].mean() for o in ("hard", "soft")],
                    width=0.23, color=COLORS[s], label=s)
    axes[0].set_xticks(range(len(best)), best.index)
    axes[0].set_title("A. Mean leave-one-person-out accuracy")
    axes[1].set_title("B. Held-out person (30 clips each)")
    axes[1].set_xlabel("person held out from training")
    axes[1].legend(frameon=False)
    axes[2].set_xticks(range(2), ["hard (n=60)", "soft (n=60)"])
    axes[2].set_title("C. Accuracy by object")
    for ax in axes:
        ax.set_ylim(0, 100)
        ax.set_ylabel("validation accuracy (%)")
        ax.axhline(50, color="#555555", ls="--", lw=1)
    fig.suptitle("Qian L motion nodes: 120 train clips, leave-one-person-out (main.py --figures)")
    fig.savefig(FIG_DIR / "validation_overview.png", dpi=180)
    plt.close(fig)
    print(f"saved {(FIG_DIR / 'validation_overview.png').relative_to(common.HERE)}")


# %% Distributions
def feature_distributions(clips):
    """Box plots of inward speed (early) and velocity jitter (late), per object and label, train clips only."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X = both.load(clips)["both"]
    tr = (clips["set"] == "train").values
    groups = [("hard", "light"), ("hard", "firm"), ("soft", "light"), ("soft", "firm")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, name, title in zip(axes, ["inward_speed_mean", "velocity_jitter_rms"],
                               ["Early inward-motion speed (speed.py)", "Later velocity jitter (shaking.py)"]):
        col = both.FEATURES.index(name)
        data = []
        for o, l in groups:
            v = X[tr & (clips["object"] == o).values & (clips["label"] == l).values, col]
            data.append(v[np.isfinite(v)])
        bp = ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True)
        for patch, (_, l) in zip(bp["boxes"], groups):
            patch.set_facecolor("#BFD8EC" if l == "light" else "#F0D0AC")
        ax.set_xticks(range(1, 5), [f"{o}\n{l}" for o, l in groups])
        ax.set_title(title)
        ax.set_ylabel("ROI diagonals / second")
    fig.suptitle("Raw feature distributions, 120 train clips (main.py --figures)")
    fig.savefig(FIG_DIR / "feature_distributions.png", dpi=180)
    plt.close(fig)
    print(f"saved {(FIG_DIR / 'feature_distributions.png').relative_to(common.HERE)}")
