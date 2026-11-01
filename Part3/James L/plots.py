# CODE MAP of plots.py   (figures for the report; called by main.py --figures)
#   Validation bars   validation_bars(): mean leave-one-person-out accuracy of every version in results/validation.csv
#   AI use            Claude Opus 5.5 (claude-opus-5-5), default setting

# %% Validation bars
import matplotlib.pyplot as plt
import numpy as np

import common

FIG_DIR = common.RESULTS / "figures"


def validation_bars(val):
    """One group of bars per feature set, one bar per classifier; dashed line = guessing."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sets = list(dict.fromkeys(val["features"]))
    versions = list(dict.fromkeys(val["classifier"]))
    w = 0.8 / len(versions)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for j, v in enumerate(versions):
        acc = [val.loc[(val["features"] == s) & (val["classifier"] == v), "mean_acc"] for s in sets]
        ax.bar(np.arange(len(sets)) + (j - (len(versions) - 1) / 2) * w,
               [100 * a.iloc[0] if len(a) else 0 for a in acc], width=w, label=v)
    ax.axhline(50, color="k", ls="--", lw=1, label="guessing")
    ax.set_xticks(np.arange(len(sets)), sets, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("mean validation accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("James L frozen encoders: leave-one-person-out validation on the 120 train clips (main.py)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "validation_bars.png", dpi=150)
    plt.close(fig)
    print(f"saved {(FIG_DIR / 'validation_bars.png').relative_to(common.HERE)}")
