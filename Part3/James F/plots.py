# CODE MAP of plots.py   (figures for the report; called by main.py --figures)
#   Validation bars   validation_bars(): mean leave-one-person-out accuracy of every version in results/validation.csv
#   Redness curves    redness_curves(): mean hand / nail a* over the clip, light vs firm, TRAIN clips only
#   AI use            Claude Opus 5.5 (claude-opus-5-5), GPT-6 Astra, GPT-6.1 Sol; all Medium setting

# %% Validation bars
import matplotlib.pyplot as plt
import numpy as np

import common
from extractors import hand_colour

FIG_DIR = common.RESULTS / "figures"


def validation_bars(val):
    """One group of bars per feature set, one bar per classifier version; dashed line = guessing."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sets = list(dict.fromkeys(val["features"]))
    versions = list(dict.fromkeys(val["classifier"]))
    w = 0.8 / len(versions)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for j, v in enumerate(versions):
        acc = [val.loc[(val["features"] == s) & (val["classifier"] == v), "mean_acc"] for s in sets]
        ax.bar(np.arange(len(sets)) + (j - (len(versions) - 1) / 2) * w,
               [100 * a.iloc[0] if len(a) else 0 for a in acc], width=w, label=v)
    ax.axhline(50, color="k", ls="--", lw=1, label="guessing")
    ax.set_xticks(np.arange(len(sets)), sets, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("mean validation accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("James F colour branch: leave-one-person-out validation on the 120 train clips (main.py)")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "validation_bars.png", dpi=150)
    plt.close(fig)
    print(f"saved {(FIG_DIR / 'validation_bars.png').relative_to(common.HERE)}")


# %% Redness curves
def redness_curves(clips):
    """2x2 panels (hand / nail x hard / soft): mean a* over normalised time, light vs firm, train clips only."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    curves = hand_colour.load_curves(clips)
    grid = np.linspace(0, 1, 50)

    def resampled(c, col):
        t = np.linspace(0, 1, len(c))
        ok = np.isfinite(c[:, col])
        return np.interp(grid, t[ok], c[ok, col]) if ok.sum() > 1 else np.full(len(grid), np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for row, part in enumerate(("hand", "nail")):
        col = hand_colour.COLUMNS.index(f"{part}_a")
        for ax, obj in zip(axes[row], ("hard", "soft")):
            for lab, colour in (("light", "tab:blue"), ("firm", "tab:red")):
                idx = clips.index[(clips["set"] == "train") & (clips["object"] == obj) & (clips["label"] == lab)]
                ax.plot(grid, np.nanmean([resampled(curves[i], col) for i in idx], axis=0), color=colour,
                        label=f"{lab} (n={len(idx)})")
            ax.axvspan(*hand_colour.HOLD, color="grey", alpha=0.15, label="hold window")
            ax.set_title(f"{part}, {obj} object")
            ax.set_ylabel("a* relative to wall (redness)")
            ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("normalised time in clip")
    fig.suptitle("Mean redness over the clip, 120 train clips (plots.py)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "redness_curves.png", dpi=150)
    plt.close(fig)
    print(f"saved {(FIG_DIR / 'redness_curves.png').relative_to(common.HERE)}")
