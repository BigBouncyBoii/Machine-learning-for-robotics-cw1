# CODE MAP of plots.py   (figures for the report; called by main.py --figures)
#   Per person    per_person(): validation accuracy for each held-out person, mean and guessing lines
#   Confusion     confusion(): pooled leave-one-person-out predictions of the 120 train clips
#   AI use        Claude Opus 5.5 (claude-opus-5-5), default setting (restructure); original code: <Jeevan: model, version, setting>

# %% Per person
import matplotlib.pyplot as plt
import numpy as np

import common

FIG_DIR = common.RESULTS / "figures"


def per_person(people, accuracies, mean_acc):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    bars = ax.bar(people, np.asarray(accuracies) * 100, color="#4c78a8")
    ax.axhline(50, color="#e45756", linestyle="--", linewidth=1.8, label="Guessing: 50%")
    ax.axhline(mean_acc * 100, color="#72b7b2", linestyle=":", linewidth=1.8, label=f"Mean: {mean_acc:.1%}")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_xlabel("Held-out person (trained on the other three, 120 train clips only)")
    ax.set_title("Joint angles + logistic regression, leave-one-person-out (main.py)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for bar, acc in zip(bars, accuracies):
        ax.annotate(f"{acc:.1%}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom")
    fig.savefig(FIG_DIR / "validation_per_person.png", dpi=180)
    plt.close(fig)
    print(f"saved {(FIG_DIR / 'validation_per_person.png').relative_to(common.HERE)}")


# %% Confusion
def confusion(y, pred):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cm = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(y, pred):
        cm[int(actual), int(predicted)] += 1
    fig, ax = plt.subplots(figsize=(5.8, 5), layout="constrained")
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Light", "Firm"])
    ax.set_yticks([0, 1], labels=["Light", "Firm"])
    ax.set_xlabel("Predicted grip")
    ax.set_ylabel("Actual grip")
    ax.set_title(f"Pooled validation predictions, 120 train clips ({np.mean(y == pred):.1%})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() * 0.9 else "black")
    fig.colorbar(image, ax=ax, label="Number of clips")
    fig.savefig(FIG_DIR / "validation_confusion.png", dpi=180)
    plt.close(fig)
    print(f"saved {(FIG_DIR / 'validation_confusion.png').relative_to(common.HERE)}")
