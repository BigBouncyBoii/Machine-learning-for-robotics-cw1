# Code map:
# - Setup: imports libraries, configures filtering and mirroring, and loads the image.
# - Brightness loop: applies darkFactor values from 1.00 to 0.10 in steps of 0.05.
# - Apple filtering: uses Task1_functions to apply colour filtering, scatter filtering
#   and largest-connected-region selection in sequence.
# - Stage comparison: displays the darkened image and three processing masks in
#   a 2x2 figure, with bounding boxes around non-empty mask regions.
# - Image saving: saves one comparison PNG per brightness level in apple_stage_results.
#
# AI use: GPT6 Astra


# Set up
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import Task1_functions as fn

image_path = Path(__file__).resolve().parent / "Fruits.jpg"

# Settings
mirror = False
window_size = 5
ratio = 0.50

# Create output folder
output_folder = os.path.join(
    Path(__file__).resolve().parent,
    "apple_stage_results"
)
os.makedirs(output_folder, exist_ok=True)

# Read original image once
original_image = plt.imread(image_path)

# Dark factors: 1.00, 0.95, ..., 0.10
for step in range(20, 1, -1):
    darkFactor = step / 20

    # Always start from the original image
    image_fruits = fn.darker(original_image, darkFactor)

    if mirror:
        image_fruits = fn.mirror_image(image_fruits)

    # Stage 1: colour filter
    mask, _ = fn.fruit_color_filter("apple", image_fruits)

    # Stage 2: scatter filter
    clean_mask = fn.scatter_filter(
        mask, window_size, ratio
    )

    # Stage 3: largest connected region
    largest_mask = fn.keep_largest_region(clean_mask)

    # Plot all stages
    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 10),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    axes[0, 0].imshow(image_fruits)
    axes[0, 0].set_title("(a) Darkened image")

    stages = [
        (axes[0, 1], mask, "(b) Colour filter"),
        (axes[1, 0], clean_mask, "(c) Scatter filter"),
        (axes[1, 1], largest_mask, "(d) Largest connected region")
    ]

    for ax, stage_mask, title in stages:
        ax.imshow(
            stage_mask,
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest"
        )
        ax.set_title(title)

        # Outline the bounding box of each stage
        geometry = fn.calculate_BoundingBox_and_Centroid(
            stage_mask
        )

        if geometry is not None:
            x_min = geometry["x_min"]
            x_max = geometry["x_max"]
            y_min = geometry["y_min"]
            y_max = geometry["y_max"]

            ax.plot(
                [x_min, x_max, x_max, x_min, x_min],
                [y_min, y_min, y_max, y_max, y_min],
                color="cyan",
                linewidth=1.5
            )

    for ax in axes.flat:
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")

    fig.suptitle(
        f"Apple processing stages | "
        f"darkFactor={darkFactor:.2f} | mirror={mirror}",
        fontsize=16
    )

    # Save stage comparison
    filename = (
        f"apple_stages_dark_{darkFactor:.2f}"
        f"_mirror_{mirror}.png"
    )
    save_path = os.path.join(output_folder, filename)

    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {save_path}")

print("\nFinished: 19 apple stage comparison images saved.")