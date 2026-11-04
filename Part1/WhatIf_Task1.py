# Code map:
# - Setup: imports libraries, configures filtering and mirroring, and loads the image.
# - Brightness loop: applies darkFactor values from 1.00 to 0.10 in steps of 0.05.
# - Fruit detection: applies colour and scatter filters and keeps the largest region
#   for each apple, capsicum and lemon mask using Task1_functions.
# - Geometry: calculates bounding boxes, bounding-box midpoints, moment centroids
#   and distances between the two centres; records failed detections as missing data.
# - Plotting and image saving: overlays geometry and centroid distances and saves
#   one annotated PNG for each brightness level in Task1_Whatif_Results.
# - Comparison figure: saves a 4x5 overview at 5 cm height as PNG and PDF.
# - Data export: saves detection status and geometry from all runs to a CSV file.
#
# AI use: GPT-6.1 Sol (setting: medium); GPT-6 Astra (setting: light)


# Set up
import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
import numpy as np

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
    "Task1_Whatif_Results"
)
os.makedirs(output_folder, exist_ok=True)

# Read original image once
original_image = plt.imread(image_path)

# Collect data from all runs
all_results = []

# Run darkFactor from 1.00 to 0.10, decreasing by 0.05
for step in range(20, 1, -1):
    darkFactor = step / 20

    # Always apply darkening to the original image
    image_fruits = fn.darker(original_image, darkFactor)

    if mirror:
        image_fruits = fn.mirror_image(image_fruits)

    # Colour filtering, scatter filtering and largest region
    fruit_masks = {}

    for fruit in ("apple", "capsicum", "lemon"):
        mask, _ = fn.fruit_color_filter(fruit, image_fruits)
        clean_mask = fn.scatter_filter(mask, window_size, ratio)
        fruit_masks[fruit] = fn.keep_largest_region(clean_mask)

    # Calculate bounding boxes and centroids
    fruit_data = {}
    distance_text = ["Centroid distance:"]

    print(f"\ndarkFactor = {darkFactor:.2f}, mirror = {mirror}")
    print("Coordinates: x rightwards, y downwards; distances in pixels.")

    for fruit, mask in fruit_masks.items():
        geometry = fn.calculate_BoundingBox_and_Centroid(mask)

        # Create a record, including empty values if detection fails
        record = {
            "darkFactor": darkFactor,
            "mirror": mirror,
            "fruit": fruit,
            "detected": False,
            "x_min": np.nan,
            "x_max": np.nan,
            "y_min": np.nan,
            "y_max": np.nan,
            "center_x_bb": np.nan,
            "center_y_bb": np.nan,
            "center_x_m": np.nan,
            "center_y_m": np.nan,
            "centroid_distance": np.nan
        }

        if geometry is None:
            fruit_data[fruit] = None
            distance_text.append(f"{fruit}: not detected")
            all_results.append(record)

            print(f"{fruit}: not detected")
            continue

        # Add moment centroid to geometry
        geometry.update(fn.calculate_MomentCentroid(mask))

        # Distance between the two centres
        dx = geometry["center_x_bb"] - geometry["center_x_m"]
        dy = geometry["center_y_bb"] - geometry["center_y_m"]

        geometry["centroid_distance"] = float(
            np.sqrt(dx**2 + dy**2)
        )

        fruit_data[fruit] = geometry

        # Collect bounding box, centroid and distance data
        record.update(geometry)
        record["detected"] = True
        all_results.append(record)

        distance_text.append(
            f"{fruit}: {geometry['centroid_distance']:.2f} pixels"
        )

        print(
            f"{fruit}: bounding box = "
            f"x[{geometry['x_min']}, {geometry['x_max']}], "
            f"y[{geometry['y_min']}, {geometry['y_max']}]"
        )
        print(
            f"  Bounding-box midpoint = "
            f"({geometry['center_x_bb']:.2f}, "
            f"{geometry['center_y_bb']:.2f})"
        )
        print(
            f"  Moment centroid = "
            f"({geometry['center_x_m']:.2f}, "
            f"{geometry['center_y_m']:.2f})"
        )
        print(
            f"  Distance = "
            f"{geometry['centroid_distance']:.2f} pixels"
        )

    # Plot results
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(image_fruits)

    for fruit, color in [
        ("apple", "red"),
        ("capsicum", "cyan"),
        ("lemon", "yellow")
    ]:
        first_line = len(ax.lines)
        fn.plot_all(fruit_data[fruit], fruit, color)
        if len(ax.lines) > first_line:
            box_line = ax.lines[first_line]
            box_line.set_linewidth(6)
            box_line.set_path_effects([
                pe.Stroke(linewidth=8, foreground="black"), pe.Normal()
            ])

    # Display centroid distances on the image
    ax.text(
        0.02, 0.98,
        "\n".join(distance_text),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="white",
        bbox=dict(
            facecolor="black",
            alpha=0.75,
            edgecolor="none"
        ),
        zorder=10
    )

    # Marker-only legend entries
    ax.plot(
        [], [],
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        linestyle="None",
        label="Bounding-box midpoint"
    )
    ax.plot(
        [], [],
        marker="x",
        color="black",
        linestyle="None",
        label="Moment centroid"
    )

    ax.legend(loc="upper right")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title(
        f"Fruit detection: darkFactor={darkFactor:.2f}, "
        f"mirror={mirror}"
    )

    fig.tight_layout()

    # Save image
    filename = (
        f"fruits_dark_{darkFactor:.2f}_mirror_{mirror}.png"
    )
    save_path = os.path.join(output_folder, filename)

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved image: {save_path}")

# Save collected data as CSV
results_table = pd.DataFrame(all_results)

csv_path = os.path.join(
    output_folder,
    f"fruit_detection_data_mirror_{mirror}.csv"
)

results_table.to_csv(
    csv_path,
    index=False,
    na_rep=""
)

print(f"\nData saved: {csv_path}")
print("Finished: 19 images and 57 data records saved.")

# Compact overview: dimensions and line widths are set for a 5 cm high figure.
overview_height_cm = 5.0
fig, axes = plt.subplots(4, 5, figsize=(8.0 / 2.54, overview_height_cm / 2.54))
fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995,
                    wspace=0.035, hspace=0.045)
for ax, step in zip(axes.flat, range(20, 1, -1)):
    factor = step / 20
    panel_image = fn.darker(original_image, factor)
    if mirror:
        panel_image = fn.mirror_image(panel_image)
    ax.imshow(panel_image)
    ax.set_axis_off()
    for record in all_results:
        if record["darkFactor"] != factor or not record["detected"]:
            continue
        colour = {"apple": "red", "capsicum": "cyan", "lemon": "yellow"}[record["fruit"]]
        box = Rectangle((record["x_min"], record["y_min"]),
                        record["x_max"] - record["x_min"],
                        record["y_max"] - record["y_min"],
                        fill=False, edgecolor=colour, linewidth=0.9)
        box.set_path_effects([pe.Stroke(linewidth=1.5, foreground="black"), pe.Normal()])
        ax.add_patch(box)
    ax.text(0.94, 0.07, f"{factor:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="#173c4b", linewidth=0.6,
                      boxstyle="round,pad=0.15"))
key = axes.flat[-1]
key.set_axis_off()
key.text(0.08, 0.84, "Key:", fontsize=5.5, fontweight="bold", transform=key.transAxes)
key.text(0.5, 0.49, "X.XX", ha="center", fontsize=6, fontweight="bold",
         transform=key.transAxes,
         bbox=dict(facecolor="white", edgecolor="#173c4b", boxstyle="round,pad=0.2"))
key.text(0.5, 0.15, "Dark factor", ha="center", fontsize=5.5,
         fontweight="bold", transform=key.transAxes)
for extension in ("png", "pdf"):
    overview_path = Path(output_folder) / f"fruit_darkness_comparison_5cm.{extension}"
    fig.savefig(overview_path, dpi=600, facecolor="white")
    print(f"Saved overview: {overview_path}")
plt.close(fig)
