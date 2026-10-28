# Set up
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import functions_A as fn

# Settings
mirror = False
window_size = 5
ratio = 0.50

# Replace with your actual image path
image_path = r"D:\ELEC0144_MachineLearning\GroupAssignment_1_G7\Fruits.jpg"

# Create output folder
output_folder = os.path.join(
    os.path.dirname(image_path),
    "darkness_results"
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
        fn.plot_all(fruit_data[fruit], fruit, color)

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