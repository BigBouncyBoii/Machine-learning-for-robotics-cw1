# Set up
import os
import matplotlib.pyplot as plt
import functions_A as fn

# Settings
mirror = False
window_size = 5
ratio = 0.50

# Image path
image_path = r"D:\ELEC0144_MachineLearning\GroupAssignment_1_G7\Fruits.jpg"

# Create output folder
output_folder = os.path.join(
    os.path.dirname(image_path),
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