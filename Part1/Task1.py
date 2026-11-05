# CODE MAP of Task1.py
#   Set up         import libraries and Task1_functions, path to Fruits.jpg
#   Settings       darkFactor, mirror, window_size (scatter filter window), ratio (scatter filter threshold)
#   Load image     read Fruits.jpg, darken by darkFactor (rounded to 8 bits), mirror if requested
#   Masks          per fruit: colour filter -> scatter filter -> keep largest connected region
#   Geometry       bounding box and its midpoint, moment centroid, distance between the two; printed
#   Plot results   image with each fruit's box, box midpoint (o) and moment centroid (x)
#   AI use         GPT-6.1 Sol (setting: medium); GPT-6 Astra (setting: light)

# %% Set up
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent
image_path = Path(__file__).resolve().parent / "Fruits.jpg"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import Task1_functions as fn

# %% Settings
darkFactor = 0.8
mirror = True
window_size = 5
ratio = 0.5

# %% Load image
image_fruits = plt.imread(image_path)
if (mirror == True): 
    image_fruits = fn.mirror_image(fn.darker(image_fruits, darkFactor))
else:
    image_fruits = fn.darker(image_fruits, darkFactor)
    
# %% Masks
R = image_fruits[:, :, 0]
G = image_fruits[:, :, 1]
B = image_fruits[:, :, 2]

fruit_masks = {}
for fruit in ('apple', 'capsicum', 'lemon'):
    mask, _ = fn.fruit_color_filter(fruit, image_fruits)
    clean_mask = fn.scatter_filter(mask, window_size, ratio)
    fruit_masks[fruit] = fn.keep_largest_region(clean_mask)

# %% Geometry
fruit_data = {}
print(f'darkFactor = {darkFactor:.2f}, mirror = {mirror}')
print('Coordinates: x rightwards, y downwards; distances in pixels.')
for fruit, mask in fruit_masks.items():
    geometry = fn.calculate_BoundingBox_and_Centroid(mask)
    if geometry is None:
        fruit_data[fruit] = None
        print(f'{fruit}: not detected')
        continue
    geometry.update(fn.calculate_MomentCentroid(mask))
    dx = geometry['center_x_bb'] - geometry['center_x_m']
    dy = geometry['center_y_bb'] - geometry['center_y_m']
    geometry['centroid_distance'] = float(np.sqrt(dx**2 + dy**2))
    fruit_data[fruit] = geometry
    print(f"{fruit}: bounding-box midpoint = "
          f"({geometry['center_x_bb']:.2f}, {geometry['center_y_bb']:.2f}); "
          f"moment centroid = ({geometry['center_x_m']:.2f}, {geometry['center_y_m']:.2f}); "
          f"distance = {geometry['centroid_distance']:.2f} pixels")

# %% Plot results
plt.figure(figsize=(12, 9))
plt.imshow(image_fruits)
for fruit, color in [('apple', 'red'), ('capsicum', 'cyan'), ('lemon', 'yellow')]:
    fn.plot_all(fruit_data[fruit], fruit, color)
# Marker-only legend entries, without adding points to the image.
plt.plot([], [], 'o', color='black', label='Bounding-box midpoint')
plt.plot([], [], 'x', color='black', label='Moment centroid')
plt.legend(loc='upper right')
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.title(f'Fruit detection: darkFactor={darkFactor:.2f}, mirror={mirror}')
plt.tight_layout()
plt.show()