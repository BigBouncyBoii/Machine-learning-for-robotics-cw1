# CODE MAP of Task1.py
#   Set up         import libraries and Task1_functions, path to Fruits.jpg
#   Settings       darkFactor, mirror, window_size (scatter filter window), ratio (scatter filter threshold)
#   Load image     read Fruits.jpg, darken by darkFactor (rounded to 8 bits), mirror if requested
#   Masks          per fruit: colour filter -> scatter filter -> keep largest connected region
#   Geometry       bounding box and its midpoint, moment centroid, distance between the two; printed
#   Plot results   image with each fruit's box, box midpoint (o) and moment centroid (x)
#   AI use         GPT-6.1 Sol (setting: medium); GPT-6 Astra (setting: light)

# %% Set up
import sys                         # used to add this folder to Python's import path
from pathlib import Path           # builds file paths that work on any OS

import matplotlib.pyplot as plt    # reads the image and draws the figure
import numpy as np                 # array maths (sqrt for the distance)

project_root = Path(__file__).resolve().parent.parent           # repo folder (not used below)
image_path = Path(__file__).resolve().parent / "Fruits.jpg"     # Fruits.jpg next to this file, so it runs from any folder

sys.path.insert(0, str(Path(__file__).resolve().parent))        # let Python find Task1_functions.py in this folder
import Task1_functions as fn                                    # our helper functions, called as fn.xxx

# %% Settings
darkFactor = 0.8     # every pixel value is multiplied by this (0.8 = 20 % darker)
mirror = True        # True: flip the image left-right
window_size = 5      # scatter filter looks at a 5x5 window around each pixel
ratio = 0.5          # keep a pixel only if at least half of its window is also in the mask

# %% Load image
image_fruits = plt.imread(image_path)    # (783, 1024, 3) uint8 array: rows = y, columns = x, channels R,G,B
if (mirror == True):
    image_fruits = fn.mirror_image(fn.darker(image_fruits, darkFactor))   # darken, then flip left-right
else:
    image_fruits = fn.darker(image_fruits, darkFactor)                    # darken only

# %% Masks
R = image_fruits[:, :, 0]    # (783, 1024) red channel (not used below; the filter splits channels itself)
G = image_fruits[:, :, 1]    # (783, 1024) green channel (not used below)
B = image_fruits[:, :, 2]    # (783, 1024) blue channel (not used below)

fruit_masks = {}                                    # fruit name -> final True/False mask (783, 1024)
for fruit in ('apple', 'capsicum', 'lemon'):
    mask, _ = fn.fruit_color_filter(fruit, image_fruits)       # 1. True where the colour matches this fruit
    clean_mask = fn.scatter_filter(mask, window_size, ratio)   # 2. remove isolated/scattered True pixels
    fruit_masks[fruit] = fn.keep_largest_region(clean_mask)    # 3. keep only the biggest connected blob

# %% Geometry
fruit_data = {}                                     # fruit name -> dict of box limits and both centroids
print(f'darkFactor = {darkFactor:.2f}, mirror = {mirror}')
print('Coordinates: x rightwards, y downwards; distances in pixels.')
for fruit, mask in fruit_masks.items():
    geometry = fn.calculate_BoundingBox_and_Centroid(mask)     # box limits + box midpoint, or None if mask empty
    if geometry is None:
        fruit_data[fruit] = None                    # nothing found: plot_all will skip it
        print(f'{fruit}: not detected')
        continue                                    # go to the next fruit
    geometry.update(fn.calculate_MomentCentroid(mask))        # add center_x_m, center_y_m to the same dict
    dx = geometry['center_x_bb'] - geometry['center_x_m']     # horizontal gap between the two centroids
    dy = geometry['center_y_bb'] - geometry['center_y_m']     # vertical gap
    geometry['centroid_distance'] = float(np.sqrt(dx**2 + dy**2))   # Euclidean distance in pixels
    fruit_data[fruit] = geometry                    # store for plotting
    print(f"{fruit}: bounding-box midpoint = "
          f"({geometry['center_x_bb']:.2f}, {geometry['center_y_bb']:.2f}); "
          f"moment centroid = ({geometry['center_x_m']:.2f}, {geometry['center_y_m']:.2f}); "
          f"distance = {geometry['centroid_distance']:.2f} pixels")

# %% Plot results
plt.figure(figsize=(12, 9))          # new figure, 12 x 9 inches
plt.imshow(image_fruits)             # show the darkened/mirrored image (the one we actually processed)
for fruit, color in [('apple', 'red'), ('capsicum', 'cyan'), ('lemon', 'yellow')]:
    fn.plot_all(fruit_data[fruit], fruit, color)   # box, o = box midpoint, x = moment centroid, name label
# Empty plots: they only create legend entries for o and x, nothing is drawn on the image.
plt.plot([], [], 'o', color='black', label='Bounding-box midpoint')
plt.plot([], [], 'x', color='black', label='Moment centroid')
plt.legend(loc='upper right')
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.title(f'Fruit detection: darkFactor={darkFactor:.2f}, mirror={mirror}')
plt.tight_layout()                   # stop labels being cut off
plt.show()                           # open the window; the script waits until it is closed
