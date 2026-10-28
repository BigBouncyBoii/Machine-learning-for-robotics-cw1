# Set up
import matplotlib.pyplot as plt
import numpy as np
import Function_A as fn

#settings
#parameters
darkFactor = 0.1
mirror = False
window_size = 5
ratio = 0.3

image_fruits = plt.imread("ELEC0144_Photo.jpg")
image_fruits = fn.darker(image_fruits, darkFactor)
#image_fruits = fn.mirror_image(fn.darker(image_fruits, darkFactor))

#pre-processing
R = image_fruits[:, :, 0]
G = image_fruits[:, :, 1]
B = image_fruits[:, :, 2]

fruit_masks = {}
for fruit in ('apple', 'capsicum', 'lemon'):
    mask, _ = fn.fruit_color_filter(fruit, image_fruits)
    clean_mask = fn.scatter_filter(mask, window_size, ratio)
    fruit_masks[fruit] = fn.keep_largest_region(clean_mask)

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