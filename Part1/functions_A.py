import matplotlib.pyplot as plt
import numpy as np

# fnA1: color filter
def fruit_color_filter(fruit, image):

    # Separate RGB values
    rgb = image[:, :, :3].astype(float)

    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]

    # Filter color according to the selected fruit
    if fruit == "apple":
        mask = (
            (R > 1.5 * G)
            & (R > 2.2 * B)
            & (B > 0.8 * G)
        )

    elif fruit == "capsicum":
        mask = (
            (R > 1.8 * G)
            & (R < 3.0 * G)
            & (B < 0.10 * R)
            & (B < 0.25 * G)
        )

    elif fruit == "lemon":
        mask = (
            (G > 0.65 * R)
            & (G < 1.05 * R)
            & (B < 0.55 * R)
            & (B < 0.70 * G)
        )
        
    else:
        raise ValueError("fruit color filter not available yet")

    ColorFiltered_image = np.zeros_like(image)
    ColorFiltered_image[mask] = image[mask]

    return mask, ColorFiltered_image

# fnA2: scatter filter
def scatter_filter(mask, window_size, ratio):

    # Window size must be a positive odd integer
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")

    # Ratio must be between 0 and 1
    if not 0 <= ratio <= 1:
        raise ValueError("ratio must be between 0 and 1")

    ratio = round(ratio, 2)

    # Integer padding width and threshold
    extension = (window_size - 1) // 2
    threshold = round(ratio * window_size**2)

    padded = np.pad(
        mask, extension,
        mode="constant",
        constant_values=False
    )

    count = np.zeros(mask.shape, dtype=int)

    for i in range(window_size):
        for j in range(window_size):
            count += padded[
                i:i + mask.shape[0],
                j:j + mask.shape[1]
            ]

    clean_mask = mask & (count >= threshold)

    return clean_mask

# fnA3：Area filter keep the largest region
def keep_largest_region(mask):

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool) # prevent repetition
    largest_region = []

    for y, x in zip(*np.where(mask)): # find coordinates of the pixels that is TRUE in the mask
        if visited[y, x]:
            continue

        stack = [(y, x)] # the selected pixel 
        visited[y, x] = True # note the pixel
        region = [] # saving other pixels in the same region

        while stack:
            cy, cx = stack.pop() # remove the checking pixel from the list and copy the value to cy & cx
            region.append((cy, cx)) 

            # checking surrounding pixels
            for dy in (-1, 0, 1): 
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0: #skip the checking pixel itself
                        continue

                    ny = cy + dy
                    nx = cx + dx

                    if 0 <= ny < height and 0 <= nx < width: #nx ny within range
                        if mask[ny, nx] and not visited[ny, nx]: #the neighbouring pixel is TRUE in mask and not visited yet
                            visited[ny, nx] = True # note visited
                            stack.append((ny, nx)) # ny, nx be the next checking pixel

        if len(region) > len(largest_region): # comparing region
            largest_region = region

    largest_mask = np.zeros_like(mask, dtype=bool)

    if largest_region:
        ys, xs = np.array(largest_region).T
        largest_mask[ys, xs] = True

    return largest_mask

import numpy as np
import matplotlib.pyplot as plt


#fn4: Calculate bounding box and centroid
#example of using the func in main code
#fruit_boundingbox_data = {
#    "apple": fn.calculate_geometry(largest_mask_apple),
#    "capsicum": fn.calculate_geometry(largest_mask_capsicum),
#    "lemon": fn.calculate_geometry(largest_mask_lemon)
#}

def calculate_BoundingBox_and_Centroid(mask):
    
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None

    return {
        "x_min": int(xs.min()),
        "x_max": int(xs.max()),
        "y_min": int(ys.min()),
        "y_max": int(ys.max()),
        #"center_x_bb": float(xs.mean()),
        #"center_y_bb": float(ys.mean())
        "center_x_bb": float((xs.min()+xs.max())/2),
        "center_y_bb": float((ys.min()+ys.max())/2)
    }


#fn5: plot bounding box and centroids(both bounding box and moment centroids)
def plot_all(geometry, name, color):

    if geometry is None:
        return

    x_min = geometry["x_min"]
    x_max = geometry["x_max"]
    y_min = geometry["y_min"]
    y_max = geometry["y_max"]

    center_x_bb = geometry["center_x_bb"]
    center_y_bb = geometry["center_y_bb"]
    center_x_m = geometry["center_x_m"]
    center_y_m = geometry["center_y_m"]

    plt.plot(
        [x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min],
        color=color,
        linewidth=2
    )

    plt.plot(
        center_x_bb, center_y_bb,
        marker="o",
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.5,
        markersize=8,
        linestyle="None",
        zorder=5
    )

    plt.plot(
        center_x_m, center_y_m,
        marker="x",
        color="black",
        markeredgewidth=2.5,
        markersize=12,
        linestyle="None",
        zorder=6
    )

    plt.text(
        x_min, y_min - 5, name,
        color=color,
        fontsize=12
    )

    print(f"{name}:")
    print(f"  Bounding box midpoint: ({center_x_bb:.2f}, {center_y_bb:.2f})")
    print(f"  Moment centroid: ({center_x_m:.2f}, {center_y_m:.2f})")




#fn6: find the centroid using the moment method
def calculate_MomentCentroid(mask):

    # Binary image: selected pixels = 1, others = 0
    binary = mask.astype(float)

    # Coordinate arrays, matching the shape of mask
    y, x = np.indices(mask.shape)

    # Image moments
    M00 = np.sum(binary)
    M10 = np.sum(x * binary)
    M01 = np.sum(y * binary)

    if M00 == 0:
        return None

    # Centroid calculated using moments
    center_x_m = M10 / M00
    center_y_m = M01 / M00

    return {
        "center_x_m": float(center_x_m),
        "center_y_m": float(center_y_m)
    }

#fnA7: Darken a uint8 image, round and convert back to 8 bits
def darker(image, darkFactor):

    darkened = image.astype(float) * darkFactor
    return np.rint(darkened).astype(np.uint8) # round to int first then change to 8bits


#fnA8: Flip the image left-to-right
def mirror_image(image):

    return np.flip(image, axis=1).copy()