"""Visualize how subtracting the minimum or mean pixel value changes the dynamic range of a Set68 grayscale crop."""

# %% imports
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib.pyplot as plt


from se.configs import PROJECT_ROOT


# %% load image
img_path = PROJECT_ROOT / Path("data/Set68/test053.png")
img = cv2.imread(img_path.as_posix(), cv2.IMREAD_GRAYSCALE)[50:250, -200:]
img = img.astype("float32") / 255.0
mean_val = img.mean()
min_val = img.min()
print(
    f"Mean pixel value: {mean_val:.6f}, Min pixel value: {min_val:.6f}, Max pixel value: {img.max():.6f}"
)

## plot image and image - min, image - mean
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
axs[0].set_title("Original Image")
img_minus_min = img - min_val
axs[1].imshow(img_minus_min, cmap="gray", vmin=0.0, vmax=1.0)
axs[1].set_title("Image - Min Pixel Value")
img_minus_mean = img - mean_val
axs[2].imshow(img_minus_mean, cmap="gray", vmin=0.0, vmax=1.0)
axs[2].set_title("Image - Mean Pixel Value")
for ax in axs:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% EOF
