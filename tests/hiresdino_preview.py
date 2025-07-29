# %%
import torch
from ncut_pytorch.predictor import NcutPredictor
from PIL import Image
import numpy as np
# %%

predictor = NcutPredictor(backbone='dino_512',
                          n_eig_hierarchy=[5, 10, 20, 40, 80]
                          )

default_images = ['/images/image_0.jpg', '/images/image_1.jpg', '/images/guitar_ego.jpg']

images = [Image.open(image_path) for image_path in default_images]
predictor.set_images(images)

# %%
%matplotlib widget
# %%
heatmaps, masks = predictor.preview([500, 500], 0)

import matplotlib.pyplot as plt

def blend_mask(img, mask, alpha=0.8):
    mask = mask.resize((img.width, img.height), resample=Image.Resampling.NEAREST)
    img = np.array(img).astype(np.float32)
    mask = np.array(mask).astype(np.float32)
    blend = img * (1 - alpha) + mask * alpha
    h, w = blend.shape[:2]
    blend = blend.reshape(h*w, 3)
    mask = mask.reshape(h*w, 3)
    img = img.reshape(h*w, 3)
    blend[mask[:, 0] == 255] = img[mask[:, 0] == 255]
    blend = blend.reshape(h, w, 3)
    blend = blend.astype(np.uint8)
    return Image.fromarray(blend)

def plot_masks(images, masks, fig, axes):
    color_values = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    color_values = plt.get_cmap('plasma_r')(color_values)
    color_values = (color_values * 255).astype(np.uint8)

    for idx, (ax, img, mask_set) in enumerate(zip(axes, images, masks)):
        ax.clear()
        mask = np.ones((*mask_set[0].shape, 3)) * 255
        mask = mask.astype(np.uint8)
        for i in range(len(color_values)):
            mask[masks[i][idx]] = color_values[i][:3]
        mask_img = Image.fromarray(mask)
        ax.imshow(blend_mask(img, mask_img, alpha=0.8))
        ax.set_axis_off()
    fig.canvas.draw_idle()

fig, axes = plt.subplots(2, 2, figsize=(5, 5))
axes = axes.flatten()
plot_masks(images, masks, fig, axes)

def on_hover(event):
    if event.inaxes is None or event.xdata is None or event.ydata is None:
        return
    for idx, ax in enumerate(axes):
        if ax == event.inaxes:
            x, y = event.xdata, event.ydata
            # Recompute masks for all images, using the hovered coordinates and image index
            heatmaps_new, masks_new = predictor.preview([x, y], idx)
            plot_masks(images, masks_new, fig, axes)
            break

cid = fig.canvas.mpl_connect('motion_notify_event', on_hover)
plt.show()

# %%



# %%
