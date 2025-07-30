# %%
import torch
from ncut_pytorch.predictor import NcutDinoPredictor
from PIL import Image
import numpy as np
# %%

predictor = NcutDinoPredictor(backbone='dino_512',
                          n_segments=[5, 10, 20, 40, 80]
                          )

default_images = ['/images/image_0.jpg', '/images/image_1.jpg', '/images/guitar_ego.jpg']

images = [Image.open(image_path) for image_path in default_images]
predictor.set_images(images)

# %%
segments = predictor.generate(n_cluster=25)
color = predictor.make_color(segments)
# %%
color = predictor.draw_segments_boundaries(color)
# %%
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i in range(3):
    axes[0, i].imshow(images[i])
    axes[1, i].imshow(color[i])

plt.show()
# %%
