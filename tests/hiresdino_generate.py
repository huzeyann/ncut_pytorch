# %%
import torch
from ncut_pytorch.predictor import NcutDinoPredictor
from PIL import Image
import numpy as np
# %%

predictor = NcutDinoPredictor(dtype=torch.float16)
predictor = predictor.to('cuda')

default_images = ['/images/image_0.jpg', '/images/image_1.jpg', '/images/guitar_ego.jpg']

images = [Image.open(image_path) for image_path in default_images]
predictor.set_images(images)

# %%
segments = predictor.generate(n_cluster=50)
color = predictor.color_discrete(segments)
# %%
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i in range(3):
    axes[0, i].imshow(images[i])
    axes[1, i].imshow(color[i])

plt.show()
# %%
