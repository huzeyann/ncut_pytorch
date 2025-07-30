# %%
import torch
from ncut_pytorch.predictor import NcutDinoPredictor
from PIL import Image
import numpy as np
# %%

predictor = NcutDinoPredictor(backbone='dino_512')

default_images = ['/images/image_0.jpg']
inference_images = ['/images/image_1.jpg'] * 10

images = [Image.open(image_path) for image_path in default_images]
inference_images = [Image.open(image_path) for image_path in inference_images]
predictor.set_images(images)


# %%
mask, heatmap = predictor.predict(
    # np.array([[248, 575], [640, 394]]), np.array([1, 1]), np.array([0, 0]), 
    # np.array([[513, 838]]), np.array([1]), np.array([0]), 
    np.array([[440, 721]]), np.array([1]), np.array([0]), 
    n_clusters=2,
    click_weight=0.5,
    bg_weight=0.1,
    )
mask.shape
heatmap.shape

import matplotlib.pyplot as plt
plt.imshow(heatmap[0].cpu().numpy())
plt.show()
plt.imshow(mask[0].cpu().numpy())
plt.show()

# %%
import time
start_time = time.time()
mask, heatmap = predictor.inference(inference_images)
plt.imshow(heatmap[0].cpu().numpy())
plt.show()
plt.imshow(mask[0].cpu().numpy())
plt.show()
end_time = time.time()
print(f"inference_fast time: {end_time - start_time}")
# %%
