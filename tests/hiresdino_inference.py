# %%
import torch
from ncut_pytorch.predictor.dino_predictor import NcutDinoPredictor
from PIL import Image
import numpy as np
# %%

predictor = NcutDinoPredictor(dtype=torch.float32)
predictor = predictor.to('mps')
predictor.run_faster()

default_images = ['./image_0.jpg']
inference_images = ['./image_1.jpg'] * 1

images = [Image.open(image_path) for image_path in default_images]
inference_images = [Image.open(image_path) for image_path in inference_images]
predictor.set_images(images)


# %%
mask, heatmap = predictor.predict(
    # np.array([[248, 575], [640, 394]]), np.array([1, 1]), np.array([0, 0]), 
    # np.array([[513, 838]]), np.array([1]), np.array([0]), 
    np.array([[440, 721]]), np.array([1]), np.array([0]), 
    click_weight=0.5,
    bg_weight=0.1,
    )

import matplotlib.pyplot as plt
plt.imshow(heatmap[0].cpu().numpy())
plt.show()
plt.imshow(mask[0].cpu().numpy())
plt.show()

# %%
import time
start_time = time.time()
mask, heatmap = predictor.inference(inference_images)
end_time = time.time()
print(f"inference_fast time: {end_time - start_time}")
plt.imshow(heatmap[0].cpu().numpy())
plt.show()
plt.imshow(mask[0].cpu().numpy())
plt.show()
# %%
