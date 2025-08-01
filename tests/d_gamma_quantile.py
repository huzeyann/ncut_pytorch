# %%
import torch
from ncut_pytorch.predictor import NcutDinoPredictor
from PIL import Image
import numpy as np
# %%

predictor = NcutDinoPredictor(dtype=torch.float32, super_resolution=False)
predictor = predictor.to('cpu')
# predictor.run_faster()

default_images = ['./image_0.jpg']
inference_images = ['./image_1.jpg'] * 1

images = [Image.open(image_path) for image_path in default_images]
inference_images = [Image.open(image_path) for image_path in inference_images]
predictor.set_images(images)

# %%
features = predictor.predictor._features
print(features.shape)
# %%
from ncut_pytorch.utils.math import rbf_affinity
A = rbf_affinity(features)

# %%
print(A.mean())
# %%
from ncut_pytorch.utils.gamma import find_gamma_by_degree_after_fps
degree = 'auto'
gamma = find_gamma_by_degree_after_fps(features, degree)
print(gamma)
# %%
print(rbf_affinity(features, gamma=gamma).mean())
# %%
import matplotlib.pyplot as plt
plt.hist(A.mean(1).flatten(), bins=100)
# show mean and median of A.mean(1)
print(f"mean: {A.mean(1).mean()}, median: {np.median(A.mean(1))}")
plt.show()
# %%
plt.hist(rbf_affinity(features, gamma=gamma).mean(1).flatten(), bins=100)
# show mean and median of A.mean(1)
print(f"mean: {rbf_affinity(features, gamma=gamma).mean(1).mean()}, median: {np.median(rbf_affinity(features, gamma=gamma).mean(1))}")
plt.show()
# %%
torch.quantile(A.mean(1), 0.5)
# %%
A.mean(1).mode()
# %%
