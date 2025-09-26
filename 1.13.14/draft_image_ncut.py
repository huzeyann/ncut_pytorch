# %%
import requests
from PIL import Image

url = "https://huzeyann.github.io/assets/img/prof_pic_old.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image
# %%
from ncut_pytorch.backbone import load_model
from ncut_pytorch import NCUT, rgb_from_tsne_3d
# %%
model = load_model("DiNO(dino_vits8_448)")
# %%
image = Image.open(requests.get(url, stream=True).raw)
image = image.convert("RGB")
image = image.resize((448, 448))
import torch
import numpy as np
image = np.array(image)
image = torch.tensor(image).permute(2, 0, 1).float() / 255.
image = image / 0.5 - 1
image = image.unsqueeze(0)
feats = model(image)
# %%
feat = feats['block'][-1]
# %%
feat.shape
# %%
eigenvectors, eigenvalues = NCUT(num_eig=100).fit_transform(feat.reshape(-1, feat.shape[-1]))
# %%
h, w = feat.shape[1], feat.shape[2]
# visualize top 9 eigenvectors, 3 eigenvectors per row
import matplotlib.pyplot as plt
from ncut_pytorch import quantile_normalize
fig, axs = plt.subplots(3, 4, figsize=(13, 10))
i_eig = 0
for i_row in range(3):
    for i_col in range(1, 4):
        ax = axs[i_row, i_col]
        # ax.imshow(eigenvectors[:, i_eig].reshape(h, w), cmap="coolwarm", vmin=-0.1, vmax=0.1)
        im = eigenvectors[:, i_eig].reshape(h, w)
        vmaxmin = max(abs(im.min()), abs(im.max()))
        ax.imshow(im, cmap="coolwarm", vmin=-vmaxmin, vmax=vmaxmin)
        ax.set_title(f"lambda_{i_eig} = {eigenvalues[i_eig]:.3f}")
        ax.axis("off")
        i_eig += 1
for i_row in range(3):
    ax = axs[i_row, 0]
    start, end = i_row * 3, (i_row + 1) * 3
    rgb = quantile_normalize(eigenvectors[:, start:end]).reshape(h, w, 3)
    ax.imshow(rgb)
    ax.set_title(f"eigenvectors {start}-{end-1}")
    ax.axis("off")
plt.suptitle("Top 9 eigenvectors of Ncut DiNOv2 last layer features")
plt.tight_layout()
plt.show()
# %%
# show 10 eigenvectors 2x5
fig, axs = plt.subplots(4, 5, figsize=(10, 8))
h, w = feat.shape[1], feat.shape[2]
i_eig = 0
for i_ax, ax in enumerate(axs.flat):
    im = eigenvectors[:, i_eig].reshape(h, w)
    vmaxmin = max(abs(im.min()), abs(im.max()))
    ax.imshow(im, cmap="coolwarm", vmin=-vmaxmin, vmax=vmaxmin)
    # ax.set_title(f"lambda_{i_eig} = {eigenvalues[i_eig]:.3f}")
    ax.set_title(f"eigenvector {i_eig}")
    ax.axis("off")
    i_eig += 1
# %%
_, rgb = rgb_from_tsne_3d(eigenvectors, perplexity=120)
# %%
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
image_rgb = rgb.reshape(h, w, 3)
image = Image.open(requests.get(url, stream=True).raw)
axs[0].imshow(image)
axs[0].axis("off")
# axs[0].set_title("Original Image")
axs[1].imshow(image_rgb)
axs[1].axis("off")
axs[1].set_title("Eigenvectors", fontsize=16)
# %%
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=20)
clusters = kmeans.fit_predict(eigenvectors[:, :20])
# %%
cluster_image = clusters.reshape(h, w)
plt.imshow(cluster_image, cmap="tab20")
# %%
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
axs[0].imshow(image_rgb)
axs[0].axis("off")
axs[0].set_title("spectral-tSNE")
axs[1].imshow(cluster_image, cmap="tab20")
axs[1].axis("off")
axs[1].set_title("KMeans")

# %%
