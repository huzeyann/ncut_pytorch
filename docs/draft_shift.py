#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate 1000 2D points in 4 clusters
X, y = make_blobs(n_samples=30, centers=3, n_features=2, random_state=42)
arg_sort = np.argsort(y)
X = X[arg_sort]
y = y[arg_sort]

# Identify centroids of clusters
centroids = np.array([X[y == i].mean(axis=0) for i in range(4)])

# Create a line of dots connecting the 2nd and 3rd cluster centroids
num_line_points = 10
line_points = np.linspace(centroids[0], centroids[1], num_line_points)

# Plot the points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10_r', s=50, alpha=0.6)
plt.scatter(line_points[:, 0], line_points[:, 1], c='grey', s=50, alpha=0.6)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Randomly Placed 2D Dots in 3 Clusters with Connecting Line')
plt.show()
# %%
all_dots = np.concatenate([X, line_points], axis=0)
from ncut_pytorch import NCUT

eigenvectors, eigenvalues = NCUT(10, distance='euclidean', normalize_features=False).fit_transform(torch.tensor(all_dots).float())
fig, axs = plt.subplots(1, 6, figsize=(15, 2))
for i, ax in enumerate(axs):
    vminmax = max(abs(eigenvectors[:, i].min()), abs(eigenvectors[:, i].max()))
    ax.scatter(all_dots[:, 0], all_dots[:, 1], c=eigenvectors[:, i], cmap='coolwarm', s=50, alpha=0.9, vmin=-vminmax, vmax=vminmax)
    ax.set_title(f'Eigenvalue={eigenvalues[i]:.2f}')
    if i == 0:
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
    # ax.set_xticks([])
    # ax.set_yticks([])
    
# %%
all_dots = np.concatenate([X, line_points], axis=0) + 1000
from ncut_pytorch import NCUT

eigenvectors, eigenvalues = NCUT(10, distance='euclidean', normalize_features=False).fit_transform(torch.tensor(all_dots).float())
fig, axs = plt.subplots(1, 6, figsize=(15, 2))
for i, ax in enumerate(axs):
    vminmax = max(abs(eigenvectors[:, i].min()), abs(eigenvectors[:, i].max()))
    ax.scatter(all_dots[:, 0], all_dots[:, 1], c=eigenvectors[:, i], cmap='coolwarm', s=50, alpha=0.9, vmin=-vminmax, vmax=vminmax)
    ax.set_title(f'Eigenvalue={eigenvalues[i]:.2f}')
    if i == 0:
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
    # ax.set_xticks([])
    # ax.set_yticks([])
# %%
