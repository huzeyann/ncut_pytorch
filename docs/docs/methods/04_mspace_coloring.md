# M-space Coloring

!!! abstract "TL;DR"

    1. **Compress**: Train a parametric AutoEncoder to map eigenvectors $Z \in \mathbb{R}^{K}$ to a low-dimensional "Mood Space" (M-space, $\mathbb{R}^3$).
    2. **Preserve Structure**: Minimize **Eigenvector Consistency Loss**. The eigenvectors of the compressed 3D points should match the original high-dimensional eigenvectors.
    3. **Maximize Contrast**: Use **Repulsion Loss** to push the embedding to fill the RGB cube, maximizing color contrast and avoiding overcrowding.
    4. **Color**: Map the 3D coordinates to RGB values.

---

**M-space Coloring** is a parametric dimensionality reduction method designed specifically for visualizing spectral embeddings. It serves as an upgrade to t-SNE or UMAP for coloring purposes, offering better preservation of global structure and smoother, more consistent color transitions.

## Why M-space?

While t-SNE and UMAP are excellent for visualizing local structure, they have some limitations when used for coloring:

1. **Global Structure Distortion**: They can sometimes break the global geometry of the data.
2. **Clumping**: They often produce clusters with empty space in between, which wastes the available color space.

M-space addresses these by training a neural network (AutoEncoder) that directly optimizes for **spectral consistency**.

## How M-space Works

### The Core Idea

We want to find a mapping $f: \mathbb{R}^K \to \mathbb{R}^3$ such that if we run Normalized Cut (NCut) on the 3D points, we get the *same* eigenvectors as the original data.

This ensures that the **clustering structure** is perfectly preserved in the visualization. If two points are in the same cluster in the high-dimensional space, they will be geometrically close in the 3D M-space.

### The Loss Functions

The model is trained with a combination of losses:

1. **Eigenvector Consistency Loss (The "Soul" of M-space)**

    We compute the eigenvectors of the compressed 3D points (let's call them $\hat{Z}$) and compare them to the original eigenvectors $Z$.

    $$
    \mathcal{L}_{eig} = \| Z Z^\top - \hat{Z} \hat{Z}^\top \|_1
    $$

    This loss ensures that the *subspace* spanned by the eigenvectors is preserved. It's computationally intensive but ensures the global topology is correct.

2. **Repulsion Loss (The "Body" of M-space)**

    To make full use of the color space (the RGB cube), we want the data to spread out. M-space samples random points in the 3D grid and pushes the data away if they clump too much.

    This acts like a "gas" that fills the container (the unit cube), ensuring high contrast in the final coloring.


## Comparison with t-SNE / UMAP

| Feature | t-SNE / UMAP | M-space |
| :--- | :--- | :--- |
| **Objective** | Preserve local distances (neighbor probabilities) | Preserve spectral clustering structure (eigenvectors) |
| **Global Structure** | Can be distorted | Strongly preserved |
| **Color Space Usage** | Can form tight, separated islands | Fills the space (via Repulsion Loss) |

## Usage

You can use `mspace_color` directly from `ncut_pytorch`.

```python
from ncut_pytorch.color import mspace_color

# Assume eigenvectors is your NxK tensor of eigenvectors
# default n_dim=3 for RGB
rgb = mspace_color(eigenvectors, n_dim=3)

# Visualize
import matplotlib.pyplot as plt
plt.scatter(x, y, c=rgb.numpy(), s=1)
plt.show()
```

<div style="display: flex; justify-content: space-between; gap: 20px; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0;">
  <a href="../03_kway_ncut" style="flex: 1; text-decoration: none; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; transition: all 0.2s;">
    <span style="font-size: 12px; color: #666; margin-bottom: 5px;">Previous</span>
    <span style="font-size: 16px; font-weight: bold; color: #007bff;">← K-way Discrete NCut</span>
  </a>
  <a href="../" style="flex: 1; text-decoration: none; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; transition: all 0.2s;">
    <span style="font-size: 16px; font-weight: bold; color: #007bff;">Back to Overview</span>
  </a>
  <a href="../05_feature_align" style="flex: 1; text-decoration: none; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; align-items: flex-end; text-align: right; transition: all 0.2s;">
    <span style="font-size: 12px; color: #666; margin-bottom: 5px;">Next</span>
    <span style="font-size: 16px; font-weight: bold; color: #007bff;">Feature Alignment →</span>
  </a>
</div>
