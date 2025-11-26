# Feature Alignment

!!! note "TL;DR"
    **Problem:** Feature spaces from different models or layers are incompatible.
    **Solution:** Convert absolute features to **relative affinity features** ($A_{ij} = \text{similarity}(x_i, x_j)$).
    **Key Insight:** Use **RBF Auto-scaling** to dynamically tune $\sigma$ so the affinity matrix has a constant mean density, making the alignment robust to scaling and outliers.

## Motivation

Deep learning models (e.g., CLIP, DINO, Stable Diffusion) learn robust visual concepts, but represent them in incompatible embedding spaces. Even within the same model, different layers operate in distinct coordinate systems. Direct comparison (e.g., $L_2$ distance) between these spaces is mathematically invalid.

**Feature Alignment** maps these disjoint representations into a common space where semantic correspondence is preserved. This enables cross-model visualization, layer-wise analysis, and valid distance computation.

> Inspired by [Representational Similarity Analysis (RSA)](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.004.2008/full), we leverage the insight that while absolute coordinates vary, the relative geometry of concepts remains consistent.

## Quick Start

```python
from ncut_pytorch.utils.math import rbf_affinity
import torch

# Features from two different sources (e.g., different models/layers)
features_A = torch.randn(100, 768) 
features_B = torch.randn(100, 1024)

# Transform to aligned relative space
# The resulting matrices encode the relative geometry of each set
aligned_A = rbf_affinity(features_A) # Shape: (100, 100)
aligned_B = rbf_affinity(features_B) # Shape: (100, 100)

# Now valid to compare aligned_A and aligned_B using standard metrics
```

## Results

Alignment reveals consistent semantic structures across models.

### Case Study: SAM vs. DINO

**SAM (Segment Anything Model)** focuses on boundaries and edges.
**DINO** captures high-level semantic parts.

| Model | Before Alignment | After Alignment |
| --- | --- | --- |
| **SAM** | ![before](../images/tutorials_05_feature_alignment/02_sam_original_ncut.png) | ![after](../images/tutorials_05_feature_alignment/02_sam_affinity_ncut.png) |
| **DINO v3** | ![before](../images/tutorials_05_feature_alignment/05_dinov3_original_ncut.png) | ![after](../images/tutorials_05_feature_alignment/04_dinov3_affinity_ncut.png) |

*Observe how facial regions (hair, eyes, nose) map to consistent colors after alignment, despite the models' differing original representations.*

<!-- ## Didactic Example

[Relevant Script: n25c0006a_toy_layeralign_rbf.py](../../images/n25c0006a_toy_layeralign_rbf.py)

We demonstrate alignment using synthetic datasets $X$ and $Y$ generated from a common latent prior $Z$, but subjected to different transformations.

**Scenario:**
$$
x_i = f(z_i) \quad \text{(Set 1)}
$$
$$
y_i = \frac{1}{4} f(z_i) R + [10, 10]^T \quad \text{(Set 2: Rotated, Scaled, Shifted)}
$$

### Before Alignment
Direct comparison fails because the sets are disjoint in the absolute space.
![Before Alignment](../images/tutorials_05_feature_alignment/image 1.png)

### Method: RBF Auto-scaling

We compute an affinity matrix for each set to capture its intrinsic geometry. To handle scale differences (isotropic or anisotropic) and outliers, we automatically tune the RBF kernel width $\sigma$.

The goal is to find $\sigma$ such that the mean affinity matches a target density $c$ (e.g., $c=0.05$ for sparse connections).

$$
\sigma = \underset{\sigma}{\text{argmin}} \left| \text{mean}(A(\sigma)) - c \right|
$$

### Handling Hard Cases

1. **Anisotropic Scaling:** Standard RBF is sensitive to non-uniform scaling (stretching). Auto-scaling helps by normalizing local density.
2. **Outliers:** Outliers can distort global affinity. A tighter $\sigma$ (found via auto-scaling) isolates outliers, preventing them from collapsing the structure.
3. **Non-Linear Transforms:** Even when $Y$ is a non-linear transformation of $X$ (e.g., via an MLP), the local neighborhood topology is often preserved. RBF alignment recovers this shared structure.

![After Alignment](../images/tutorials_05_feature_alignment/image 6.png)
*After alignment (using auto-scaled RBF), the 1-to-1 correspondence between clusters in Set 1 and Set 2 is restored.* -->
