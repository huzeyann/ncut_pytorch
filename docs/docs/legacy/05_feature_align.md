# Feature Alignment

!!! note "TL;DR"
    **Problem:** Feature spaces from different models or layers are incompatible.
    **Solution:** Convert absolute features to **relative affinity features** ($A_{ij} = \text{similarity}(x_i, x_j)$).
    
## Motivation

Comparing representations across different models (e.g., DINO vs. CLIP) or layers is challenging because features reside in distinct embedding spaces with different coordinate systems and manifold geometries. Direct metrics like Euclidean distance are often invalid across these spaces.

**Feature Alignment** transforms these diverse representations into a unified space where corresponding semantic concepts are mapped to consistent locations. This enables valid comparison, cross-model visualization, and transfer learning.

> Inspired by [Representational Similarity Analysis (RSA)](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.004.2008/full)

## Method: RBF Auto-scaling (Intrinsic Geometry)

The most direct alignment method relies on the **intrinsic geometry** of the data. While absolute coordinates differ, the relative relationships (distances) between data points within a dataset should remain consistent if the underlying semantic structure is the same.

We convert absolute features into a relative representation using an affinity matrix.

### The Transformation

Given features $X \in \mathbb{R}^{n \times d}$, we compute the RBF affinity matrix $A \in \mathbb{R}^{n \times n}$. We then use the rows of this matrix as the new feature representation:

$$
\text{New Feature}(x_i) = A_i = [A_{i,1}, A_{i,2}, \dots, A_{i,n}] \in \mathbb{R}^n
$$

Where each entry is the RBF similarity:
$$
A_{ij} = \exp\left(-\frac{\|x_i - x_j\|^2}{\sigma}\right)
$$

**Why this works**: Instead of describing "where I am" (absolute coordinates), the new feature vector $A_i$ describes "how I relate to everyone else" (relative topology). Since the relative topology is preserved across aligned models, these new feature vectors can be directly compared using standard metrics (e.g., Euclidean distance or Cosine similarity) even if the original spaces were incompatible.

### Auto-scaling Algorithm

Standard RBF kernels can be sensitive to anisotropic scaling and density variations. **RBF Auto-scaling** dynamically solves for a kernel width $\sigma$ such that the resulting affinity matrix maintains a target sparsity or mean connection strength $c$.

$$
\sigma = \underset{\sigma}{\text{argmin}} \left| \text{mean}(A(\sigma)) - c \right|
$$

By enforcing a constant mean affinity (e.g., $c=0.05$) across different feature sets, we ensure that the aligned representations share the same connectivity properties, making the alignment robust to outliers and varying embedding scales.

<div style="display: flex; justify-content: space-between; gap: 20px; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0;">
  <a href="../04_mspace_coloring" style="flex: 1; text-decoration: none; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; transition: all 0.2s;">
    <span style="font-size: 12px; color: #666; margin-bottom: 5px;">Previous</span>
    <span style="font-size: 16px; font-weight: bold; color: #007bff;">← Mspace Coloring</span>
  </a>
  <a href="../" style="flex: 1; text-decoration: none; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; transition: all 0.2s;">
    <span style="font-size: 16px; font-weight: bold; color: #007bff;">Back to Overview</span>
  </a>
  <a href="#" style="flex: 1; visibility: hidden;"></a>
</div>
