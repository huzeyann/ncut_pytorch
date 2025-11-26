# Nyström Ncut (Quality)

While the Nyström method is often employed for its computational efficiency, it provides a distinct advantage in segmentation quality by mitigating **class imbalance**, a common pathology in graph-based spectral clustering.

## The Class Imbalance Problem

Standard Normalized Cut (NCut) optimization is sensitive to the density distribution of the data. In typical datasets, such as natural images, class sizes are highly imbalanced (e.g., a large background region versus a small foreground object).

- **Density Bias**: Since standard graph construction relies on all data points, regions with high point density (majority classes) dominate the graph volume.
- **Eigenvector Distortion**: The leading eigenvectors of the graph Laplacian tend to minimize the cut cost by isolating these massive, dense regions, often at the expense of smaller, distinct structures.
- **Result**: Minority classes are frequently merged into dominant clusters or fragmented, as the optimization objective "overlooks" them in favor of the majority signal.

## FPS as a Density Equalizer

The use of **Farthest Point Sampling (FPS)** to select landmarks for the Nyström approximation acts as a crucial regularization step.

Unlike random sampling, which preserves the underlying probability density function of the data (retaining the imbalance), FPS enforces **spatial uniformity** in the feature space:

1. **Downsampling Majority Classes**: In dense regions (large classes), FPS skips redundant points to satisfy the distance criterion.
2. **Preserving Minority Classes**: In sparse but distinct regions (small classes), FPS retains a higher proportion of points relative to the region's size.

## Impact on Segmentation

By constructing the Nyström approximation on this spatially uniform subset, we effectively compute the spectral embedding on a **rebalanced representation** of the dataset.

![TODO image]()

The resulting eigenvectors capture the geometry of the feature space rather than the density of the sampling. When these eigenvectors are interpolated back to the full dataset, they provide segmentation boundaries that respect the structural differences between classes, regardless of their relative sizes. This leads to significantly improved recall for small objects and sharper boundaries for under-represented classes.

<div style="display: flex; justify-content: space-between; gap: 20px; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0;">
  <a href="../02a_nystrom_ncut_complexity" style="flex: 1; text-decoration: none; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; transition: all 0.2s;">
    <span style="font-size: 12px; color: #666; margin-bottom: 5px;">Previous</span>
    <span style="font-size: 16px; font-weight: bold; color: #007bff;">← Nyström Ncut (Complexity)</span>
  </a>
  <a href="../" style="flex: 1; text-decoration: none; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; transition: all 0.2s;">
    <span style="font-size: 16px; font-weight: bold; color: #007bff;">Back to Overview</span>
  </a>
  <a href="../03_kway_ncut" style="flex: 1; text-decoration: none; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; align-items: flex-end; text-align: right; transition: all 0.2s;">
    <span style="font-size: 12px; color: #666; margin-bottom: 5px;">Next</span>
    <span style="font-size: 16px; font-weight: bold; color: #007bff;">K-way Discrete Ncut →</span>
  </a>
</div>
