# Methods


!!! abstract "INTRODUCTION"
    
    This documentation series guides you through the mathematical foundations and practical implementations of Normalized Cuts. You will learn about exact methods for small graphs, linear-time approximations for million-scale datasets, and specialized techniques for feature alignment and visualization.

## What you'll learn

* **Spectral Clustering**: The math behind Normalized Cuts.
* **Scalability**: How to handle large datasets efficiently with Nyström approximation.
* **Analyze Quality**: How balanced sampling improve segmentation quality.
* **Advanced Partitioning**: Multi-way cuts and discrete optimization.
* **Visualization**: Tools for interpreting high-dimensional embeddings.

## Core Modules

<!-- Row 1: Full Width -->
<div style="margin-bottom: 20px;">
  <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
    <h3 style="margin-top: 0;">1. Basic NCut</h3>
    <p>Master the foundational implementation of Normalized Cuts and spectral clustering. Understand the graph Laplacian, eigenvector computations, and the core mathematics behind the algorithm.</p>
    <a href="/methods/01_basic_ncut" style="color: #007bff; font-weight: bold; text-decoration: none;">Start learning →</a>
  </div>
</div>

<!-- Row 2: Two Columns -->
<div style="display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 40px;">

  <div style="flex: 1; min-width: 300px; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
    <h3 style="margin-top: 0;">2. Nyström NCut: Complexity</h3>
    <p>Learn how to scale spectral clustering to million-node graphs using Nyström approximation. This module covers sub-sampling strategies and kNN propagation to achieve linear time complexity O(N).</p>
    <a href="/methods/02a_nystrom_ncut_complexity" style="color: #007bff; font-weight: bold; text-decoration: none;">Explore complexity →</a>
  </div>

  <div style="flex: 1; min-width: 300px; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
    <h3 style="margin-top: 0;">3. Nyström NCut: Quality</h3>
    <p>Discover how Farthest Point Sampling (FPS) and balanced sampling strategies enhance segmentation quality, especially for handling class imbalance and boundary preservation.</p>
    <a href="/methods/02b_nystrom_ncut_quality" style="color: #007bff; font-weight: bold; text-decoration: none;">Analyze quality →</a>
  </div>

</div>

## Advanced Topics

<!-- Row 3: Full Width -->
<div style="margin-bottom: 20px;">
  <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
    <h3 style="margin-top: 0;">4. K-way Discrete NCut</h3>
    <p>Extend binary partitioning to multi-way clustering. Learn algorithms for partitioning graphs into k clusters simultaneously without recursive bisection.</p>
    <a href="/methods/03_kway_ncut" style="color: #007bff; font-weight: bold; text-decoration: none;">Go to K-way NCut →</a>
  </div>
</div>

<!-- Row 4: Two Columns -->
<div style="display: flex; flex-wrap: wrap; gap: 20px;">

  <div style="flex: 1; min-width: 300px; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
    <h3 style="margin-top: 0;">5. Mspace Coloring</h3>
    <p>Techniques for visualizing high-dimensional spectral embeddings in interpretable color spaces.</p>
    <a href="/methods/04_mspace_coloring" style="color: #007bff; font-weight: bold; text-decoration: none;">View coloring methods →</a>
  </div>

  <div style="flex: 1; min-width: 300px; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
    <h3 style="margin-top: 0;">6. Feature Alignment</h3>
    <p>Methods for aligning features from different models or embedding spaces, useful for model comparison and ensemble techniques.</p>
    <a href="/methods/05_feature_align" style="color: #007bff; font-weight: bold; text-decoration: none;">Learn alignment →</a>
  </div>

</div>
