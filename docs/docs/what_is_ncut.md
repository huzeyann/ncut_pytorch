# What is Normalized Cut?

!!! abstract "TL;DR"
    **Normalized Cut** embeds data by finding the "weakest links" in a similarity graph.

    `ncut-pytorch` uses the **Nyström method** to make Ncur fast ($O(N)$) and memory-efficient ($O(1)$), enabling scaling to millions of points (e.g., high-res images, videos) on a GPU.

---

## Overview

**Normalized Cut (Ncut)**, also known as **Spectral Clustering**, is a graph-theoretic method for unsupervised data grouping. It analyzes the "spectrum" (eigenvectors) of the data's similarity graph to find natural clusters.

**However, the most powerful part is the spectral embedding.** The eigenvectors provide a **low-dimensional, smooth coordinate system** that reveals the underlying geometry of the data. This embedding maps similar data points close together and dissimilar ones far apart, making it ideal for:

1.  **Visual Concept Discovery**: Finding semantic objects in feature space.
2.  **Unsupervised Segmentation**: Grouping pixels without labels.
3.  **Data Visualization**: Projecting high-dimensional features into 3D for RGB visualization.

Originally introduced by [Shi & Malik (2000)](https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf), it became a foundational algorithm in computer vision for image segmentation. Today, with the rise of powerful feature extractors like DINO and SAM, Normalized Cut is experiencing a renaissance.

![summary](https://github.com/user-attachments/assets/a5d8a966-990b-4f6d-be10-abb00291bee2)

---

## The Concept: Graph Partitioning

Imagine an image not as a grid of pixels, but as a **graph**:

*   **Nodes**: Every pixel (or data point) is a node.
*   **Edges**: Connecting lines between nodes, weighted by how *similar* they are (e.g., color similarity, feature similarity).

The goal of Normalized Cut is to "cut" this graph into pieces (clusters) such that:
1.  **Minimize the cut**: The total weight of edges being cut is small (dissimilar regions are separated).
2.  **Maximize the association**: The total weight of edges remaining within each piece is large (similar regions stay together).
3.  **Balance**: The pieces are roughly balanced in size (avoiding tiny, isolated clusters).

Mathematically, this is solved by finding the **eigenvectors** of the graph's **Laplacian matrix**. These eigenvectors provide a continuous coordinate system (a "spectral embedding") where similar data points are mapped close together, making them easy to cluster with simple methods like K-Means.

For a deep dive into the math, see the [Methods > Basic Ncut](methods/01_basic_ncut.md) section.

---

## The Problem: Scalability

Traditional Spectral Clustering is powerful but computationally expensive.

*   **Time Complexity**: $O(N^2)$, where $N$ is the number of points.

*   **Memory Complexity**: $O(N^2)$ to store the similarity matrix.

For a single image (using DINO tokens or superpixels), $N \approx 4,000$, which is manageable.
For a batch of **250 images** (or a video sequence), $N \approx 1,000,000$. A standard $N \times N$ matrix would require **terabytes of RAM**, making it impossible to run.

---

## The Solution: Nyström Approximation

This library, **`ncut_pytorch`**, implements the **Nyström method** to approximate the spectral clustering solution efficiently.

*   **Time Complexity**: $O(N)$ (Linear!)
*   **Memory Complexity**: $O(1)$ (Constant!)

By sampling a small set of "landmark" points and computing their relations to the rest of the data, we can reconstruct the eigenvectors of the full graph with high accuracy. This allows us to perform spectral clustering on **millions of nodes** in **milliseconds** on a GPU.

For a deep dive into the Nyström methods, see the [Methods > Nyström Ncut](methods/02a_nystrom_ncut_complexity.md) section.

| Feature | Standard Spectral Clustering | Nyström Ncut (`ncut_pytorch`) |
| :--- | :--- | :--- |
| **Complexity** | $O(N^2)$ | **$O(N)$** |
| **Memory** | $O(N^2)$ | **$O(1)$** |
| **Scale** | < 50k nodes | **> 1M nodes** |
| **Hardware** | CPU/GPU | **GPU Accelerated** |

---

## Why use it now?

With modern Self-Supervised Learning (SSL) models like **DINO**, we have access to rich, semantic feature representations. Normalized Cut provides a **parameter-free** way to organize these features into coherent objects and segments without needing to train a specific segmentation head.

<div  style="text-align: center;">
<video width="90%" controls muted autoplay loop>
  <source src="../images/index/Ncut_video_sam_264_small.mp4" type="video/mp4">
</video>
</div>

---

## Where to go next?

*   **Try it out**: Install `pip install ncut-pytorch` and follow the [Quick Start](index.md#quick-start-ncut-with-dino-features).
*   **Understand the Math**: Read [Methods > Basic Ncut](methods/01_basic_ncut.md).
*   **See the Code**: Check out the [Tutorials](tutorials/index.md).

