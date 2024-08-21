
## NCUT: Nyström Normalized Cut

**Normalized Cut**, aka. spectral clustering, is a graphical method to analyze data grouping in the affinity eigenvector space. It has been widely used for unsupervised segmentation in the 2000s.

**Nyström Normalized Cut**, is a new approximation algorithm developed for large-scale graph cuts,  a large-graph of million nodes can be processed in under 10s (cpu) or 2s (gpu).  

## Gallery

Model features visualized by NCUT

<div style="text-align: center;">
<a href="./gallery/">
<img src="./images/ncut_gallery_cover.jpg" style="width:100%;">
</a>
</div>


## Installation

PyPI install, our package is based on [PyTorch](https://pytorch.org/get-started/locally/), presuming you already have PyTorch installed

```shell
pip install ncut-pytorch
```

[Install PyTorch](https://pytorch.org/get-started/locally/) by `pip` (for CPU only) or `conda` (recommended for GPU)
```shell
pip install torch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
## Why NCUT

Normalized cut offers two advantages:

1. soft-cluster assignments as eigenvectors

2. hierarchical clustering by varying the number of eigenvectors

Please see [NCUT and t-SNE/UMAP](compare.md) for a comparison over common PCA, t-SNE, UMAP.


> paper in prep, Yang 2024
>
> AlignedCut: Visual Concepts Discovery on Brain-Guided Universal Feature Space, Huzheng Yang, James Gee\*, Jianbo Shi\*, 2024
> 
> Normalized Cuts and Image Segmentation, Jianbo Shi and Jitendra Malik, 2000
> 

<div style="max-width: 600px; margin: 50px auto; border: 1px solid #ddd; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
    <a href="https://github.com/huzeyann/ncut_pytorch" target="_blank" style="text-decoration: none; color: inherit;">
        <div style="display: flex; align-items: center; padding: 15px; background-color: #f6f8fa;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub Logo" style="width: 50px; height: 50px; margin-right: 15px;">
            <div>
                <h2 style="margin: 0;">ncut-pytorch</h2>
                <p style="margin: 5px 0 0; color: #586069;">github.com/huzeyann/ncut_pytorch</p>
            </div>
        </div>
        <div style="padding: 15px; background-color: #fff;">
            <p style="margin: 0; color: #333;"></p>
        </div>
    </a>
</div>


## Table of Contents

- [Overview & Install](index.md)
- [Gallery](gallery.md)
- [How NCUT Works](how_ncut_works.md)
- [NCUT and t-SNE/UMAP](compare.md)
- [Tutorial 1 - Quick Start](tutorials.md)
- [Tutorial 2 - Parameters](parameters.md)
- [Tutorial 3 - Add Nodes](add_nodes.md)
- [Tutorial 4 - Mixing Data](mixing_data.md)
- [Tutorial 5 - Coloring](coloring.md)
- [How to Get Better Segmentation](how_to_get_better_segmentation.md)
- [Speed and Performance](speed_and_performance.md)
- [Gradient of NCUT](gradient_of_ncut.md)
- [API Reference](api_reference.md)