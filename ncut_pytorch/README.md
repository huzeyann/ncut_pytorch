


Documentation [https://ncut-pytorch.readthedocs.io/](https://ncut-pytorch.readthedocs.io/)


## NCUT: Nyström Normalized Cut

**Normalized Cut**, aka. spectral clustering, is a graphical method to analyze data grouping in the affinity eigenvector space. It has been widely used for unsupervised segmentation in the 2000s.

**Nyström Normalized Cut**, is a new approximation algorithm developed for large-scale graph cuts,  a large-graph of million nodes can be processed in under 10s (cpu) or 2s (gpu).  

## Gallery
TODO

## Installation

PyPI install, our package is based on [PyTorch](https://pytorch.org/get-started/locally/), presuming you already have PyTorch installed

```shell
pip install ncut-pytorch
```

[Install PyTorch](https://pytorch.org/get-started/locally/) if you haven't
```shell
pip install torch
```
## Why NCUT

Normalized cut offers two advantages:

1. soft-cluster assignments as eigenvectors

2. hierarchical clustering by varying the number of eigenvectors

Please see [NCUT and t-SNE/UMAP](compare.md) for a full comparison.


> paper in prep, Yang 2024
>
> AlignedCut: Visual Concepts Discovery on Brain-Guided Universal Feature Space, Huzheng Yang, James Gee\*, Jianbo Shi\*, 2024
> 
> Normalized Cuts and Image Segmentation, Jianbo Shi and Jitendra Malik, 2000
> 
