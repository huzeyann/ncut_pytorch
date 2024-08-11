
## NCUT: Nystrom Normalized Cut

**Normalized cut**, aka. spectral clustering, is a graphical method to analyze data grouping in the affinity eigenvector space. It's used for unsupervised segmentation, without any model training. 

**Nystrom Normalized Cut**, is an approximation algorithm developed for large-scale graph cut,  a large-graph of million nodes can be processed in under 10s (cpu) or 2s (gpu).  


## Gallery

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

Please see [NCUT and t-SNE/UMAP](/compare/) for a full comparison.


> Normalized Cuts and Image Segmentation, Shi 2000
> 
> in prep, Yang 2024