<div style="text-align: center;">
  <img src="../docs/images/ncut.svg" alt="NCUT" style="width: 80%; filter: brightness(60%) grayscale(100%);"/>
</div>

### [🌐Documentation](https://ncut-pytorch.readthedocs.io/) | [🤗HuggingFace Demo](https://huggingface.co/spaces/huzey/ncut-pytorch)


## NCUT: Nyström Normalized Cut

**Normalized Cut**, aka. spectral clustering, is a graphical method to analyze data grouping in the affinity eigenvector space. It has been widely used for unsupervised segmentation in the 2000s.

**Nyström Normalized Cut**, is a new approximation algorithm developed for large-scale graph cuts,  a large-graph of million nodes can be processed in under 10s (cpu) or 2s (gpu).  


---

## Installation & Quick Start

PyPI install, our package is based on PyTorch, please [install PyTorch](https://pytorch.org/get-started/locally/) first

```shell
pip install ncut-pytorch
```

Minimal example on how to run NCUT, more examples in [Documentation](https://ncut-pytorch.readthedocs.io/).

```py linenums="1"
import torch
from ncut_pytorch import NCUT, rgb_from_tsne_3d

model_features = torch.rand(20, 64, 64, 768)

inp = model_features.reshape(-1, 768)  # flatten
eigvectors, eigvalues = NCUT(num_eig=100, device='cuda:0').fit_transform(inp)
tsne_x3d, tsne_rgb = rgb_from_tsne_3d(eigvectors, device='cuda:0')

eigvectors = eigvectors.reshape(20, 64, 64, 100)
tsne_rgb = tsne_rgb.reshape(20, 64, 64, 3)
```

---

> paper in prep, Yang 2024
>
> AlignedCut: Visual Concepts Discovery on Brain-Guided Universal Feature Space, Huzheng Yang, James Gee\*, Jianbo Shi\*,2024
> 
> Normalized Cuts and Image Segmentation, Jianbo Shi and Jitendra Malik, 2000
