

<div style="text-align: center;">
  <img src="./docs/images/ncut.svg" alt="NCUT" style="width: 80%; filter: brightness(60%) grayscale(100%);"/>
</div>

### [🌐Documentation](https://ncut-pytorch.readthedocs.io/) | [🤗HuggingFace Demo](https://huggingface.co/spaces/huzey/ncut-pytorch)


## Nyström Normalized Cut

Normalized Cut and spectral embedding, 100x faster than sklean implementation, linear complexity.


https://github.com/user-attachments/assets/f0d40b1f-b8a5-4077-ab5f-e405f3ffb70f



<div align="center">
  Video: Ncut spectral embedding eigenvectors, on SAM features.
</div>


---

## Installation

<div style="text-align:">
    <pre><code class="language-shell">pip install ncut-pytorch</code></pre>
</div>


## Quick Start: plain Ncut


```py linenums="1"
import torch
from ncut_pytorch import Ncut

features = torch.rand(1960, 768)
eigvecs = Ncut(features, n_eig=20)  # (1960, 20)

from ncut_pytorch import kway_ncut
kway_eigvecs = kway_ncut(eigvecs)
cluster_assignment = kway_eigvecs.argmax(1)
cluster_centroids = kway_eigvecs.argmax(0)
```

## Quick Start: Ncut DINOv3 Predictor

```py linenums="1"
from ncut_pytorch.predictor import NcutDinov3Predictor
from PIL import Image

predictor = NcutDinov3Predictor(model_cfg="dinov3_vitl16")
predictor = predictor.to('cuda')

images = [Image.open(f"images/view_{i}.jpg") for i in range(4)]
predictor.set_images(images)

image = predictor.summary(n_segments=[10, 25, 50, 100], draw_border=True)

```



>
> AlignedCut: Visual Concepts Discovery on Brain-Guided Universal Feature Space, Huzheng Yang, James Gee\*, Jianbo Shi\*,2024
> 
> Normalized Cuts and Image Segmentation, Jianbo Shi and Jitendra Malik, 2000
