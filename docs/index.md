

<div style="text-align: center;">
  <img src="./images/ncut.svg" alt="NCUT" style="width: 80%; filter: brightness(60%) grayscale(100%);"/>
</div>


<div style="display: flex; justify-content: center; margin-top: 20px;">

<a href="https://github.com/huzeyann/ncut_pytorch" target="_blank" style="width: 30%; text-align: center; background-color: #007BFF; color: white; padding: 10px; border-radius: 5px; margin-right: 5%;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub" style="width: 24px; height: 24px; vertical-align: middle;"/> GitHub
</a>

<a href="https://huggingface.co/spaces/huzey/ncut-pytorch" target="_blank" style="width: 30%; text-align: center; background-color: #FF5733; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
🤗 HuggingFace Demo
</a>

</div>

```shell
pip install ncut-pytorch
```


# NCUT: Nyström Normalized Cut

**Normalized Cut**, aka. spectral clustering, is a graphical method to analyze data grouping in the affinity eigenvector space. It has been widely used for unsupervised segmentation in the 2000s.

**Nyström Normalized Cut**, is a new approximation algorithm developed for large-scale graph cuts,  a large-graph of million nodes can be processed in under 10s (cpu) or 2s (gpu).  

<div  style="text-align: center;">
<video width="90%" controls muted autoplay loop>
  <source src="./images/ncut_video_sam_264_small.mp4" type="video/mp4">
</video>
<p>Video: NCUT applied to image encoder features from Segment Anything Model.
<a href="./gallery_sam_video">code</a>
</p>
</div>




[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gllutIdACcEHtJ81n_tGVNgR6fTupV46) Interactive heatmap plot, ready-to-use for pseudo labeling.

<div  style="text-align: center;">
<video width="90%" controls muted autoplay loop>
  <source src="./images/demo_heatmap.mp4" type="video/mp4">
</video>
</div>


Please visit our <a href="https://huggingface.co/spaces/huzey/ncut-pytorch" target="_blank">🤗HuggingFace Demo</a>
. Upload your images and get NCUT output. Play around backbone models and parameters.

<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/4.42.0/gradio.js"
></script>

<gradio-app src="https://huzey-ncut-pytorch.hf.space"></gradio-app>

<!-- <iframe
	src="https://huzey-ncut-pytorch.hf.space"
	frameborder="0"
	width="100%"
	height="800"
></iframe> -->

## Gallery
Just plugin features extracted from any pre-trained model and ready to go. NCUT works for any input -- image, text, video, 3D, .... Planty examples code and plots in the [Gallery](gallery.md)

<div style="text-align: center;">
<a href="./gallery/">
<img src="./images/ncut_gallery_cover.jpg" style="width:100%;">
</a>
</div>

---

## Installation & Quick Start

PyPI install, our package is based on PyTorch, please [install PyTorch](https://pytorch.org/get-started/locally/) first

```shell
pip install ncut-pytorch
```


<details>
<summary>

How to install PyTorch (click to expand):

</summary>

Install PyTorch by pip (for CPU only) or conda (for GPU)

``` shell
# for cpu only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# for gpu
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

</details>

Minimal example on how to run NCUT, more examples in [Tutorial](tutorials.md) and [Gallery](gallery.md).

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

## Why NCUT

Normalized cut offers two advantages:

1. soft-cluster assignments as eigenvectors

2. hierarchical clustering by varying the number of eigenvectors

Please see [NCUT and t-SNE/UMAP](compare.md) for a comparison over common PCA, t-SNE, UMAP.


---

> paper in prep, Yang 2024
>
> AlignedCut: Visual Concepts Discovery on Brain-Guided Universal Feature Space, Huzheng Yang, James Gee\*, Jianbo Shi\*,2024
> 
> Normalized Cuts and Image Segmentation, Jianbo Shi and Jitendra Malik, 2000



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

---

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
- [Memory Usage](memory_usage.md)
- [Speed and Performance](speed_and_performance.md)
- [Gradient of NCUT](gradient_of_ncut.md)
- [API Reference](api_reference.md)

---