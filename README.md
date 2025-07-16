

<div style="text-align: center;">
  <img src="./docs/images/ncut.svg" alt="NCUT" style="width: 80%; filter: brightness(60%) grayscale(100%);"/>
</div>

### [🌐Documentation](https://ncut-pytorch.readthedocs.io/) | [🤗HuggingFace Demo](https://huggingface.co/spaces/huzey/ncut-pytorch)


## NCUT: Nyström Normalized Cut

**Normalized Cut**, aka. spectral clustering, is a graphical method to analyze data grouping in the affinity eigenvector space. It has been widely used for unsupervised segmentation in the 2000s.

**Nyström Normalized Cut**, is a new approximation algorithm developed for large-scale graph cuts,  a large-graph of million nodes can be processed in under 10s (cpu) or 2s (gpu).  



https://github.com/user-attachments/assets/f0d40b1f-b8a5-4077-ab5f-e405f3ffb70f



<div align="center">
  Video: NCUT applied to image encoder features from Segment Anything Model.
</div>


---

## Installation

#### 1. Install PyTorch

<div style="text-align:">
<pre><code class="language-shell">conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
</code></pre>
</div>

#### 2. Install `ncut-pytorch`

<div style="text-align:">
    <pre><code class="language-shell">pip install ncut-pytorch</code></pre>
</div>


#### Trouble Shooting

In case of `pip` install failed, please try install the build dependencies

Option A:
<div style="text-align:">
    <pre><code class="language-shell">sudo apt-get update && sudo apt-get install build-essential cargo rustc -y</code></pre>
</div>

Option B:
<div style="text-align:">
    <pre><code class="language-shell">conda install rust -c conda-forge</code></pre>
</div>

Option C:
<div style="text-align:">
    <pre><code class="language-shell">curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && . "$HOME/.cargo/env"</code></pre>
</div>

## Quick Start


Minimal example on how to run NCUT:

```py linenums="1"
import torch
from ncut_pytorch import NCUT, rgb_from_tsne_3d

model_features = torch.rand(20, 64, 64, 768)  # (B, H, W, C)

inp = model_features.reshape(-1, 768)  # flatten
eigvectors, eigvalues = NCUT(num_eig=100, device='cuda:0').fit_transform(inp)
tsne_x3d, tsne_rgb = rgb_from_tsne_3d(eigvectors, device='cuda:0')

eigvectors = eigvectors.reshape(20, 64, 64, 100)  # (B, H, W, num_eig)
tsne_rgb = tsne_rgb.reshape(20, 64, 64, 3)  # (B, H, W, 3)
```

#### Load Feature Extractor Model

Any backbone model works as plug-in feature extractor. 
We have implemented some backbone models, here is a list of available models:

```py
from ncut_pytorch.backbone import list_models
print(list_models())
[
  'SAM2(sam2_hiera_t)', 'SAM2(sam2_hiera_s)', 'SAM2(sam2_hiera_b+)', 'SAM2(sam2_hiera_l)', 
  'SAM(sam_vit_b)', 'SAM(sam_vit_l)', 'SAM(sam_vit_h)', 'MobileSAM(TinyViT)', 
  'DiNOv2reg(dinov2_vits14_reg)', 'DiNOv2reg(dinov2_vitb14_reg)', 'DiNOv2reg(dinov2_vitl14_reg)', 'DiNOv2reg(dinov2_vitg14_reg)', 
  'DiNOv2(dinov2_vits14)', 'DiNOv2(dinov2_vitb14)', 'DiNOv2(dinov2_vitl14)', 'DiNOv2(dinov2_vitg14)', 
  'DiNO(dino_vits8_896)', 'DiNO(dino_vitb8_896)', 'DiNO(dino_vits8_672)', 'DiNO(dino_vitb8_672)', 'DiNO(dino_vits8_448)', 'DiNO(dino_vitb8_448)', 'DiNO(dino_vits16_448)', 'DiNO(dino_vitb16_448)',
  'Diffusion(stabilityai/stable-diffusion-2)', 'Diffusion(CompVis/stable-diffusion-v1-4)', 'Diffusion(stabilityai/stable-diffusion-3-medium-diffusers)',
  'CLIP(ViT-B-16/openai)', 'CLIP(ViT-L-14/openai)', 'CLIP(ViT-H-14/openai)', 'CLIP(ViT-B-16/laion2b_s34b_b88k)', 
  'CLIP(convnext_base_w_320/laion_aesthetic_s13b_b82k)', 'CLIP(convnext_large_d_320/laion2b_s29b_b131k_ft_soup)', 'CLIP(convnext_xxlarge/laion2b_s34b_b82k_augreg_soup)', 
  'CLIP(eva02_base_patch14_448/mim_in22k_ft_in1k)', "CLIP(eva02_large_patch14_448/mim_m38m_ft_in22k_in1k)",
  'MAE(vit_base)', 'MAE(vit_large)', 'MAE(vit_huge)', 
  'ImageNet(vit_base)'
]
```

#### Image model example:

```py linenums="1"
import torch
from ncut_pytorch import NCUT, rgb_from_tsne_3d
from ncut_pytorch.backbone import load_model, extract_features

model = load_model(model_name="SAM(sam_vit_b)")
images = torch.rand(20, 3, 1024, 1024)
model_features = extract_features(images, model, node_type='attn', layer=6)
# model_features = model(images)['attn'][6]  # this also works

inp = model_features.reshape(-1, 768)  # flatten
eigvectors, eigvalues = NCUT(num_eig=100, device='cuda:0').fit_transform(inp)
tsne_x3d, tsne_rgb = rgb_from_tsne_3d(eigvectors, device='cuda:0')

eigvectors = eigvectors.reshape(20, 64, 64, 100)  # (B, H, W, num_eig)
tsne_rgb = tsne_rgb.reshape(20, 64, 64, 3)  # (B, H, W, 3)
```

#### Text model example:

```py linenums="1"
import os
from ncut_pytorch import NCUT, rgb_from_tsne_3d
from ncut_pytorch.backbone_text import load_text_model

os.environ['HF_ACCESS_TOKEN'] = "your_huggingface_token"
llama = load_text_model("meta-llama/Meta-Llama-3.1-8B").cuda()
output_dict = llama("The quick white fox jumps over the lazy cat.")

model_features = output_dict['block'][31].squeeze(0)  # 32nd block output
token_texts = output_dict['token_texts']
eigvectors, eigvalues = NCUT(num_eig=5, device='cuda:0').fit_transform(model_features)
tsne_x3d, tsne_rgb = rgb_from_tsne_3d(eigvectors, device='cuda:0')
# eigvectors.shape[0] == tsne_rgb.shape[0] == len(token_texts)
```

---

## Testing

The package includes a comprehensive test suite to ensure its functionality works as expected. The tests are located in the `unit_tests` directory and are written using the pytest framework.

### Running the Tests

To run the tests, first install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

Then run the tests using pytest:

```bash
pytest
```

For more information about the tests, see the [unit_tests/README.md](unit_tests/README.md) file.

---

> paper in prep, Yang 2024
>
> AlignedCut: Visual Concepts Discovery on Brain-Guided Universal Feature Space, Huzheng Yang, James Gee\*, Jianbo Shi\*,2024
> 
> Normalized Cuts and Image Segmentation, Jianbo Shi and Jitendra Malik, 2000
