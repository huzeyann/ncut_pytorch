# Channel Tracing

Identify which feature channels contribute most to a given segment produced by k-way Normalized Cuts. We adopt a simple, Grad-CAM–style gradient attribution over feature channels to rank their importance and visualize them.

The examples of the results of channel tracing:

<div class="ct-tabs" style="text-align:center;">
  <input type="radio" id="ct-ex1" name="ct" checked>
  <label for="ct-ex1" class="ctbtn">Example 1</label>
  <input type="radio" id="ct-ex2" name="ct">
  <label for="ct-ex2" class="ctbtn">Example 2</label>
  <input type="radio" id="ct-ex3" name="ct">
  <label for="ct-ex3" class="ctbtn">Example 3</label>

<div class="ct-img ct-img-ex1">
  <img src="../images/grad_exmaple_1.png" alt="Channel tracing - Example 1" style="width:100%; height:auto; display:block; margin:0 auto;" />
</div>

<div class="ct-img ct-img-ex2">
  <img src="../images/grad_exmaple_2.png" alt="Channel tracing - Example 2" style="width:100%; height:auto; display:block; margin:0 auto;" />
</div>

<div class="ct-img ct-img-ex3">
  <img src="../images/grad_exmaple_3.png" alt="Channel tracing - Example 3" style="width:100%; height:auto; display:block; margin:0 auto;" />
</div>

</div>
<style>
.ct-tabs input[type="radio"]{display:none;}
.ct-tabs .ct-img{display:none;}
#ct-ex1:checked ~ .ct-img-ex1{display:block;}
#ct-ex2:checked ~ .ct-img-ex2{display:block;}
#ct-ex3:checked ~ .ct-img-ex3{display:block;}
.ctbtn{display:inline-block; padding:6px 12px; border:1px solid var(--md-default-fg-color--lighter, #ccc); border-radius:6px; margin:0 4px; cursor:pointer;}
#ct-ex1:checked + label.ctbtn, #ct-ex2:checked + label.ctbtn, #ct-ex3:checked + label.ctbtn{background: var(--md-primary-fg-color, #3f51b5); color:#fff; border-color: transparent;}
.ct-img img{max-width:100%; height:auto; display:block; margin:8px auto;}
</style>


## Brief Explanations

Assume you already have differentiable soft cluster scores for every pixel (or patch) from k-way N-cut. To trace which feature channels are most responsible for a target cluster:

1. Pick a target cluster and build a hard region mask by assigning each pixel to the cluster with the highest soft score. 
2. Define a single scalar objective that becomes larger when those masked pixels are scored more confidently as the target cluster; a simple choice is the average absolute cluster score over the masked pixels. 
3. Backpropagate this objective to the input features so that every pixel-channel gets a gradient indicating how increasing that channel would increase the cluster confidence on its own region. 
4. Aggregate the gradients within the masked region by averaging across pixels, resulting in one value per channel. 
5. Rank channels by the magnitude of these aggregated gradients. The largest values indicate channels that most strongly support the target cluster on its region. Select the top-1 (or top-k) as the most influential. 

Practical notes: ensure the feature tensor requires gradients and that the N-cut pipeline is configured to propagate gradients. Clear feature gradients between clusters, and retain the computation graph if you need to compute attributions for multiple clusters from the same forward pass.

## Minimal Setup

Below is an end-to-end minimal flow. Bring your own image feature extractor (e.g., DINO). The only requirements are: a feature tensor with shape `(num_pixels, num_channels)` and enabling gradients on it.

```python
from einops import rearrange
import torch
from PIL import Image
from ncut_pytorch import Ncut, kway_ncut

# Example helpers (adapt to your own feature extractor)
def load_images_helper(pil_images_or_paths):
    if isinstance(pil_images_or_paths[0], str):
        pil_images = [Image.open(p) for p in pil_images_or_paths]
    else:
        pil_images = pil_images_or_paths
    pil_images = [im.convert("RGB") for im in pil_images]
    return pil_images

# Suppose you have: dino_img_transform(image) -> tensor, and extract_dino_image_embeds(batch) -> [B, L+1, C]
# Feel free to swap with your own extractor.
def prepare_grid_features(images, dino_img_transform, extract_dino_image_embeds):
    images_tensor = torch.stack([dino_img_transform(im) for im in images])
    dino_embeds = extract_dino_image_embeds(images_tensor)[:, 1:, :]  # drop cls token -> [B, L, C]
    b, l, c = dino_embeds.shape
    h = w = int(l ** 0.5)
    features = rearrange(dino_embeds, 'b l c -> (b l) c')  # [(B*H*W), C]
    return features, b, h, w
```

## Differentiable k-way N-cut

Make the feature tensor require gradients. Then run differentiable N-cut to get eigenvectors and their k-way discretization.

```python
def differentiable_kway_ncut(features, n_segment):
    assert features.requires_grad is True
    eigvec = Ncut(n_eig=n_segment, track_grad=True).fit_transform(features)
    kway_eigvec = kway_ncut(eigvec)  # shape: [(B*H*W), K]
    return eigvec, kway_eigvec

features = features.clone().requires_grad_(True)
eigvec, kway_eigvec = differentiable_kway_ncut(features, n_segment=4)
```

## Channel attribution per cluster

We define a cluster-specific objective by selecting the pixels belonging to that cluster (via argmax) and maximizing the average absolute cluster score. Then we backprop to features and aggregate gradients over pixels to obtain a per-channel importance score.

```python
def channel_gradient_from_cluster(features, cluster_mask, kway_eigvec, cluster_idx):
    """Return a length-C tensor: average gradient per channel for a given cluster."""
    assert features.requires_grad is True
    if features.grad is not None:
        features.grad.zero_()

    # Encourage larger magnitude of the selected cluster score
    loss = - kway_eigvec[cluster_mask, cluster_idx].abs().mean()
    loss.backward(retain_graph=True)

    grad = features.grad[cluster_mask].mean(0)  # [C]
    return grad  # higher magnitude => more influential channel
```

Typical usage pattern:

```python
# Derive hard assignments
cluster_assign = kway_eigvec.argmax(1)  # [(B*H*W)]

all_cluster_grads = []
for k in range(kway_eigvec.shape[1]):
    mask_k = (cluster_assign == k).detach().cpu()
    grad_k = channel_gradient_from_cluster(features, mask_k, kway_eigvec, k)
    all_cluster_grads.append(grad_k)
```

You can rank channels for cluster `k` by sorting `grad_k` by magnitude or by signed value depending on your interpretation.

## Visualization snippets

Below are lightweight helpers to visualize top channels per cluster. Adapt plotting to your needs.

```python
import numpy as np

def rank_channels(grad, descending=True):
    # By default, rank by signed value. Use grad.abs() if you prefer magnitude.
    return torch.argsort(grad, descending=descending)

def reshape_for_view(features, b, h, w):
    # -> [B, H, W, C] without gradients
    return features.detach().reshape(b, h, w, -1).cpu().numpy()

def extract_topk_maps(feature_grid, channel_indices, image_index=0, topk=5):
    # feature_grid: [B, H, W, C] ndarray
    chs = channel_indices[:topk].tolist()
    maps = [feature_grid[image_index, :, :, ch] for ch in chs]
    return chs, maps
```

