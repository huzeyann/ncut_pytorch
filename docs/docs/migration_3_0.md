# Migration to 3.0.0

Most of this documentation was originally written against the late 2.x API. The core workflow is still the same in 3.0.0, but several public symbols, parameter names, and example patterns changed.

## High-Impact API Changes

| 2.x / legacy docs | 3.0.0 |
| --- | --- |
| `NCUT(num_eig=...)` | `Ncut(n_eig=...)` |
| `num_eig` | `n_eig` |
| `eigvecs, eigvals = model.fit_transform(X)` | `eigvecs = model.fit_transform(X)` and `eigvals = model.eigval` |
| `d_gamma`, `gamma`, `repulsion_gamma` | `quantile_sigma`, `sigma`, `repulsion_sigma` |
| `track_grad=True` | `exact_gradient=True` when you need the gradient-stable eigensolver path |
| `rgb_from_tsne_3d(eigvecs, ...)` | `tsne_color(eigvecs, ...)` |
| top-level color helper imports | `from ncut_pytorch.color import ...` |
| `ncut_pytorch.utils.gamma.find_gamma_by_degree` | `ncut_pytorch.utils.sigma.find_sigma_by_degree` |
| `ncut-pytorch[torch]` | install `ncut-pytorch`, plus PyTorch and optional extras separately |

## Installation Changes

The `torch` extra used by older docs is gone in 3.0.0.

- Base install: `pip install -U ncut-pytorch`
- If your environment does not already provide PyTorch, install `torch` separately.
- Predictor examples also need `torchvision`.
- The optional Lightning-based M-space trainer uses `pytorch-lightning~=2.0`.

## Example Rewrite Patterns

### Plain Ncut

```python
from ncut_pytorch import Ncut, kway_ncut
from ncut_pytorch.color import tsne_color

ncut = Ncut(n_eig=20, n_sample=10000, n_neighbors=10, device="cuda:0")
eigvecs = ncut.fit_transform(features)
eigvals = ncut.eigval

rgb = tsne_color(eigvecs[:, :20], device="cuda:0")
kway = kway_ncut(eigvecs[:, :10])
labels = kway.argmax(dim=1)
```

### Gradient-Sensitive Usage

```python
from ncut_pytorch import Ncut

ncut = Ncut(n_eig=20, exact_gradient=True)
eigvecs = ncut.fit_transform(features.requires_grad_(True))
loss = eigvecs.sum()
loss.backward()
```

## Pages That Still Need Full Manual Migration

These older notebook-style pages contain heavier pre-3.0 examples and should either be rewritten or marked as legacy when touched next:

- `docs/docs/tutorials/07_parameters.md`
- `docs/docs/tutorials/10_application_segmentation.md`
- `docs/docs/usage_examples/gallery/gallery_clip.md`
- `docs/docs/usage_examples/gallery/gallery_dinov2.md`
- `docs/docs/usage_examples/gallery/gallery_dinov2_video.md`
- `docs/docs/usage_examples/gallery/gallery_gpt2.md`
- `docs/docs/usage_examples/gallery/gallery_llama3.md`
- `docs/docs/usage_examples/gallery/gallery_mae.md`
- `docs/docs/usage_examples/gallery/gallery_sam.md`
- `docs/docs/usage_examples/gallery/gallery_sam2_video.md`
- `docs/docs/usage_examples/gallery/gallery_sam_video.md`

The common migration work in those files is the same: replace `NCUT` with `Ncut`, stop unpacking `fit_transform`, replace `rgb_from_tsne_3d`, and rewrite any use of removed helpers such as `ncut_pytorch.backbone` or `get_mask`.
