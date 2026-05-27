__all__ = ["farthest_point_sampling"]

import torch

from .device import auto_device
from .math import pca_lowrank

from torch_quickfps import sample_idx

_op = "torch_quickfps::_sample_idx_impl"
_HAS_CUDA_KERNEL = torch._C._dispatch_has_kernel_for_dispatch_key(_op, "CUDA")
_DEFAULT_MAX_DRAW_RATIO = 2.0


def _stratified_presample_indices(
    num_data: int,
    num_draw: int,
) -> torch.Tensor:
    """Draw evenly spread candidate indices without materializing randperm(num_data)."""
    if num_draw >= num_data:
        return torch.arange(num_data)

    step = num_data / num_draw
    offset = torch.rand((), dtype=torch.float64) * step
    base = torch.arange(num_draw, dtype=torch.float64)
    draw_indices = torch.floor(offset + base * step).to(torch.long)
    draw_indices.clamp_(max=num_data - 1)
    return draw_indices


@torch.no_grad()
def farthest_point_sampling(
    X: torch.Tensor,          # [N,D]
    n_sample: int,
    max_draw_ratio: float = _DEFAULT_MAX_DRAW_RATIO,
    max_dim: int = 8,
    device: str | None = None,
) -> torch.Tensor:             # [n_sample]
    """Farthest point sampling with optional stratified pre-sampling for large datasets."""
    num_data = X.shape[0]
    num_draw = min(num_data, max(n_sample, int(n_sample * max_draw_ratio)))

    if num_draw >= num_data:
        return _farthest_point_sampling(X, n_sample, device=device, max_dim=max_dim)

    draw_indices = _stratified_presample_indices(num_data, num_draw)
    subset_indices = draw_indices if X.device.type == "cpu" else draw_indices.to(X.device)
    sampled_indices = _farthest_point_sampling(
        X.index_select(0, subset_indices),
        n_sample=n_sample,
        max_dim=max_dim,
        device=device,
    )
    return draw_indices[sampled_indices.cpu()]


@torch.no_grad()
def _farthest_point_sampling(
    X: torch.Tensor,          # [N,D]
    n_sample: int,
    max_dim: int = 8,
    device: str | None = None,
) -> torch.Tensor:             # [n_sample]
    """Internal farthest point sampling implementation using torch-quickfps."""
    num_data = X.shape[0]
    if n_sample >= num_data:
        return torch.arange(num_data)

    target_device = "cpu" if not _HAS_CUDA_KERNEL else auto_device(X.device, device)
    X = X.to(target_device)

    # PCA to reduce dimension 
    if X.shape[1] > max_dim:
        X = pca_lowrank(X, q=max_dim)

    assert X.ndim == 2, "X should be a 2D tensor"
    assert X.shape[0] > 0, "X should have at least 1 data point"
    assert X.shape[1] > 0, "X should have at least 1 dimension"
    assert not torch.any(torch.isnan(X)), "X contains NaN"
    assert not torch.any(torch.isinf(X)), "X contains Inf"

    samples_idx = sample_idx(X, n_sample)

    return samples_idx.cpu()
