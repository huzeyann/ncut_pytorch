__all__ = ["farthest_point_sampling"]

import numpy as np
import torch

from .device import auto_device
from .math import pca_lowrank

from torch_quickfps import sample_idx


@torch.no_grad()
def farthest_point_sampling(
    X: torch.Tensor,          # [N,D]
    n_sample: int,
    max_draw_ratio: float = 4.0,
    device: str | None = None,
) -> np.ndarray:             # [n_sample]
    """Farthest point sampling with optional random pre-sampling for large datasets."""
    num_data = X.shape[0]
    num_draw = int(n_sample * max_draw_ratio)
    
    if num_draw > num_data:
        return _farthest_point_sampling(X, n_sample, device=device)
    
    # Randomly pre-sample to reduce computation
    draw_indices = torch.randperm(num_data)[:num_draw]
    sampled_indices = _farthest_point_sampling(
        X[draw_indices],
        n_sample=n_sample,
        device=device,
    )
    return draw_indices[sampled_indices]


@torch.no_grad()
def _farthest_point_sampling(
    X: torch.Tensor,          # [N,D]
    n_sample: int,
    device: str | None = None,
) -> np.ndarray:             # [n_sample]
    """Internal farthest point sampling implementation using torch-quickfps."""
    num_data = X.shape[0]
    if n_sample >= num_data:
        return np.arange(num_data)

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    
    device = auto_device(X.device, device)
    X = X.to(device)

    # PCA to reduce dimension 
    if X.shape[1] > 8:
        X = pca_lowrank(X, q=8)

    assert X.ndim == 2, "X should be a 2D tensor"
    assert X.shape[0] > 0, "X should have at least 1 data point"
    assert X.shape[1] > 0, "X should have at least 1 dimension"
    assert not torch.any(torch.isnan(X)), "X contains NaN"
    assert not torch.any(torch.isinf(X)), "X contains Inf"
    
    samples_idx = sample_idx(X.cpu(), n_sample).numpy()
    return samples_idx
