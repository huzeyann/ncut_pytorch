__all__ = ["farthest_point_sampling"]

import warnings

import fpsample
import numpy as np
import torch

from .device import auto_device
from .math import pca_lowrank

__IS_FAST_FPS_AVAILABLE = False
try:
    from fpsample import bucket_fps_kdline_sampling
    __IS_FAST_FPS_AVAILABLE = True
except ImportError:
    warnings.warn("""package `fpsample` is not installed with latest version. 
    Using a slower implementation (~100ms slower).
    To install fpsample, run:
    >>> pip install fpsample==0.3.3

    if the above pip install fails, please try install build dependencies (Rust) first:
    >>> sudo apt-get update
    >>> sudo apt-get install build-essential cargo rustc -y
    or 
    >>> conda install rust -c conda-forge""")


@torch.no_grad()
def farthest_point_sampling(
    X: torch.Tensor,
    n_sample: int,
    max_draw_ratio: float = 4.0,
    device: str = None,
):
    # if num_data is too large, use random sampling to reduce the load of farthest point sampling

    num_data = X.shape[0]

    num_draw = int(n_sample * max_draw_ratio)
    if num_draw > num_data:
        return _farthest_point_sampling(X, n_sample, device=device)

    # random draw num_draw samples to reduce the load of farthest point sampling
    draw_indices = np.random.permutation(num_data)[:num_draw]
    sampled_indices = _farthest_point_sampling(
        X[draw_indices],
        n_sample=n_sample,
        device=device,
    )
    return draw_indices[sampled_indices]


@torch.no_grad()
def _farthest_point_sampling(
    X: torch.Tensor,
    n_sample: int,
    h: int = 7,
    device: str = None,
):
    num_data = X.shape[0]
    if n_sample >= num_data:
        return np.arange(num_data)

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    
    device = auto_device(X.device, device)
    X = X.to(device)

    # PCA to reduce the dimension, because fpsample.bucket_fps_kdline_sampling only supports up to 8 dimensions
    if X.shape[1] > 8:
        X = pca_lowrank(X, q=8)

    assert X.ndim == 2, "X should be a 2D tensor"
    assert X.shape[0] > 0, "X should have at least 1 data point"
    assert X.shape[1] <= 8, "fpsample only supports up to 8 dimensions"
    assert X.shape[1] > 0, "X should have at least 1 dimension"
    assert not torch.any(torch.isnan(X)), "X contains NaN"
    assert not torch.any(torch.isinf(X)), "X contains Inf"
    
    if __IS_FAST_FPS_AVAILABLE:
        h = min(h, int(np.log2(num_data)))
        samples_idx = fpsample.bucket_fps_kdline_sampling(
            X.cpu().numpy(), n_sample, h
        )
    else:
        samples_idx = fpsample.fps_npdu_kdtree_sampling(
            X.cpu().numpy(), n_sample,
        )
    samples_idx = samples_idx.astype(np.int64)
    return samples_idx

