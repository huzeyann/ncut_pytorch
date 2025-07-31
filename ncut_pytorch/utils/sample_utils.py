import warnings
from typing import Literal

import fpsample
import numpy as np
import torch

from .device import auto_device
from .math_utils import pca_lowrank


@torch.no_grad()
def run_subgraph_sampling(
    X: torch.Tensor,
    n_sample: int,
    sample_method: Literal["farthest", "random"] = "farthest",
    max_draw_ratio: float = 4.0,
    device: str = None,
):
    if n_sample >= X.shape[0]:
        return torch.arange(X.shape[0])

    if sample_method == "farthest": 
        sampled_indices = farthest_point_sampling(
            X, n_sample, max_draw_ratio, device
        )
    elif sample_method == "random": 
        sampled_indices = torch.randperm(X.shape[0])[:n_sample]
    else:
        raise ValueError("sample_method should be 'farthest' or 'random'")
    return sampled_indices


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
    draw_indices = torch.randperm(num_data)[:num_draw]
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
    
    if _is_fast_fps_available():
        h = min(h, int(np.log2(num_data)))
        samples_idx = fpsample.bucket_fps_kdline_sampling(
            X.cpu().numpy(), n_sample, h
        ).astype(np.int64)
    else:
        samples_idx = fpsample.fps_npdu_kdtree_sampling(
            X.cpu().numpy(), n_sample,
        ).astype(np.int64)

    return samples_idx


def _is_fast_fps_available():
    # a fallback implementation of fpsample is provided for users who cannot install fpsample
    # but the performance is much slower
    try:
        from fpsample import bucket_fps_kdline_sampling
        return True
    except ImportError:
        message = """
        ---
        Farthest Point Sampling (fpsample>=0.3.3) installation Not Found. 
        Using an old and slower implementation.
        ---
        To install fpsample, run:
        
        >>> pip install fpsample==0.3.3
        
        if the above pip install fails, please try install build dependencies (Rust) first:
        >>> sudo apt-get update && sudo apt-get install build-essential cargo rustc -y
        or 
        >>> conda install rust -c conda-forge

        see https://ncut-pytorch.readthedocs.io/en/latest/trouble_shooting/ for more help
        ---
        ---
        """
        warnings.warn(message, stacklevel=2, category=ImportWarning)
        return False


