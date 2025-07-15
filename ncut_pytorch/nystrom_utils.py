import logging
from typing import Literal

import numpy as np
import torch

import functools

from .math_utils import pca_lowrank
from .math_utils import get_affinity

import fpsample


# internal configuration for nystrom approximation, can be overridden by kwargs
# values are optimized based on empirical experiments, no need to change the values
_NYSTROM_CONFIG = {
    'n_sample': 10240,  # number of samples for nystrom approximation, 10240 is large enough for most cases
    'n_sample2': 1024,  # number of samples for eigenvector propagation, 1024 is large enough for most cases
    'n_neighbors': 10,  # number of neighbors for eigenvector propagation, 10 is large enough for most cases
    'matmul_chunk_size': 16384,  # chunk size for matrix multiplication, larger chunk size is faster but requires more memory
    'sample_method': "farthest",  # sample method for nystrom approximation, 'farthest' is FPS(Farthest Point Sampling)
    'move_output_to_cpu': True,  # if True, will move output to cpu, which saves memory but loses gradients
}


def auto_divice(feature_device = "cuda:0", user_input_device = None):
    if str(user_input_device) == "cpu":
        return "cpu"
    is_cuda_available = torch.cuda.is_available()
    if not is_cuda_available:
        return "cpu"
    if is_cuda_available:
        if "cuda" in str(feature_device):
            # cuda:1, cuda:2, etc.
            return feature_device
        else:
            return "cuda:0"

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
    if n_sample > num_data:
        return np.arange(num_data)

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = X.to(device)

    # PCA to reduce the dimension, because fpsample.bucket_fps_kdline_sampling only supports up to 8 dimensions
    if X.shape[1] > 8:
        X = pca_lowrank(X, q=8)

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
        Using a old and slower implementation.
        ---
        To install fpsample, run:
        
        >>> pip install fpsample==0.3.3
        
        if the above pip install fails, try install build dependencies first:
        >>> sudo apt-get update && sudo apt-get install build-essential cargo rustc -y
        or 
        >>> conda install rust -c conda-forge

        see https://ncut-pytorch.readthedocs.io/en/latest/trouble_shooting/ for more help
        ---
        ---
        """
        logger = logging.getLogger("ncut_pytorch")
        logger.warning(message)
        return False


def nystrom_propagate(
    nystrom_out: torch.Tensor,
    X: torch.Tensor,
    nystrom_X: torch.Tensor,
    n_neighbors: int = 10,
    gamma: float = 1.0,
    n_sample: int = 1024,
    chunk_size: int = 16384,
    device: str = None,
    move_output_to_cpu: bool = False,
    track_grad: bool = False,
    **kwargs,
):
    """A generic function to propagate new nodes using KNN.
    nystrom_out is propagated to fullgraph, using KNN look up from nystrom_X to X.

    Args:
        nystrom_out (torch.Tensor): output from subgraph, shape (num_sample, D)
        X (torch.Tensor): features from existing nodes, shape (new_num_samples, n_features)
        nystrom_X (torch.Tensor): features from subgraph, shape (num_sample, n_features)
        knn (int): number of KNN to propagate eigenvectors
        chunk_size (int): chunk size for matrix multiplication
        device (str): device to use for computation, if None, will not change device
        move_output_to_cpu (bool): move output to cpu to save GPU memory
        track_grad (bool): keep track of pytorch gradients, default False

    Returns:
        torch.Tensor: propagated eigenvectors, shape (new_num_samples, D)

    Examples:
        >>> old_eigenvectors = torch.randn(3000, 20)
        >>> old_features = torch.randn(3000, 100)
        >>> new_features = torch.randn(200, 100)
        >>> new_eigenvectors = propagate_knn(old_eigenvectors, new_features, old_features, knn=3)
        >>> # new_eigenvectors.shape = (200, 20)

    """

    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(track_grad)

    nystrom_indices = farthest_point_sampling(nystrom_out, n_sample)
    nystrom_out = nystrom_out[nystrom_indices]
    nystrom_X = nystrom_X[nystrom_indices]
    
    device = auto_divice(nystrom_out.device, device)

    nystrom_out = nystrom_out.to(device)
    nystrom_X = nystrom_X.to(device)

    # eigvec of each data point is a weighted sum of the nystrom_out eigvecs
    # the weighted sum is only on the topk nearest neighbors of the data point
    all_outs = []
    for i in range(0, X.shape[0], chunk_size):
        end = min(i + chunk_size, X.shape[0])

        _Xi = X[i:end].to(device)

        # compute affinity matrix from each chunk of data points to the nystrom sampled nodes
        _Ai = get_affinity(_Xi, nystrom_X, gamma=gamma)

        # keep topk nearest neighbors for each row (sampled nodes)
        topk_A, topk_indices = _Ai.topk(k=n_neighbors, dim=-1, largest=True)
        # normalize the topk neighbors affinity to sum to 1 on each row
        _D = topk_A.sum(-1)
        topk_A = topk_A / _D[:, None]
        _weights = topk_A.flatten()  # (n * n_neighbors)
        
        # for each data point (row), it's output eigvec is a weighted sum of 
        # the topk neighbors eigvecs, each row of affinity matrix is normalized to sum to 1
        _values = nystrom_out[topk_indices.flatten()]  # (n * n_neighbors, d)
        topk_output = _values * _weights[:, None]  # (n * n_neighbors, d)
        topk_output = topk_output.reshape(-1, n_neighbors, _values.shape[-1])  # (n, n_neighbors, d)
        topk_output = topk_output.sum(dim=1)  # (n, d)

        if move_output_to_cpu and not track_grad:  # move output to cpu to save GPU memory
            topk_output = topk_output.cpu()
        all_outs.append(topk_output)

    all_outs = torch.cat(all_outs, dim=0)

    torch.set_grad_enabled(prev_grad_state)

    return all_outs
