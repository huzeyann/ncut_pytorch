import logging
from typing import Literal

import numpy as np
import torch

from .math_utils import pca_lowrank
from .math_utils import affinity_from_features

import fpsample


def which_device(feature_device = "cuda:0", user_input_device = None):
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
    features: torch.Tensor,
    num_sample: int,
    sample_method: Literal["farthest", "random"] = "farthest",
    max_draw_ratio: float = 4.0,
    device: str = None,
):
    if num_sample >= features.shape[0]:
        return torch.arange(features.shape[0])

    if sample_method == "farthest": 
        sampled_indices = farthest_point_sampling(
            features, num_sample, max_draw_ratio, device
        )
    elif sample_method == "random": 
        sampled_indices = torch.randperm(features.shape[0])[:num_sample]
    else:
        raise ValueError("sample_method should be 'farthest' or 'random'")
    return sampled_indices


@torch.no_grad()
def farthest_point_sampling(
    features: torch.Tensor,
    num_sample: int,
    max_draw_ratio: float = 4.0,
    device: str = None,
):
    # if num_data is too large, use random sampling to reduce the load of farthest point sampling

    num_data = features.shape[0]

    num_draw = int(num_sample * max_draw_ratio)
    if num_draw > num_data:
        return _farthest_point_sampling(features, num_sample, device=device)

    # random draw num_draw samples to reduce the load of farthest point sampling
    draw_indices = torch.randperm(num_data)[:num_draw]
    sampled_indices = _farthest_point_sampling(
        features[draw_indices],
        num_sample=num_sample,
        device=device,
    )
    return draw_indices[sampled_indices]


@torch.no_grad()
def _farthest_point_sampling(
    features: torch.Tensor,
    num_sample: int,
    h: int = 7,
    device: str = None,
):
    num_data = features.shape[0]
    if num_sample > num_data:
        return np.arange(num_data)

    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    features = features.to(device)

    # PCA to reduce the dimension, because fpsample.bucket_fps_kdline_sampling only supports up to 8 dimensions
    if features.shape[1] > 8:
        features = pca_lowrank(features, q=8)

    if _is_fast_fps_available():
        h = min(h, int(np.log2(num_data)))
        samples_idx = fpsample.bucket_fps_kdline_sampling(
            features.cpu().numpy(), num_sample, h
        ).astype(np.int64)
    else:
        samples_idx = fpsample.fps_npdu_kdtree_sampling(
            features.cpu().numpy(), num_sample,
        ).astype(np.int64)

    return samples_idx


def _is_fast_fps_available():
    
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


def propagate_knn(
    subgraph_output: torch.Tensor,
    fullgraph_features: torch.Tensor,
    subgraph_features: torch.Tensor,
    knn: int = 10,
    affinity_focal_gamma: float = 1.0,
    distance: Literal["cosine", "euclidean", "rbf"] = "rbf",
    num_sample: int = 1024,
    chunk_size: int = 16384,
    device: str = None,
    move_output_to_cpu: bool = False,
    **kwargs,
):
    """A generic function to propagate new nodes using KNN.
    subgraph_output is propagated to fullgraph, using KNN look up from subgraph_features to fullgraph_features.

    Args:
        subgraph_output (torch.Tensor): output from subgraph, shape (num_sample, D)
        fullgraph_features (torch.Tensor): features from existing nodes, shape (new_num_samples, n_features)
        subgraph_features (torch.Tensor): features from subgraph, shape (num_sample, n_features)
        knn (int): number of KNN to propagate eigenvectors
        chunk_size (int): chunk size for matrix multiplication
        device (str): device to use for computation, if None, will not change device
        move_output_to_cpu (bool): move output to cpu to save GPU memory

    Returns:
        torch.Tensor: propagated eigenvectors, shape (new_num_samples, D)

    Examples:
        >>> old_eigenvectors = torch.randn(3000, 20)
        >>> old_features = torch.randn(3000, 100)
        >>> new_features = torch.randn(200, 100)
        >>> new_eigenvectors = propagate_knn(old_eigenvectors, new_features, old_features, knn=3)
        >>> # new_eigenvectors.shape = (200, 20)

    """

    # sub-sample, for speed up
    sample_idx = farthest_point_sampling(subgraph_output, num_sample)
    subgraph_output = subgraph_output[sample_idx]
    subgraph_features = subgraph_features[sample_idx]
    
    device = which_device(subgraph_output.device, device)
    subgraph_output = subgraph_output.to(device)
    subgraph_features = subgraph_features.to(device)

    fullgraph_outputs = []
    for i in range(0, fullgraph_features.shape[0], chunk_size):
        end = min(i + chunk_size, fullgraph_features.shape[0])

        with torch.no_grad():
            _v = fullgraph_features[i:end].to(device)
            _A = affinity_from_features(_v, subgraph_features, distance=distance, affinity_focal_gamma=affinity_focal_gamma)

            # keep topk nearest neighbors for each row (sampled nodes)
            topk_A, topk_indices = _A.topk(k=knn, dim=-1, largest=True)
            _D = topk_A.sum(-1)
            topk_A = topk_A / _D[:, None]
            _weights = topk_A.flatten()  # (n * knn)
        
        _values = subgraph_output[topk_indices.flatten()]  # (n * knn, d)
        topk_output = _values * _weights[:, None]  # (n * knn, d)
        topk_output = topk_output.reshape(-1, knn, _values.shape[-1])  # (n, knn, d)
        topk_output = topk_output.sum(dim=1)  # (n, d)

        if move_output_to_cpu:
            topk_output = topk_output.cpu()
        fullgraph_outputs.append(topk_output)

    fullgraph_outputs = torch.cat(fullgraph_outputs, dim=0)

    return fullgraph_outputs
