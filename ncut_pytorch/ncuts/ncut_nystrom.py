__all__ = ['ncut_fn', 'nystrom_propagate']

from typing import Callable, Union

import torch
import numpy as np
from ncut_pytorch.utils.sigma import find_sigma_by_degree
from ncut_pytorch.utils.math import rbf_affinity, cosine_affinity
from ncut_pytorch.utils.math import gram_schmidt, normalize_affinity, grad_safe_eig_solve, correct_rotation, keep_topk_per_row, svd_lowrank
from ncut_pytorch.utils.sample import farthest_point_sampling
from ncut_pytorch.utils.device import auto_device
import logging

MATMUL_CHUNK_SIZE = 65536
SMALL_SCALE_THRESHOLD = 8192    # if the number of nodes is less than SMALL_SCALE_THRESHOLD, skip nystrom approximation use exact ncut

class NystromConfig:
    """
    Internal configuration for nystrom approximation, can be overridden by kwargs
    Values are optimized based on empirical experiments, no need to change the values
    """
    n_sample = 10240                # number of samples for nystrom approximation, 10240 is large enough for most cases
    n_sample_max_ratio = 1/4        # max ratio of n_sample to n_data, not full sample ensures balanced sampling
    n_sample2 = 1024                # number of samples for eigenvector propagation, 1024 is large enough for most cases
    n_neighbors = 32                # number of neighbors for eigenvector propagation, 10 is large enough for most cases
    n_neighbors_max_ratio = 1/32    # max ratio of n_neighbors to n_sample2, to avoid over smoothing
    
    def update(self, kwargs: dict):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid kwarg: {key}")


def ncut_fn(
        X: torch.Tensor,
        n_eig: int = 100,
        quantile_sigma: float = 0.25,
        quantile_sigma_repulsion: float = 0.20,
        sigma: float | None = None,
        repulsion_sigma: float | None = None,
        repulsion_weight: float | None = None,
        affinity_fn: Union["rbf_affinity", "cosine_affinity"] = rbf_affinity,
        extrapolation_factor: float = 1.0,
        exact_gradient: bool = False,
        device: str | None = None,
        make_orthogonal: bool = False,
        no_propagation: bool = False,
        **kwargs,
) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]]:
    """Normalized Cut, balanced sampling and nystrom approximation.

    Args:
        X (torch.Tensor): input features, shape (N, D)
        n_eig (int): number of eigenvectors
        quantile_sigma (float): quantile of affinity sigma parameter, lower quantile_sigma results in sharper eigenvectors
        quantile_sigma_repulsion (float): quantile of repulsion sigma parameter, lower quantile_sigma_repulsion results in sharper eigenvectors
        sigma (float): affinity parameter, override d_sigma if provided
        repulsion_sigma (float): (if use repulsion) repulsion sigma parameter, default None (no repulsion)
        repulsion_weight (float): (if use repulsion) repulsion weight, default 0.2
        affinity_fn (callable): affinity function, default rbf_affinity. Should accept (X1, X2=None, sigma=float) and return affinity matrix
        extrapolation_factor (float): control how far can we extrapolate, larger extrapolation_factor means we can extrapolate further, default 1.0
        exact_gradient (bool): use full spectrum and exact gradient, can be slower and unstable, default False
        make_orthogonal (bool): make eigenvectors orthogonal
        
    Returns:
        eigenvectors (torch.Tensor): shape (N, n_eig)
        eigenvalues (torch.Tensor): sorted in descending order, shape (n_eig,)
    
    Examples:
        >>> from ncut_pytorch import ncut_fn
        >>> import torch
        >>> features = torch.rand(10000, 100)
        >>> eigvec, eigval = ncut_fn(features, n_eig=20)
        >>> print(eigvec.shape, eigval.shape)  # (10000, 20) (20,)
    """
    config = NystromConfig()
    config.update(kwargs)
    device = auto_device(X.device, device)

    # subsample for nystrom approximation
    n_sample = min(config.n_sample, int(X.shape[0]*config.n_sample_max_ratio))
    if X.shape[0] > SMALL_SCALE_THRESHOLD:
        nystrom_indices = farthest_point_sampling(X, n_sample=n_sample, device=device)
    else:
        nystrom_indices = torch.arange(X.shape[0])
    nystrom_X = X[nystrom_indices].to(device)

    sigma, repulsion_sigma = find_optimal_sigma(nystrom_X, quantile_sigma, quantile_sigma_repulsion, sigma, repulsion_sigma, affinity_fn)

    if repulsion_sigma and repulsion_weight:
        nystrom_eigvec, eigval = ncut_with_repulsion(nystrom_X, n_eig, sigma, 
            repulsion_sigma, repulsion_weight, affinity_fn, exact_gradient)
    else:
        A = affinity_fn(nystrom_X, sigma=sigma)
        nystrom_eigvec, eigval = _plain_ncut(A, n_eig, exact_gradient)

    if no_propagation:
        return nystrom_eigvec, eigval, nystrom_indices, sigma

    # propagate eigenvectors from subgraph to full graph
    eigvec = nystrom_propagate(
        nystrom_eigvec,
        X,
        nystrom_X,
        extrapolation_factor=extrapolation_factor,
        n_neighbors=config.n_neighbors,
        n_sample=config.n_sample2,
        device=device,
    )

    # post-hoc orthogonalization
    if make_orthogonal:
        eigvec = gram_schmidt(eigvec)

    return eigvec, eigval
    

def find_optimal_sigma(
    X: torch.Tensor,
    quantile_sigma: float = 0.25,
    quantile_sigma_repulsion: float = 0.20,
    sigma: float | None = None,
    repulsion_sigma: float | None = None,
    affinity_fn: Union["rbf_affinity", "cosine_affinity"] = rbf_affinity,
):
    """Find optimal sigma for affinity matrix and repulsion matrix."""
    if affinity_fn == rbf_affinity:
        sigma = sigma or find_sigma_by_degree(X, quantile_sigma, affinity_fn)
        repulsion_sigma = repulsion_sigma or find_sigma_by_degree(X, quantile_sigma_repulsion, affinity_fn, init_sigma=sigma)
    elif affinity_fn == cosine_affinity:
        sigma = sigma or 0.5
        repulsion_sigma = repulsion_sigma or 0.3
    else:
        if sigma is None:
            raise ValueError(f"`sigma` need to be provided for affinity function {affinity_fn}, (sigma=0.5, repulsion_sigma=0.3)")
    return sigma, repulsion_sigma


def ncut_with_repulsion(
    X: torch.Tensor,
    n_eig: int = 100,
    sigma_attraction: float = None,
    sigma_repulsion: float = None,
    repulsion_weight: float = 0.2,
    affinity_fn: Union["rbf_affinity", "cosine_affinity"] = cosine_affinity,
    exact_gradient: bool = False,
    eps: float = 1e-8,
):
    A = affinity_fn(X, sigma=sigma_attraction)
    R = affinity_fn(X, sigma=sigma_repulsion, repulse=True)
    R = R * repulsion_weight
    D_A = A.sum(1) + eps
    D_R = R.sum(1) + eps
    D = D_A + D_R
    W = A - R + torch.diag(D_R)
    W = W / D[:, None]
    if exact_gradient:
        eigvec, eigval, _ = grad_safe_eig_solve(W, n_eig)
    else:
        eigvec, eigval, _ = svd_lowrank(W, n_eig)
    eigvec = correct_rotation(eigvec)
    return eigvec, eigval


def _plain_ncut(
        A: torch.Tensor,
        n_eig: int = 100,
        exact_gradient: bool = False,
):
    A = normalize_affinity(A)
    if exact_gradient:
        eigvec, eigval, _ = grad_safe_eig_solve(A, n_eig)
    else:
        eigvec, eigval, _ = svd_lowrank(A, n_eig)
    eigvec = eigvec[:, :n_eig]
    eigval = eigval[:n_eig]
    eigvec = correct_rotation(eigvec)
    return eigvec, eigval


def nystrom_propagate(
        nystrom_out: torch.Tensor,
        X: torch.Tensor,
        nystrom_X: torch.Tensor,
        extrapolation_factor: float = 1.0,
        device: str = None,
        return_indices: bool = False,
        **kwargs,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """propagate output from nystrom sampled nodes to all nodes,
    use a weighted sum of the nearest neighbors to propagate the output.

    Args:   
        nystrom_out (torch.Tensor): output from nystrom sampled nodes, shape (m, D)
        X (torch.Tensor): input features for all nodes, shape (N, D)
        nystrom_X (torch.Tensor): input features from nystrom sampled nodes, shape (m, D)
        extrapolation_factor (float): control how far can we extrapolate, larger extrapolation_factor means we can extrapolate further, default 1.0
        device (str): device to use for computation, if 'auto', will detect GPU automatically
        return_indices (bool): whether to return the indices used for propagation

    Returns:
        torch.Tensor: output propagated by nearest neighbors, shape (N, D)
    """
    if X.shape[0] <= SMALL_SCALE_THRESHOLD and nystrom_X.shape == X.shape and torch.allclose(nystrom_X.to(X.device), X, atol=1e-6):
        # skip propagation if nystrom_out is the same as X, for small scale graph that don't need nystrom approximation
        if return_indices:
            return nystrom_out, np.arange(X.shape[0])
        return nystrom_out

    config = NystromConfig()
    config.update(kwargs)

    device = auto_device(nystrom_out.device, device)
    output_device = X.device
    indices = farthest_point_sampling(nystrom_out, config.n_sample2, device=device)
    nystrom_out = nystrom_out[indices].to(device)
    nystrom_X = nystrom_X[indices].to(device)
    
    sigma = find_sigma_by_degree(nystrom_X, affinity_fn=rbf_affinity, quantile_sigma=0.25)
    sigma = sigma * extrapolation_factor
    
    D = rbf_affinity(nystrom_X, sigma=sigma).mean(1)

    n_neighbors = int(min(config.n_neighbors, len(indices)*config.n_neighbors_max_ratio))
    n_neighbors = max(n_neighbors, 4)
    n_chunk = _find_max_chunk_size(X, nystrom_X, device)

    all_outs = torch.empty((X.shape[0], nystrom_out.shape[-1]), device=output_device, dtype=nystrom_out.dtype)
    for i in range(0, X.shape[0], n_chunk):
        end = min(i + n_chunk, X.shape[0])

        _Ai = rbf_affinity(X[i:end].to(device), nystrom_X, sigma=sigma)
        _Ai, _indices = keep_topk_per_row(_Ai, n_neighbors)  # (n, n_neighbors)
        
        _Di = D[_indices].sum(1)
        _Ai = _Ai / _Di[:, None]

        out = torch.einsum('nk,nkd->nd', _Ai, nystrom_out[_indices])

        all_outs[i:end] = out.to(output_device)

    if return_indices:
        return all_outs, indices
    return all_outs


def _find_max_chunk_size(X: torch.Tensor, nystrom_X: torch.Tensor, device: str):
    max_chunk_size = MATMUL_CHUNK_SIZE
    while max_chunk_size > 1:
        try:
            _ = rbf_affinity(X[:max_chunk_size].to(device), nystrom_X)
            return max_chunk_size
        except RuntimeError as e:
            max_chunk_size = max_chunk_size // 2
            continue
    raise RuntimeError("failed to find max chunk size")
