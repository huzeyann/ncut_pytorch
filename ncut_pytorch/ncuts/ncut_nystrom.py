__all__ = ['ncut_fn', 'nystrom_propagate']

from typing import Callable, Union

import torch

from ncut_pytorch.utils.gamma import find_gamma_by_degree_after_fps
from ncut_pytorch.utils.math import rbf_affinity, gram_schmidt, normalize_affinity, svd_lowrank, correct_rotation, \
    keep_topk_per_row
from ncut_pytorch.utils.sample import farthest_point_sampling
from ncut_pytorch.utils.device import auto_device


class NystromConfig:
    """
    Internal configuration for nystrom approximation, can be overridden by kwargs
    Values are optimized based on empirical experiments, no need to change the values
    """
    n_sample = 10240  # number of samples for nystrom approximation, 10240 is large enough for most cases
    n_sample2 = 1024  # number of samples for eigenvector propagation, 1024 is large enough for most cases
    n_neighbors = 8  # number of neighbors for eigenvector propagation, 10 is large enough for most cases
    matmul_chunk_size = 65536  # chunk size for matrix multiplication, larger chunk size is faster but requires more memory
    move_output_to_cpu = True  # if True, will move output to cpu, saves VRAM
    
    def update(self, kwargs: dict):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid kwarg: {key}")


def ncut_fn(
        X: torch.Tensor,
        n_eig: int = 100,
        track_grad: bool = False,
        d_gamma: float = 'auto',
        device: str = None,
        gamma: float = None,
        make_orthogonal: bool = False,
        no_propagation: bool = False,
        affinity_fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor] = rbf_affinity,
        **kwargs,
) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]]:
    """Normalized Cut, balanced sampling and nystrom approximation.
    Function interface that returns both eigenvectors and eigenvalues.

    Args:
        X (torch.Tensor): input features, shape (N, D)
        n_eig (int): number of eigenvectors
        track_grad (bool): keep track of pytorch gradients
        d_gamma (float): affinity gamma parameter, lower d_gamma results in sharper eigenvectors
        device (str): device, default 'auto' (auto detect GPU)
        gamma (float): affinity parameter, override d_gamma if provided
        make_orthogonal (bool): make eigenvectors orthogonal
        no_propagation (bool): if True, return intermediate results without propagation
        affinity_fn (callable): affinity function, default rbf_affinity. Should accept (X1, X2=None, gamma=float) and return affinity matrix
    Returns:
        (torch.Tensor): eigenvectors, shape (N, n_eig)
        (torch.Tensor): eigenvalues, sorted in descending order, shape (n_eig,)
    Examples:
        >>> from ncut_pytorch import ncut_fn
        >>> import torch
        >>> features = torch.rand(10000, 100)
        >>> eigvec, eigval = ncut_fn(features, n_eig=20)
        >>> print(eigvec.shape, eigval.shape)  # (10000, 20) (20,)
    """
    config = NystromConfig()
    config.update(kwargs)

    # use GPU if available
    device = auto_device(X.device, device)

    # skip pytorch gradient computation if track_grad is False
    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(track_grad)

    # subsample for nystrom approximation
    nystrom_indices = farthest_point_sampling(X, n_sample=config.n_sample, device=device)
    nystrom_X = X[nystrom_indices].to(device)

    # find optimal gamma for affinity matrix
    if gamma is None:
        gamma = find_gamma_by_degree_after_fps(nystrom_X, d_gamma, affinity_fn)

    # compute Ncut on the nystrom sampled subgraph
    A = affinity_fn(nystrom_X, gamma=gamma)
    nystrom_eigvec, eigval = _plain_ncut(A, n_eig)

    if no_propagation:
        torch.set_grad_enabled(prev_grad_state)
        return nystrom_eigvec, eigval, nystrom_indices, gamma

    # propagate eigenvectors from subgraph to full graph
    eigvec = nystrom_propagate(
        nystrom_eigvec,
        X,
        nystrom_X,
        n_neighbors=config.n_neighbors,
        n_sample=config.n_sample2,
        gamma=gamma,
        matmul_chunk_size=config.matmul_chunk_size,
        device=device,
        move_output_to_cpu=config.move_output_to_cpu,
        track_grad=track_grad,
        affinity_fn=affinity_fn,
    )

    # post-hoc orthogonalization
    if make_orthogonal:
        eigvec = gram_schmidt(eigvec)

    torch.set_grad_enabled(prev_grad_state)

    return eigvec, eigval


def _plain_ncut(
        A: torch.Tensor,
        n_eig: int = 100,
):
    # normalization; A = D^(-1/2) A D^(-1/2)
    A = normalize_affinity(A)

    eigvec, eigval, _ = svd_lowrank(A, n_eig)

    eigvec = correct_rotation(eigvec)

    return eigvec, eigval


def nystrom_propagate(
        nystrom_out: torch.Tensor,
        X: torch.Tensor,
        nystrom_X: torch.Tensor,
        gamma: float = 1.0,
        track_grad: bool = False,
        device: str = None,
        return_indices: bool = False,
        affinity_fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor] = rbf_affinity,
        **kwargs,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """propagate output from nystrom sampled nodes to all nodes,
    use a weighted sum of the nearest neighbors to propagate the output.

    Args:
        nystrom_out (torch.Tensor): output from nystrom sampled nodes, shape (m, D)
        X (torch.Tensor): input features for all nodes, shape (N, D)
        nystrom_X (torch.Tensor): input features from nystrom sampled nodes, shape (m, D)
        gamma (float): affinity parameter, default 1.0
        track_grad (bool): keep track of pytorch gradients, default False
        device (str): device to use for computation, if 'auto', will detect GPU automatically
        affinity_fn (callable): affinity function, default rbf_affinity. Should accept (X1, X2=None, gamma=float) and return affinity matrix

    Returns:
        torch.Tensor: output propagated by nearest neighbors, shape (N, D)
    """

    config = NystromConfig()
    config.update(kwargs)

    # skip pytorch gradient computation if track_grad is False
    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(track_grad)

    device = auto_device(nystrom_out.device, device)
    indices = farthest_point_sampling(nystrom_out, config.n_sample2, device=device)
    nystrom_out = nystrom_out[indices].to(device)
    nystrom_X = nystrom_X[indices].to(device)
    
    D = affinity_fn(nystrom_X, gamma=gamma).mean(1)

    all_outs = []
    n_chunk = config.matmul_chunk_size
    n_neighbors = min(config.n_neighbors, len(indices))
    for i in range(0, X.shape[0], n_chunk):
        end = min(i + n_chunk, X.shape[0])

        _Ai = affinity_fn(X[i:end].to(device), nystrom_X, gamma=gamma)
        _Ai, _indices = keep_topk_per_row(_Ai, n_neighbors)  # (n, n_neighbors)
        _Di = D[_indices].sum(1)
        _Ai = _Ai / _Di[:, None]

        weights = _Ai[..., None]  # (n, n_neighbors, 1)
        neighbors = nystrom_out[_indices.flatten()]
        neighbors = neighbors.reshape(-1, n_neighbors, nystrom_out.shape[-1])  # (n, n_neighbors, d)
        out = weights * neighbors  # (n, n_neighbors, d)
        out = out.sum(dim=1)  # (n, d)

        if config.move_output_to_cpu:
            out = out.to("cpu")
        all_outs.append(out)

    all_outs = torch.cat(all_outs, dim=0)

    torch.set_grad_enabled(prev_grad_state)

    if return_indices:
        return all_outs, indices
    
    return all_outs



