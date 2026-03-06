__all__ = ['ncut_click_prompt']

from typing import Callable, Union

import numpy as np
import torch

from ncut_pytorch.utils.sigma import find_sigma_by_degree
from ncut_pytorch.utils.math import rbf_affinity, cosine_affinity, normalize_affinity
from ncut_pytorch.utils.sample import farthest_point_sampling
from ncut_pytorch.utils.device import auto_device
from .ncut_nystrom import NystromConfig
from .ncut_nystrom import nystrom_propagate
from .ncut_nystrom import _plain_ncut


#TODO: automatically optimize click_weight based on the iou of fg and bg
def ncut_click_prompt(
        X: torch.Tensor,
        fg_indices: np.ndarray,
        bg_indices: np.ndarray = None,
        click_weight: float = 0.5,
        bg_weight: float = 0.1,
        n_eig: int = 2,
        quantile_sigma: float = 0.25,
        device: str = None,
        sigma: float = None,
        affinity_fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor] = rbf_affinity,
        exact_gradient: bool = False,
        no_propagation: bool = False,
        return_indices_and_sigma: bool = False,
        **kwargs,
) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]]:

    config = NystromConfig()
    config.update(kwargs)

    # use GPU if available
    device = auto_device(X.device, device)

    if bg_indices is None:
        bg_indices = np.array([], dtype=np.int64)

    # subsample for nystrom approximation
    nystrom_indices = farthest_point_sampling(X, n_sample=config.n_sample, device=device)
    nystrom_indices = torch.tensor(nystrom_indices, dtype=torch.long)
    # remove fg and bg from fps_idx
    nystrom_indices = nystrom_indices[~np.isin(nystrom_indices, np.concatenate([fg_indices, bg_indices]))]
    # add fg and bg to fps_idx
    nystrom_indices = np.concatenate([fg_indices, bg_indices, nystrom_indices])
    fg_indices = np.arange(len(fg_indices))
    bg_indices = np.arange(len(bg_indices)) + len(fg_indices)
    n_fgbg = len(fg_indices) + len(bg_indices)
    
    nystrom_X = X[nystrom_indices].to(device)
    
    # find optimal sigma for affinity matrix
    if sigma is None and affinity_fn == rbf_affinity:
        sigma = find_sigma_by_degree(nystrom_X, quantile_sigma, affinity_fn)
        # TODO: change to std()
    elif sigma is None and affinity_fn == cosine_affinity:
        sigma = 0.5

    # compute Ncut on the nystrom sampled subgraph
    A = affinity_fn(nystrom_X, sigma=sigma)
    A = normalize_affinity(A)

    # modify the affinity from the clicks
    X_click = 1 * A[fg_indices].mean(0)
    if len(bg_indices) > 0:
        X_click = X_click - bg_weight * A[bg_indices].mean(0)
    
    X_click = X_click * A.shape[0]

    A_click = affinity_fn(X_click.unsqueeze(1), sigma=0.5)
    A_click = normalize_affinity(A_click)
    
    _A = click_weight * A_click + (1 - click_weight) * A
        
    nystrom_eigvec, eigval = _plain_ncut(_A, n_eig, exact_gradient=exact_gradient)
    
    if no_propagation:
        return nystrom_eigvec, eigval, nystrom_indices, sigma

    # propagate eigenvectors from subgraph to full graph
    eigvec, nystrom_indices2 = nystrom_propagate(
        nystrom_eigvec,
        X,
        nystrom_X,
        n_neighbors=config.n_neighbors,
        n_sample=config.n_sample2,
        matmul_chunk_size=config.matmul_chunk_size,
        device=device,
        return_indices=True,
    )
    

    if return_indices_and_sigma:
        indices = nystrom_indices[nystrom_indices2]
        return eigvec, eigval, indices, sigma

    return eigvec, eigval
