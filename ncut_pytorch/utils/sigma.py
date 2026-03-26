__all__ = ["find_sigma_by_degree"]

import torch

from .math import rbf_affinity
from .sample import farthest_point_sampling


@torch.no_grad()
def _find_sigma_by_degree(
    X: torch.Tensor,                    # [n_samples, n_features]
    quantile_sigma: float = 0.25,
    affinity_fn: callable = rbf_affinity,
    X2: torch.Tensor | None = None,
    init_sigma: float = 0.5,
    r_tol: float = 1e-2,
    max_iter: int = 100,
) -> float:
    """Binary search for optimal sigma to achieve target mean edge weight."""
    if quantile_sigma <= 0 or quantile_sigma >= 1:
        raise ValueError(f"quantile_sigma must be between 0 and 1, got {quantile_sigma}")
    sigma = init_sigma
    
    scale_inv_sigma = X.std(0).sum()
    current_degrees = affinity_fn(X, X2=X2, sigma=scale_inv_sigma).mean(1)
    target_degree = current_degrees.float().quantile(quantile_sigma).item()
    
    # Binary search for sigma
    current_degree = affinity_fn(X, X2=X2, sigma=sigma).mean().item()
    low, high = 0, float('inf')
    tol = r_tol * target_degree
    i_iter = 0
    while abs(current_degree - target_degree) > tol and i_iter < max_iter:
        if current_degree > target_degree:
            high = sigma
            sigma = (low + sigma) / 2
        else:
            low = sigma
            sigma = sigma * 2 if high == float('inf') else (sigma + high) / 2
        current_degree = affinity_fn(X, X2=X2, sigma=sigma).mean().item()
        i_iter += 1
        
    return sigma


@torch.no_grad()
def find_sigma_by_degree(
    X: torch.Tensor,                    # [n_samples, n_features]
    quantile_sigma: float = 0.25,
    affinity_fn: callable = rbf_affinity,
    X2: torch.Tensor | None = None,
    init_sigma: float = 0.5,
    r_tol: float = 1e-2,
    max_iter: int = 100,
    n_sample: int = 1000,
) -> float:
    """Find sigma after FPS-based downsampling for efficiency."""
    indices = farthest_point_sampling(X, n_sample)
    return _find_sigma_by_degree(X[indices], quantile_sigma, affinity_fn, X2=X2, init_sigma=init_sigma, r_tol=r_tol, max_iter=max_iter)
