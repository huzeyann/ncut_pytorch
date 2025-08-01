import torch

from .math import rbf_affinity
from .sample import farthest_point_sampling

@torch.no_grad()
def find_gamma_by_degree(X, d_gamma='auto', affinity_fn=rbf_affinity, X2=None, init_gamma=0.5, r_tol=1e-2, max_iter=100):
    # X: (n_samples, n_features)
    # d_gamma: target mean edge weight
    # binary search for optimal gamma, such that the mean edge weight is close to d_gamma
    if isinstance(d_gamma, float):
        assert d_gamma > 0, "d_gamma must be positive"
    gamma = init_gamma
    current_degrees = affinity_fn(X, X2=X2, gamma=gamma).mean(1)
    current_degree = current_degrees.mean().item()
    if d_gamma == 'auto' or d_gamma is None:
        mask = current_degrees < current_degree
        d_gamma = bin_and_find_mode(current_degrees[mask])

    i_iter = 0
    low, high = 0, float('inf')
    tol = r_tol * d_gamma
    while abs(current_degree - d_gamma) > tol and i_iter < max_iter and gamma < 1:
        if current_degree > d_gamma:
            high = gamma
            gamma = (low + gamma) / 2
        else:
            low = gamma
            gamma = gamma * 2 if high == float('inf') else (gamma + high) / 2
        current_degree = affinity_fn(X, X2=X2, gamma=gamma).mean().item()
        i_iter += 1
        
    gamma = min(gamma, 1)
    return gamma

@torch.no_grad()
def find_gamma_by_degree_after_fps(X, d_gamma='auto', affinity_fn=rbf_affinity, n_sample=1000, **kwargs):
    indices = farthest_point_sampling(X, n_sample)
    return find_gamma_by_degree(X[indices], d_gamma, affinity_fn, **kwargs)


@torch.no_grad()
def bin_and_find_mode(degrees, n_bins=20):
    degrees = degrees.to(torch.float32).to(torch.device('cpu'))
    counts, bin_edges = torch.histogram(degrees, bins=n_bins)
    
    # Find the bin with most data
    max_bin_idx = counts.argmax()
    
    # Get the actual data in that bin
    left_edge = bin_edges[max_bin_idx]
    right_edge = bin_edges[max_bin_idx + 1]
    
    mask = (degrees >= left_edge) & (degrees < right_edge)
    
    # Return the mean of the most populous bin
    return degrees[mask].mean().item()