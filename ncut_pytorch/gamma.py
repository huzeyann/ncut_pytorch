import torch
from .math_utils import get_affinity
from .nystrom_utils import farthest_point_sampling


@torch.no_grad()
def find_gamma_by_degree(X, d_gamma, X2=None, init_gamma=0.5, r_tol=1e-2, max_iter=100):
    # X: (n_samples, n_features)
    # d_gamma: target mean edge weight
    # binary search for optimal gamma, such that the mean edge weight is close to d_gamma
    gamma = init_gamma
    current_degree = get_affinity(X, X2=X2, gamma=gamma).mean()
    i_iter = 0
    low, high = 0, float('inf')
    tol = r_tol * d_gamma
    while abs(current_degree - d_gamma) > tol and i_iter < max_iter:
        if current_degree > d_gamma:
            high = gamma
            gamma = (low + gamma) / 2
        else:
            low = gamma
            gamma = gamma * 2 if high == float('inf') else (gamma + high) / 2
        current_degree = get_affinity(X, X2=X2, gamma=gamma).mean()
        i_iter += 1
    return gamma

@torch.no_grad()
def find_gamma_by_degree_after_fps(X, d_gamma, init_gamma=0.5, r_tol=1e-2, max_iter=100, n_sample=1000):
    indices = farthest_point_sampling(X, n_sample)
    return find_gamma_by_degree(X[indices], d_gamma, init_gamma=init_gamma, r_tol=r_tol, max_iter=max_iter)

