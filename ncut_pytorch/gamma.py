import torch
from .math_utils import get_affinity
from .nystrom_utils import farthest_point_sampling


@torch.no_grad()
def find_gamma_by_degree(features, degree, feature_B=None, init_gamma=0.5, r_tol=1e-2, max_iter=100):
    # features: (n_samples, n_features)
    # binary search for optimal gamma, such that the mean edge weight is close to target_d
    gamma = init_gamma
    current_degree = get_affinity(features, X2=feature_B, gamma=gamma).mean()
    i_iter = 0
    low, high = 0, float('inf')
    tol = r_tol * degree
    while abs(current_degree - degree) > tol and i_iter < max_iter:
        if current_degree > degree:
            high = gamma
            gamma = (low + gamma) / 2
        else:
            low = gamma
            gamma = gamma * 2 if high == float('inf') else (gamma + high) / 2
        current_degree = get_affinity(features, X2=feature_B, gamma=gamma).mean()
        i_iter += 1
    return gamma

@torch.no_grad()
def find_gamma_by_degree_after_fps(features, degree, init_gamma=0.5, r_tol=1e-2, max_iter=100, n_sample=1000):
    # features: (n_samples, n_features)
    # binary search for optimal gamma, such that the mean edge weight is close to target_d
    sample_indices = farthest_point_sampling(features, n_sample)
    sampled_features = features[sample_indices]
    return find_gamma_by_degree(sampled_features, degree, init_gamma=init_gamma, r_tol=r_tol, max_iter=max_iter)

