import torch

from .nystrom_utils import farthest_point_sampling, propagate_knn, which_device
from .affinity_gamma import find_gamma_by_degree_after_fps
from .math_utils import affinity_from_features, normalize_affinity, svd_lowrank, correct_rotation
from .kway_ncut import kway_ncut

def bias_ncut_soft(features, fg_idx, bg_idx=None,
                 num_eig=50, 
                 bias_factor=0.5, 
                 bg_factor=0.1,
                 num_sample=10240,
                 distance='rbf', degree=0.1, 
                 device=None):
    """
    Args:
        features: (n_nodes, n_features)
        fg_idx: (n_clicks) indices of the clicked points for the foreground
        bg_idx: (n_clicks) indices of the clicked points for the background
        bias_factor: (float) the factor of the bias term, decrease it to grow the mask bigger, need to be tuned for different images
        device: (torch.device)
        distance: (str) 'rbf' or 'cosine'
        degree: (float) the degree of the affinity matrix, 0.1 is good for most cases, decrease it will sharpen the affinity matrix
        num_sample: (int) increasing it does not necessarily improve the result
        num_eig: (int) does not matter since we only need a binary mask
    """
    if bg_idx is None:
        bg_idx = torch.tensor([], dtype=torch.long)

    n_nodes, n_features = features.shape
    num_sample = min(num_sample, n_nodes//4)
    # farthest point sampling
    fps_idx = farthest_point_sampling(features, num_sample=num_sample)
    fps_idx = torch.tensor(fps_idx, dtype=torch.long)
    # remove pos_idx and neg_idx from fps_idx
    fps_idx = fps_idx[~torch.isin(fps_idx, torch.cat([fg_idx, bg_idx]))]
    # add pos_idx and neg_idx to fps_idx
    fps_idx = torch.cat([fg_idx, bg_idx, fps_idx])
    fg_idx = torch.arange(len(fg_idx))
    bg_idx = torch.arange(len(bg_idx)) + len(fg_idx)
    
    device = which_device(features.device, device)
    _input = features[fps_idx].to(device)

    gamma = find_gamma_by_degree_after_fps(_input, degree=degree, distance=distance)
    affinity = affinity_from_features(_input, distance=distance, affinity_focal_gamma=gamma)
    affinity = normalize_affinity(affinity)
    
    # modify the affinity from the clicks
    click_f = 1 * affinity[fg_idx].mean(0)
    if len(bg_idx) > 0:
        click_f = click_f - bg_factor * affinity[bg_idx].mean(0)
    click_affinity = affinity_from_features(click_f.unsqueeze(1), distance=distance, affinity_focal_gamma=gamma)
    click_affinity = normalize_affinity(click_affinity)
    
    _A = bias_factor * click_affinity + (1 - bias_factor) * affinity
        
    eigvecs, eigvals, _ = svd_lowrank(_A, q=num_eig)
    eigvecs = correct_rotation(eigvecs)

    # propagate the eigenvectors to the full graph
    eigvecs = propagate_knn(eigvecs, features, features[fps_idx], distance=distance, device=device)
        
    return eigvecs, eigvals



def get_mask_and_heatmap(eigvecs, click_idx, num_cluster=2, device=None):
    device = which_device(eigvecs.device, device)
    eigvecs = eigvecs[:, :num_cluster].to(device)

    eigvecs = kway_ncut(eigvecs, return_continuous=True)
    # find which cluster is the foreground
    fg_eigvecs = eigvecs[click_idx]
    fg_idx = fg_eigvecs.mean(0).argmax().item()
    bg_idx = 1 if fg_idx == 0 else 0
    
    # discretize the eigvecs
    mask = eigvecs.argmax(dim=-1) == fg_idx

    heatmap = eigvecs[:, fg_idx] - eigvecs[:, bg_idx]
    
    return mask, heatmap