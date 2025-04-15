import torch
import numpy as np


def truncate_inv(A, n_inv=-1):

    if n_inv > 0:
        ev, ei, _ = torch.svd_lowrank(A, n_inv)
        c = ei.sum() / torch.trace(A)
        # ei *= c
        print("eig_inv, c", c)
    else:
        ei, ev = torch.linalg.eigh(A)
        print("eig_inv, ei", ei.min(), ei.max())
    A_inv = ev @ torch.diag(1 / (ei.abs() + 1e-8)) @ ev.T
    return A_inv



from ncut_pytorch.ncut_pytorch import correct_rotation, affinity_from_features, ncut
@torch.no_grad()
def force_nystrom_ncut(
    features,
    sample_indices,
    distance="rbf",
    num_eig=50,
    n_inv=50,
    **kwargs,
):
    A = affinity_from_features(features[sample_indices], distance=distance)
    not_sample_indices = np.setdiff1d(np.arange(features.shape[0]), sample_indices)
    B = affinity_from_features(features[sample_indices], features[not_sample_indices], distance=distance, fill_diagonal=False)

    indices = np.concatenate([sample_indices, not_sample_indices])
    reverse_indices = np.argsort(indices)

    if n_inv > 0:
        A_inv = truncate_inv(A, n_inv=n_inv)
    else:
        A_inv = torch.inverse(A)
    
    C = B.T @ A_inv @ B
    
    # W = [A, B; B.T, C]
    W = torch.cat([torch.cat([A, B], dim=1), torch.cat([B.T, C], dim=1)], dim=0)

    V, L = ncut(W, num_eig)
    V = V[reverse_indices]
    V = correct_rotation(V)

    return V, L
    # eigenvectors, eigenvalues