__all__ = ['kway_ncut', 'axis_align']

import torch
import torch.nn.functional as F

from ncut_pytorch.utils.sample import farthest_point_sampling
from ncut_pytorch.utils.device import auto_device
from ncut_pytorch.utils.math import chunked_matmul


def kway_ncut(eigvec: torch.Tensor, device: str = None, **kwargs):
    """
    Args:
        eigvec (torch.Tensor): eigenvectors from Ncut output, shape (n, k)
    Returns:
        torch.Tensor: eigenvectors, shape (n, k).
            eigvec.argmax(dim=1) is the cluster assignment.
            eigvec.argmax(dim=0) is the cluster centroids.
    """
    R = axis_align(eigvec, device=device, **kwargs)
    device = auto_device(eigvec.device, device)
    # below is equivalent to: eigvec = eigvec @ R
    eigvec = chunked_matmul(eigvec, R, device=device, large_device=eigvec.device)
    return eigvec


@torch.no_grad()
def axis_align(eigvec: torch.Tensor, device: str = None, max_iter=1000, n_sample=10240, sample_idx=None):
    """Multiclass Spectral Clustering, SX Yu, J Shi, 2003

    Args:
        eigvec (torch.Tensor): continuous eigenvectors from NCUT, shape (n, k)
        max_iter (int, optional): Maximum number of iterations.
        n_sample (int, optional): Number of data points to sample.
    Returns:
        torch.Tensor: Rotation matrix, shape (k, k).
    """

    # subsample the eigenvectors, to speed up the computation
    n, k = eigvec.shape
    if sample_idx is None:
        sample_idx = farthest_point_sampling(eigvec, n_sample, device=device)
    eigvec = eigvec[sample_idx]

    eigvec = F.normalize(eigvec, dim=1)

    # Initialize R matrix with the first column from Farthest Point Sampling
    _sample_idx = farthest_point_sampling(eigvec, k, device=device)
    R = eigvec[_sample_idx].T
    
    original_device = eigvec.device
    original_dtype = eigvec.dtype
    device = auto_device(original_device, device)
    eigvec = eigvec.to(device=device, dtype=torch.float32)
    R = R.to(device=device, dtype=torch.float32)
    
    # Iterative optimization loop
    last_objective_value = 0
    exit_loop = False
    nb_iterations_discretisation = 0

    while not exit_loop:
        nb_iterations_discretisation += 1

        # Discretize the projected eigenvectors
        _eigenvectors_continuous = eigvec @ R
        _eigenvectors_discrete = _onehot_discretize(_eigenvectors_continuous)
        _eigenvectors_discrete = _eigenvectors_discrete.to(device=device, dtype=eigvec.dtype)

        # SVD decomposition
        _out = _eigenvectors_discrete.T @ eigvec
        # Handle autocast for SVD - SVD operations don't support half precision
        _out_dtype = _out.dtype
        with torch.autocast(device_type=_out.device.type, enabled=False):
            if _out_dtype == torch.float16 or _out_dtype == torch.bfloat16:
                _out = _out.float()
            U, S, Vh = torch.linalg.svd(_out, full_matrices=False)
        # Convert back to original dtype
        U = U.to(_out_dtype)
        S = S.to(_out_dtype)
        Vh = Vh.to(_out_dtype)
        V = Vh.T

        # Compute the Ncut value
        ncut_value = 2 * (n - torch.sum(S))

        # Check for convergence
        if torch.abs(ncut_value - last_objective_value) < torch.finfo(
            torch.float32).eps or nb_iterations_discretisation > max_iter:
            exit_loop = True
        else:
            last_objective_value = ncut_value
            R = V @ U.T
            
    R = R.to(device=original_device, dtype=original_dtype)
    return R


def _onehot_discretize(eigvec):
    _, max_idx = torch.max(eigvec, dim=1)
    eigvec = torch.nn.functional.one_hot(max_idx, num_classes=eigvec.shape[1])
    return eigvec  # (n, k) one-hot vector