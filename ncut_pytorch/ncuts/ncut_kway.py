__all__ = ['kway_ncut', 'axis_align', 'quick_kway']

import torch
import torch.nn.functional as F

from ncut_pytorch.utils.sample import farthest_point_sampling
from ncut_pytorch.utils.device import auto_device
from ncut_pytorch.utils.math import chunked_matmul


def kway_ncut(
    eigvec: torch.Tensor,          # [n, k]
    n_clusters: int | None = None,
    device: str | None = None,
    ret_R: bool = False,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:  # [n, k] or ([n, k], [k, k])
    """
    K-way Ncut discretization.
    
    Args:
        eigvec: Eigenvectors from Ncut output.
        n_clusters: Number of clusters to use. If None, will use the number of eigenvectors.
        device: Device to use for computation.
        ret_R: Whether to return the rotation matrix.
    
    Returns:
        Discretized eigenvectors (rotation matrix if ret_R=True).
    """
    n_clusters = n_clusters or eigvec.shape[1]
    R = axis_align(eigvec[:, :n_clusters], device=device, **kwargs)
    if ret_R:
        return R
    device = auto_device(eigvec.device, device)
    eigvec = chunked_matmul(eigvec[:, :n_clusters], R, device=device, large_device=eigvec.device)
    return eigvec


def quick_kway(
    eigvec: torch.Tensor,          # [n, k]
    n_clusters: int = 10,
    n_eig: int = 10,
    n_sample: int = 10240,
    device: str | None = None,
    kmeans_iter: int = 10,
    ret_R: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:  # [n, k] or ([n, k], [k, k])
    """Quick K-way Ncut using K-means for rotation matrix."""
    R = _kmeans_kway(eigvec, n_clusters, n_eig, n_sample, device, kmeans_iter)
    if ret_R:
        return R
    device = auto_device(eigvec.device, device)
    eigvec = chunked_matmul(eigvec[:, :n_eig], R, device=device, large_device=eigvec.device)
    return eigvec


@torch.no_grad()
def axis_align(
    eigvec: torch.Tensor,          # [n, k]
    device: str | None = None,
    max_iter: int = 1000,
    n_sample: int = 10240,
    sample_idx: torch.Tensor | None = None,
) -> torch.Tensor:                 # [k, k]
    """Multiclass Spectral Clustering (SX Yu, J Shi, 2003)."""
    n, k = eigvec.shape
    if sample_idx is None:
        sample_idx = farthest_point_sampling(eigvec, n_sample, device=device)
    eigvec = eigvec[sample_idx]

    eigvec = F.normalize(eigvec, dim=1)

    # Initialize R matrix with FPS
    _sample_idx = farthest_point_sampling(eigvec, k, device=device)
    R = eigvec[_sample_idx].T
    
    original_device = eigvec.device
    original_dtype = eigvec.dtype
    device = auto_device(original_device, device)
    eigvec = eigvec.to(device=device, dtype=torch.float32)
    R = R.to(device=device, dtype=torch.float32)
    
    last_obj = 0.0
    exit_loop = False
    iter_count = 0

    while not exit_loop:
        iter_count += 1

        # Discretize projected eigenvectors
        _eig_cont = eigvec @ R
        _eig_disc = _onehot_discretize(_eig_cont)
        _eig_disc = _eig_disc.to(device=device, dtype=eigvec.dtype)

        # SVD decomposition
        _out = _eig_disc.T @ eigvec
        _out_dtype = _out.dtype
        try:
            with torch.autocast(device_type=_out.device.type, enabled=False):
                if _out_dtype in (torch.float16, torch.bfloat16):
                    _out = _out.float()
                U, S, Vh = torch.linalg.svd(_out, full_matrices=False)
        except RuntimeError:
            if _out_dtype in (torch.float16, torch.bfloat16):
                _out = _out.float()
            U, S, Vh = torch.linalg.svd(_out, full_matrices=False)
        U, S, Vh = U.to(_out_dtype), S.to(_out_dtype), Vh.to(_out_dtype)
        V = Vh.T

        ncut_val = 2 * (n - torch.sum(S))

        # Check convergence
        if torch.abs(ncut_val - last_obj) < torch.finfo(torch.float32).eps or iter_count > max_iter:
            exit_loop = True
        else:
            last_obj = ncut_val
            R = V @ U.T
            
    R = R.to(device=original_device, dtype=original_dtype)
    R = R[:, torch.argsort(R[1])]
    return R


def _onehot_discretize(eigvec: torch.Tensor) -> torch.Tensor:  # [n, k]
    _, max_idx = torch.max(eigvec, dim=1)
    return torch.nn.functional.one_hot(max_idx, num_classes=eigvec.shape[1])


@torch.no_grad()
def _kmeans_kway(
    eigvec: torch.Tensor,          # [n, k]
    n_clusters: int = 10,
    n_eig: int = 10,
    n_sample: int = 10240,
    device: str | None = None,
    kmeans_iter: int = 10,
) -> torch.Tensor:                 # [k, k]
    """Internal K-means-based rotation matrix computation."""
    original_device = eigvec.device
    original_dtype = eigvec.dtype
    device = auto_device(original_device, device)
    if n_sample is not None and n_sample < eigvec.shape[0]:
        random_indices = torch.randperm(eigvec.shape[0])[:n_sample]
        _eigvec = eigvec[random_indices]
    else:
        _eigvec = eigvec
    _eigvec = _eigvec[:, :n_eig]
    _eigvec = _eigvec.to(device)
    _eigvec = F.normalize(_eigvec, dim=1)
    indices = farthest_point_sampling(_eigvec, n_clusters)
    centroids = _eigvec[indices].clone()
    feature_dim = _eigvec.shape[1]
    ones = torch.ones(_eigvec.shape[0], device=_eigvec.device, dtype=_eigvec.dtype)
    for _ in range(kmeans_iter):
        similarities = torch.mm(_eigvec, centroids.t())
        assignments = similarities.argmax(dim=1)
        counts = torch.zeros(n_clusters, device=_eigvec.device, dtype=_eigvec.dtype)
        counts.index_add_(0, assignments, ones)
        sums = torch.zeros((n_clusters, feature_dim), device=_eigvec.device, dtype=_eigvec.dtype)
        sums.index_add_(0, assignments, _eigvec)
        means = sums / counts.clamp_min(1).unsqueeze(1)
        means = F.normalize(means, dim=1)
        nonempty = counts > 0
        centroids = torch.where(nonempty[:, None], means, centroids)
    R = centroids.t()
    R = R[:, torch.argsort(R[1])]
    R = R.to(device=original_device, dtype=original_dtype)
    return R
