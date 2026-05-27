__all__ = [
    "rbf_affinity",
    "cosine_affinity",
    "keep_topk_per_row",
    "grad_safe_eig_solve",
    "pca_lowrank",
    "quantile_min_max",
    "quantile_normalize",
    "gram_schmidt",
    "chunked_matmul",
    "correct_rotation",
    "normalize_affinity",
]

import math
import logging

import numpy as np
import torch


_GAMMA_DEPRECATION_WARNED = False

from .torch_mod import svd_lowrank as my_svd_lowrank


def check_gamma_deprecated(gamma: float | None) -> float:
    global _GAMMA_DEPRECATION_WARNED
    if gamma is not None:
        if not _GAMMA_DEPRECATION_WARNED:
            logging.getLogger(__name__).warning("gamma is deprecated, use sigma instead")
            _GAMMA_DEPRECATION_WARNED = True
        sigma = np.sqrt(gamma)
        return sigma


def rbf_affinity(
    X1: torch.Tensor,          # [N,D]
    X2: torch.Tensor | None = None,  # [M,D]
    sigma: float = 1.0,
    zero_diag: bool = False,
    gamma: float | None = None,  # deprecated
) -> torch.Tensor:             # [N,M]
    """Computes RBF affinity matrix: W_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2))."""
    sigma = sigma if gamma is None else check_gamma_deprecated(gamma)
    X2 = X1 if X2 is None else X2

    try:
        x1_sq = X1.pow(2).sum(dim=1, keepdim=True)
        if X2 is X1:
            dist2 = x1_sq + x1_sq.T
        else:
            x2_sq = X2.pow(2).sum(dim=1).unsqueeze(0)
            dist2 = x1_sq + x2_sq
        dist2.addmm_(X1, X2.T, beta=1.0, alpha=-2.0)
        dist2.clamp_min_(0)
    except RuntimeError:
        try:
            dist2 = torch.cdist(X1, X2, p=2).pow_(2)
        except NotImplementedError:
            dist2 = X1.unsqueeze(1) - X2.unsqueeze(0)
            dist2 = dist2.pow(2).sum(dim=-1)
    W = dist2.mul_(-0.5 / (sigma * sigma)).exp_()   # [N,M]
    if zero_diag and X1 is X2:
        W.fill_diagonal_(0.0)
    return W


def cosine_affinity(
    X1: torch.Tensor,          # [N,D]
    X2: torch.Tensor | None = None,  # [M,D]
    sigma: float = 1.0,
    repulse: bool = False,
    zero_diag: bool = False,
    gamma: float | None = None,  # deprecated
) -> torch.Tensor:             # [N,M]
    """Computes cosine-based affinity matrix."""
    sigma = sigma if gamma is None else check_gamma_deprecated(gamma)
    X2 = X1 if X2 is None else X2

    X1_norm = torch.nn.functional.normalize(X1, p=2, dim=1, eps=1e-8)
    X2_norm = torch.nn.functional.normalize(X2, p=2, dim=1, eps=1e-8)
    S = torch.mm(X1_norm, X2_norm.T)
    num = S + 1 if repulse else S - 1
    W = torch.exp(- num**2 / (2.0 * sigma * sigma))
    if not repulse:
        W = W + 1e-3
    if zero_diag and X1 is X2:
        W = W.clone()
        W.fill_diagonal_(0.0)
    return W


def keep_topk_per_row(
    A: torch.Tensor,          # [n_samples, n_samples]
    k: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:  # ([n_samples, k], [n_samples, k])
    """Keeps top-k values per row of affinity matrix."""
    return A.topk(k=k, dim=-1, largest=True)


def grad_safe_eig_solve(
    mat: torch.Tensor,          # [n, m]
    q: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # ([n, q], [q], [m, q])
    """Gradient-safe eigenvalue decomposition."""
    dtype = mat.dtype

    if mat.requires_grad:
        # svd_lowrank is not good for gradient because of truncation, so use eigh with full spectrum instead
        is_symmetric = mat.shape[0] == mat.shape[1]
        if is_symmetric:
            s, u = torch.linalg.eigh(mat)
            s = torch.flip(s, dims=[0])
            u = torch.flip(u, dims=[1])
        else:
            s, u = torch.linalg.eig(mat)
        return u.to(dtype), s.to(dtype), None

    try:
        with torch.autocast(device_type=mat.device.type, enabled=False):
            if dtype in (torch.float16, torch.bfloat16):
                mat = mat.float()
            u, s, v = svd_lowrank(mat, q=q + 10)
    except RuntimeError:
        if dtype in (torch.float16, torch.bfloat16):
            mat = mat.float()
        u, s, v = svd_lowrank(mat, q=q + 10)

    u, s, v = u[:, :q], s[:q], v[:, :q]
    return u.to(dtype), s.to(dtype), v.to(dtype)


def pca_lowrank(
    mat: torch.Tensor,          # [n, m]
    q: int,
) -> torch.Tensor:             # [n, q]
    """Low-rank PCA projection."""
    u, s, v = svd_lowrank(mat, q)
    s /= math.sqrt(mat.shape[0])
    return u @ torch.diag(s)


def svd_lowrank(mat: torch.Tensor, q: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD lowrank implementation for float16 and bfloat16."""
    dtype = mat.dtype
    try:
        with torch.autocast(device_type=mat.device.type, enabled=False):
            if dtype == torch.float16 or dtype == torch.bfloat16:
                mat = mat.float()  # svd_lowrank does not support float16
            u, s, v = my_svd_lowrank(mat, q=q + 10)
    except RuntimeError:
        if dtype == torch.float16 or dtype == torch.bfloat16:
            mat = mat.float()
        u, s, v = my_svd_lowrank(mat, q=q + 10)

    u, s, v = u[:, :q], s[:q], v[:, :q]
    return u.to(dtype), s.to(dtype), v.to(dtype)


def quantile_min_max(
    x: torch.Tensor,
    q1: float = 0.01,
    q2: float = 0.99,
    n_sample: int = 10000,
) -> tuple[float, float]:
    """Computes robust min and max using quantiles."""
    if x.shape[0] > n_sample:
        np.random.seed(0)
        idx = np.random.choice(x.shape[0], n_sample, replace=False)
        return x[idx].quantile(q1), x[idx].quantile(q2)
    return x.quantile(q1), x.quantile(q2)


def quantile_normalize(
    x: torch.Tensor | np.ndarray,  # [n_samples, n_features]
    q: float = 0.95,
) -> torch.Tensor:             # [n_samples, n_features]
    """Normalizes each dimension of x to [0, 1] using quantiles, robust to outliers."""
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    
    vmax, vmin = quantile_min_max(x, q, 1 - q)
    x = (x - vmin) / (vmax - vmin)
    return x.clamp(0, 1)


def gram_schmidt(
    matrix: torch.Tensor,       # [m, n]
) -> torch.Tensor:             # [m, n]
    """Orthogonalizes a matrix column-wise using the Gram-Schmidt process."""
    dtype = matrix.dtype
    qr_input = matrix

    try:
        with torch.autocast(device_type=matrix.device.type, enabled=False):
            if dtype in (torch.float16, torch.bfloat16):
                qr_input = matrix.float()
            orthogonal_matrix, upper = torch.linalg.qr(qr_input, mode="reduced")
    except RuntimeError:
        if dtype in (torch.float16, torch.bfloat16):
            qr_input = matrix.float()
        orthogonal_matrix, upper = torch.linalg.qr(qr_input, mode="reduced")

    # Keep a stable column orientation that matches the classical implementation.
    signs = torch.sign(torch.diagonal(upper))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    orthogonal_matrix = orthogonal_matrix * signs.unsqueeze(0)
    return orthogonal_matrix.to(dtype)


def chunked_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    device: torch.device | str,
    chunk_size: int = 65536,
    large_device: str = "cpu",
    transform: callable = lambda x: x,
) -> torch.Tensor:
    """Chunked matrix multiplication to avoid OOM, equivalent to out = A @ B."""
    A = A.to(large_device)
    B = B.to(large_device)
    C = torch.zeros(A.shape[0], B.shape[1], device=large_device, dtype=A.dtype)
    
    for i in range(0, A.shape[0], chunk_size):
        end_i = min(i + chunk_size, A.shape[0])
        for j in range(0, B.shape[1], chunk_size):
            end_j = min(j + chunk_size, B.shape[1])
            _A, _B = A[i:end_i], B[:, j:end_j]
            _C_ij = None
            for k in range(0, A.shape[1], chunk_size):
                end_k = min(k + chunk_size, A.shape[1])
                __A, __B = _A[:, k:end_k].to(device), _B[k:end_k].to(device)
                _C = __A @ __B
                _C_ij = _C if _C_ij is None else _C_ij + _C
            _C_ij = transform(_C_ij).to(large_device)
            C[i:end_i, j:end_j] = _C_ij
    return C


def correct_rotation(
    eigvec: torch.Tensor,       # [N, K]
) -> torch.Tensor:             # [N, K]
    """Corrects the random sign (rotation) of eigenvectors for consistency."""
    with torch.no_grad():
        rand_w = torch.ones(eigvec.shape[0], device=eigvec.device, dtype=eigvec.dtype)
        s = (rand_w[None, :] @ eigvec).sign()
    return eigvec * s


def normalize_affinity(
    W: torch.Tensor,          # [N, N]
    eps: float = 1e-8,
) -> torch.Tensor:             # [N, N]
    """Symmetric normalization of affinity matrix: A = S W S, S = diag(D^{-1/2})."""
    D = W.sum(dim=1).clamp_min(eps)
    s = torch.rsqrt(D)                                         # [N]
    return (s[:, None] * W) * s[None, :]                     # [N,N] symmetric
