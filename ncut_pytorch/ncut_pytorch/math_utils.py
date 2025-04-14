import torch
import math
import numpy as np
from typing import Literal
from torch.nn import functional as F


def affinity_from_features(
    features: torch.Tensor,
    features_B: torch.Tensor = None,
    affinity_focal_gamma: float = 1.0,
    distance: Literal["cosine", "euclidean", "rbf"] = "rbf",
    **kwargs,
):
    """Compute affinity matrix from input features.

    Args:
        features (torch.Tensor): input features, shape (n_samples, n_features)
        feature_B (torch.Tensor, optional): optional, if not None, compute affinity between two features
        affinity_focal_gamma (float): affinity matrix parameter, lower t reduce the edge weights
            on weak connections, default 1.0
        distance (str): distance metric, 'cosine' (default) or 'euclidean', 'rbf'.
        normalize_features (bool): normalize input features before computing affinity matrix

    Returns:
        (torch.Tensor): affinity matrix, shape (n_samples, n_samples)
    """
    # compute affinity matrix from input features

    features_B = features if features_B is None else features_B

    if distance == "cosine":
        if not check_if_normalized(features):
            features = F.normalize(features, dim=-1)
        if not check_if_normalized(features_B):
            features_B = F.normalize(features_B, dim=-1)
        A = 1 - features @ features_B.T
    elif distance == "euclidean":
        A = torch.cdist(features, features_B, p=2)
    elif distance == "rbf":
        d = torch.cdist(features, features_B, p=2)
        A = torch.pow(d, 2)
    else:
        raise ValueError("distance should be 'cosine' or 'euclidean', 'rbf'")

    if distance == "rbf":
        sigma = 2 * affinity_focal_gamma * features.var(dim=0).sum()
    else:
        sigma = affinity_focal_gamma

    A = torch.exp(-A / sigma)

    return A

def svd_lowrank(mat, q):
    """
    SVD lowrank
    mat: (n, m), n data, m features
    q: int
    return: (n, q), (q,), (q, m)
    """
    u, s, v = torch.svd_lowrank(mat, q=q+10)
    # take 10 extra components to reduce the error, because error in svd_lowrank
    u = u[:, :q]
    s = s[:q]
    v = v[:, :q]
    return u, s, v


def pca_lowrank(mat, q):
    """
    PCA lowrank
    mat: (n, m), n data, m features
    q: int
    return: (n, q), (q,), (q, m)
    """
    u, s, v = svd_lowrank(mat, q)
    _n = mat.shape[0]
    s /= math.sqrt(_n)
    return u @ torch.diag(s) 


def check_if_normalized(x, n_sample=1000):
    """check if the input tensor is normalized (unit norm)"""
    n_sample = min(n_sample, x.shape[0])
    random_indices = torch.randperm(x.shape[0])[:n_sample]
    _x = x[random_indices]
    flag = torch.allclose(torch.norm(_x, dim=-1), torch.ones(n_sample, device=x.device))
    return flag


def quantile_min_max(x, q1=0.01, q2=0.99, n_sample=10000):
    if x.shape[0] > n_sample:
        # random sampling to reduce the load of quantile calculation, torch.quantile does not support large tensor
        np.random.seed(0)
        random_idx = np.random.choice(x.shape[0], n_sample, replace=False)
        vmin, vmax = x[random_idx].quantile(q1), x[random_idx].quantile(q2)
    else:
        vmin, vmax = x.quantile(q1), x.quantile(q2)
    return vmin, vmax


def quantile_normalize(x, q=0.95):
    """normalize each dimension of x to [0, 1], take 95-th percentage, this robust to outliers
        </br> 1. sort x
        </br> 2. take q-th quantile
        </br>     min_value -> (1-q)-th quantile
        </br>     max_value -> q-th quantile
        </br> 3. normalize
        </br> x = (x - min_value) / (max_value - min_value)

    Args:
        x (torch.Tensor): input tensor, shape (n_samples, n_features)
            normalize each feature to 0-1 range
        q (float): quantile, default 0.95

    Returns:
        torch.Tensor: quantile normalized tensor
    """
    # normalize x to 0-1 range, max value is q-th quantile
    # quantile makes the normalization robust to outliers
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    vmax, vmin = quantile_min_max(x, q, 1 - q)
    x = (x - vmin) / (vmax - vmin)
    x = x.clamp(0, 1)
    return x


def gram_schmidt(matrix):
    """Orthogonalize a matrix column-wise using the Gram-Schmidt process.

    Args:
        matrix (torch.Tensor): A matrix to be orthogonalized (m x n).
            the second dimension is orthogonalized
    Returns:
        torch.Tensor: Orthogonalized matrix (m x n).
    """

    # Get the number of rows (m) and columns (n) of the input matrix
    m, n = matrix.shape

    # Create an empty matrix to store the orthogonalized columns
    orthogonal_matrix = torch.zeros((m, n), dtype=matrix.dtype, device=matrix.device)

    for i in range(n):
        # Start with the i-th column of the input matrix
        vec = matrix[:, i]

        for j in range(i):
            # Subtract the projection of vec onto the j-th orthogonal column
            proj = torch.dot(orthogonal_matrix[:, j], matrix[:, i]) / torch.dot(
                orthogonal_matrix[:, j], orthogonal_matrix[:, j]
            )
            vec = vec - proj * orthogonal_matrix[:, j]

        # Store the orthogonalized vector
        orthogonal_matrix[:, i] = vec / torch.norm(vec)

    return orthogonal_matrix


def chunked_matmul(
    A,
    B,
    chunk_size=8096,
    device="cuda:0",
    large_device="cpu",
    transform=lambda x: x,
):
    A = A.to(large_device)
    B = B.to(large_device)
    C = torch.zeros(A.shape[0], B.shape[1], device=large_device)
    iterator = range(0, A.shape[0], chunk_size)
    for i in iterator:
        end_i = min(i + chunk_size, A.shape[0])
        for j in range(0, B.shape[1], chunk_size):
            end_j = min(j + chunk_size, B.shape[1])
            _A = A[i:end_i]
            _B = B[:, j:end_j]
            _C_ij = None
            for k in range(0, A.shape[1], chunk_size):
                end_k = min(k + chunk_size, A.shape[1])
                __A = _A[:, k:end_k].to(device)
                __B = _B[k:end_k].to(device)
                _C = __A @ __B
                _C_ij = _C if _C_ij is None else _C_ij + _C
            _C_ij = transform(_C_ij)

            _C_ij = _C_ij.to(large_device)
            C[i:end_i, j:end_j] = _C_ij
    return C


def correct_rotation(eigen_vector):
    # correct the random rotation (flipping sign) of eigenvectors
    with torch.no_grad():
        rand_w = torch.ones(
            eigen_vector.shape[0], device=eigen_vector.device, dtype=eigen_vector.dtype
        )
        s = rand_w[None, :] @ eigen_vector
        s = s.sign()
    eigen_vector = eigen_vector * s
    return eigen_vector


def normalize_affinity(A, eps=1e-8):
    with torch.no_grad():
        D = A.abs().sum(dim=-1)
        D = D + eps
    A = A / torch.sqrt(D)[:, None]
    A = A / torch.sqrt(D)[None, :]
    return A
