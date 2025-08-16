import math

import numpy as np
import torch

from .torch_mod import svd_lowrank as my_torch_svd_lowrank


def rbf_affinity(
        X1: torch.Tensor,
        X2: torch.Tensor = None,
        gamma: float = 1.0,
):
    """Compute affinity matrix from input features.

    Args:
        X1 (torch.Tensor): input features, shape (n_samples, n_features)
        X2 (torch.Tensor, optional): optional, if not None, compute affinity between two features
        gamma (float): affinity matrix parameter, lower t reduce the edge weights
            on weak connections, default 1.0

    Returns:
        (torch.Tensor): affinity matrix, shape (n_samples, n_samples)
    """
    X2 = X1 if X2 is None else X2

    distances = torch.cdist(X1, X2, p=2) ** 2
    A = torch.exp(-distances / (2 * gamma * X1.var(0).sum() + 1e-8))

    return A


def keep_topk_per_row(A: torch.Tensor, k: int = 10):
    """
    Args:
        A (torch.Tensor): affinity matrix, shape (n_samples, n_samples)
        k (int): number of topk values to return
    Returns:
        (torch.Tensor): topk values, shape (n_samples, k)
    """
    topk_A, topk_indices = A.topk(k=k, dim=-1, largest=True)
    return topk_A, topk_indices

def svd_lowrank(mat: torch.Tensor, q: int):
    """
    SVD lowrank
    mat: (n, m), n data, m features
    q: int
    return: (n, q), (q,), (q, m)
    """
    dtype = mat.dtype
    with torch.autocast(device_type=mat.device.type, enabled=False):
        if dtype == torch.float16 or dtype == torch.bfloat16:
            mat = mat.float()  # svd_lowrank does not support float16

        u, s, v = my_torch_svd_lowrank(mat, q=q + 10)

    u = u[:, :q]
    s = s[:q]
    v = v[:, :q]

    u = u.to(dtype)
    s = s.to(dtype)
    v = v.to(dtype)
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
        device,
        chunk_size=65536,
        large_device="cpu",
        transform=lambda x: x,
):
    """
    Chunked matrix multiplication, to avoid OOM
    equivalent to: out = A @ B
    """
    A = A.to(large_device)
    B = B.to(large_device)
    C = torch.zeros(A.shape[0], B.shape[1], device=large_device, dtype=A.dtype)
    for i in range(0, A.shape[0], chunk_size):
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


def correct_rotation(eigvec):
    # correct the random rotation (flipping sign) of eigenvectors
    with torch.no_grad():
        rand_w = torch.ones(eigvec.shape[0], device=eigvec.device, dtype=eigvec.dtype)
        s = rand_w[None, :] @ eigvec
        s = s.sign()
    eigvec = eigvec * s
    return eigvec


def normalize_affinity(A, eps=1e-8):
    with torch.no_grad():
        D = A.abs().sum(dim=-1)
        D = D + eps
    A = A / torch.sqrt(D)[:, None]
    A = A / torch.sqrt(D)[None, :]
    return A


@torch.no_grad()
def compute_delaunay(points):
    """Compute Delaunay triangulation of points"""
    from scipy.spatial import Delaunay
    if points.shape[1] > 3:
        points = pca_lowrank(points, 3)
    return Delaunay(points.cpu().numpy()).simplices


def compute_riemann_curvature_loss(points, simplices=None, domain_min=0, domain_max=1):
    """
    Calculate loss based on approximated Riemann curvature.
    
    The loss measures deviations from uniform metric tensors across simplices,
    which approximates variations in Riemann curvature.
    """
    if simplices is None:
        simplices = compute_delaunay(points)

    ideal_det = torch.tensor(1.0, device=points.device, dtype=torch.float64)

    # Process each simplex in parallel 
    simplices_tensor = torch.tensor(simplices, device=points.device)

    # Extract points that form each simplex
    simplex_points = points[simplices_tensor]

    # Calculate edge vectors from the first point of each simplex
    edges = simplex_points[:, 1:] - simplex_points[:, 0].unsqueeze(1)

    # Compute metric tensors (Gram matrices) for each simplex
    metric_tensors = torch.matmul(edges, edges.transpose(1, 2))

    # Calculate determinants (related to volume distortion)
    dets = torch.linalg.det(metric_tensors)

    # Penalize deviations from constant determinant
    valid_dets = dets[dets > 0]
    total_curvature = torch.mean((valid_dets - ideal_det) ** 2)
    return total_curvature


def compute_axis_align_loss(points):
    """ Encourage axis alignment by minimizing off-diagonal elements in the covariance matrix """
    n, d = points.shape
    centered_data = points - points.mean(dim=0)  # Center the data
    cov_matrix = (centered_data.T @ centered_data) / n  # Compute covariance matrix

    eye = torch.eye(d, device=points.device)
    return torch.mean((cov_matrix - eye) ** 2)


def compute_repulsion_loss(points):
    dist_matrix = torch.cdist(points, points)
    # Set diagonal to large value to avoid self-repulsion 
    mask = torch.eye(points.shape[0], device=points.device).bool()
    dist_matrix = dist_matrix + mask * 1e10

    # For each point, only consider repulsion from nearest neighbor
    nearest_dists, _ = torch.min(dist_matrix, dim=1)
    repulsion = 1.0 / (nearest_dists + 0.01)  # the shift is to avoid big gradient
    return torch.mean(repulsion)


def compute_attraction_loss(points):
    center = points.mean(dim=0)
    dist = (points - center) ** 2
    return dist.mean()


def compute_boundary_loss(points, domain_min=-1, domain_max=1):
    return torch.mean((torch.relu(domain_min - points)) ** 2) + \
        torch.mean((torch.relu(points - domain_max)) ** 2)


def find_elbow(eigvals, n_elbows=5):
    # Convert to numpy array if tensor
    if torch.is_tensor(eigvals):
        eigvals = eigvals.cpu().detach().numpy()

    # Calculate coordinates
    coords = np.vstack((np.arange(len(eigvals)), eigvals)).T

    # Get vector from first to last point
    line_vec = coords[-1] - coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))

    # Vector from point to first point
    vec_from_first = coords - coords[0]

    # Distance from points to line
    dist_from_line = np.cross(line_vec_norm, vec_from_first)

    # Find elbow points (maximum distances)
    elbow_indices = []
    remaining_distances = np.abs(dist_from_line)

    for _ in range(n_elbows):
        if len(remaining_distances) == 0:
            break
        elbow_idx = np.argmax(remaining_distances)
        elbow_indices.append(elbow_idx)
        # Zero out a window around the found elbow to find the next one
        window = 5  # Adjust this window size as needed
        start_idx = max(0, elbow_idx - window)
        end_idx = min(len(remaining_distances), elbow_idx + window)
        remaining_distances[start_idx:end_idx] = 0

    return sorted(elbow_indices)
