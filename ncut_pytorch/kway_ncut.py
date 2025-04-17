import torch
from .nystrom_utils import farthest_point_sampling


# Multiclass Spectral Clustering, SX Yu, J Shi, 2003
def _discretisation_eigenvector(eigen_vector):
    # Function that discretizes rotated eigenvectors
    n, k = eigen_vector.shape

    # Find the maximum index along each row
    _, J = torch.max(eigen_vector, dim=1)
    Y = torch.zeros(n, k, device=eigen_vector.device).scatter_(1, J.unsqueeze(1), 1)

    return Y

@torch.no_grad()
def kway_ncut(eigen_vectors: torch.Tensor, 
              max_iter=1000, num_sample=10240,
              return_continuous=False,
              return_rotation=False):
    """Multiclass Spectral Clustering, SX Yu, J Shi, 2003

    Args:
        eigen_vectors (torch.Tensor): continuous eigenvectors from NCUT, shape (n, k)
        max_iter (int, optional): Maximum number of iterations.

    Returns:
        torch.Tensor: Discretized eigenvectors, shape (n, k), each row is a one-hot vector.
    """

    # Normalize eigenvectors
    vm = torch.sqrt(torch.sum(eigen_vectors ** 2, dim=1))
    eigen_vectors = eigen_vectors / vm.unsqueeze(1)

    # subsample the eigenvectors, to speed up the computation
    n, k = eigen_vectors.shape
    num_sample = max(num_sample, k)
    sample_idx = farthest_point_sampling(eigen_vectors, num_sample)
    _eigen_vectors = eigen_vectors[sample_idx]


    # Initialize R matrix with the first column from Farthest Point Sampling
    _sample_idx = farthest_point_sampling(_eigen_vectors, k)
    R = _eigen_vectors[_sample_idx].T
    
    # Iterative optimization loop
    last_objective_value = 0
    exit_loop = False
    nb_iterations_discretisation = 0

    while not exit_loop:
        nb_iterations_discretisation += 1

        # Discretize the projected eigenvectors
        _eigenvectors_continuous = _eigen_vectors @ R
        _eigenvectors_discrete = _discretisation_eigenvector(_eigenvectors_continuous)

        # SVD decomposition
        _out = _eigenvectors_discrete.T @ _eigen_vectors
        U, S, Vh = torch.linalg.svd(_out, full_matrices=False)
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

    eigenvectors_continuous = eigen_vectors @ R
    eigenvectors_discrete = _discretisation_eigenvector(eigenvectors_continuous)

    if return_rotation:
        return eigenvectors_discrete, R

    if return_continuous:
        return eigenvectors_continuous

    return eigenvectors_discrete


def axis_align(eigen_vectors, max_iter=1000, num_sample=10240):
    """deprecated, use kway_ncut(return_rotation=True) instead"""
    return kway_ncut(eigen_vectors, max_iter=max_iter, num_sample=num_sample, return_rotation=True)
