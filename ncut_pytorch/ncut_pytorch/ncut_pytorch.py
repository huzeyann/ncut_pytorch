# %%
import logging
import math
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from .nystrom_utils import (
    run_subgraph_sampling,
    propagate_knn,
    check_if_normalized,
)


class NCUT:
    """Nystrom Normalized Cut for large scale graph."""

    def __init__(
        self,
        num_eig: int = 100,
        knn: int = 10,
        affinity_focal_gamma: float = 1.0,
        num_sample: int = 10000,
        sample_method: Literal["farthest", "random"] = "farthest",
        distance: Literal["cosine", "euclidean", "rbf"] = "cosine",
        indirect_connection: bool = False,
        indirect_pca_dim: int = 100,
        device: str = None,
        move_output_to_cpu: bool = False,
        eig_solver: Literal["svd_lowrank", "lobpcg", "svd", "eigh"] = "svd_lowrank",
        normalize_features: bool = None,
        matmul_chunk_size: int = 8096,
        make_orthogonal: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            num_eig (int): number of top eigenvectors to return
            knn (int): number of KNN for propagating eigenvectors from subgraph to full graph,
                smaller knn result in more sharp eigenvectors.
            affinity_focal_gamma (float): affinity matrix temperature, lower t reduce the not-so-connected edge weights,
                smaller t result in more sharp eigenvectors.
            num_sample (int): number of samples for Nystrom-like approximation,
                reduce only if memory is not enough, increase for better approximation
            sample_method (str): subgraph sampling, ['farthest', 'random'].
                farthest point sampling is recommended for better Nystrom-approximation accuracy
            distance (str): distance metric for affinity matrix, ['cosine', 'euclidean', 'rbf'].
            indirect_connection (bool): include indirect connection in the Nystrom-like approximation
            indirect_pca_dim (int): when compute indirect connection, PCA to reduce the node dimension,
            device (str): device to use for eigen computation,
                move to GPU to speeds up a bit (~5x faster)
            move_output_to_cpu (bool): move output to CPU, set to True if you have memory issue
            eig_solver (str): eigen decompose solver, ['svd_lowrank', 'lobpcg', 'svd', 'eigh'].
            normalize_features (bool): normalize input features before computing affinity matrix,
                default 'None' is True for cosine distance, False for euclidean distance and rbf
            matmul_chunk_size (int): chunk size for large-scale matrix multiplication
            make_orthogonal (bool): make eigenvectors orthogonal post-hoc
            verbose (bool): progress bar

        Examples:
            >>> from ncut_pytorch import NCUT
            >>> import torch
            >>> features = torch.rand(10000, 100)
            >>> ncut = NCUT(num_eig=20)
            >>> ncut.fit(features)
            >>> eigenvectors, eigenvalues = ncut.transform(features)
            >>> print(eigenvectors.shape, eigenvalues.shape)
            >>> # (10000, 20) (20,)

            >>> from ncut_pytorch import eigenvector_to_rgb
            >>> # use t-SNE or UMAP to convert eigenvectors to RGB
            >>> X_3d, rgb = eigenvector_to_rgb(eigenvectors, method='tsne_3d')
            >>> print(X_3d.shape, rgb.shape)
            >>> # (10000, 3) (10000, 3)

            >>> # transform new features
            >>> new_features = torch.rand(500, 100)
            >>> new_eigenvectors, _ = ncut.transform(new_features)
            >>> print(new_eigenvectors.shape)
            >>> # (500, 20)
        """
        self.num_eig = num_eig
        self.num_sample = num_sample
        self.knn = knn
        self.sample_method = sample_method
        self.distance = distance
        self.affinity_focal_gamma = affinity_focal_gamma
        self.indirect_connection = indirect_connection
        self.indirect_pca_dim = indirect_pca_dim
        self.device = device
        self.move_output_to_cpu = move_output_to_cpu
        self.eig_solver = eig_solver
        self.normalize_features = normalize_features
        if self.normalize_features is None:
            if distance in ["cosine"]:
                self.normalize_features = True
            if distance in ["euclidean", "rbf"]:
                self.normalize_features = False
        self.matmul_chunk_size = matmul_chunk_size
        self.make_orthogonal = make_orthogonal
        self.verbose = verbose

        self.subgraph_eigen_vector = None
        self.eigen_value = None
        self.subgraph_indices = None
        self.subgraph_features = None

    def fit(self,
            features: torch.Tensor,
            precomputed_sampled_indices: torch.Tensor = None
            ):
        """Fit Nystrom Normalized Cut on the input features.
        Args:
            features (torch.Tensor): input features, shape (n_samples, n_features)
            precomputed_sampled_indices (torch.Tensor): precomputed sampled indices, shape (num_sample,)
                override the sample_method, if not None
        Returns:
            (NCUT): self
        """
        _n = features.shape[0]
        if self.num_sample >= _n:
            logging.info(
                f"NCUT nystrom num_sample is larger than number of input samples, nyström approximation is not needed, setting num_sample={_n} and knn=1"
            )
            self.num_sample = _n
            self.knn = 1

        # save the eigenvectors solution on the sub-sampled graph, do not propagate to full graph yet
        self.subgraph_eigen_vector, self.eigen_value, self.subgraph_indices = nystrom_ncut(
            features,
            num_eig=self.num_eig,
            num_sample=self.num_sample,
            sample_method=self.sample_method,
            precomputed_sampled_indices=precomputed_sampled_indices,
            distance=self.distance,
            affinity_focal_gamma=self.affinity_focal_gamma,
            indirect_connection=self.indirect_connection,
            indirect_pca_dim=self.indirect_pca_dim,
            device=self.device,
            eig_solver=self.eig_solver,
            normalize_features=self.normalize_features,
            matmul_chunk_size=self.matmul_chunk_size,
            verbose=self.verbose,
            no_propagation=True,
            move_output_to_cpu=self.move_output_to_cpu,
        )
        self.subgraph_features = features[self.subgraph_indices]
        return self

    def transform(self, features: torch.Tensor, knn: int = None):
        """Transform new features using the fitted Nystrom Normalized Cut.
        Args:
            features (torch.Tensor): new features, shape (n_samples, n_features)
            knn (int): number of KNN for propagating eigenvectors from subgraph to full graph,
        Returns:
            (torch.Tensor): eigen_vectors, shape (n_samples, num_eig)
            (torch.Tensor): eigen_values, sorted in descending order, shape (num_eig,)
        """

        knn = self.knn if knn is None else knn

        # propagate eigenvectors from subgraph to full graph
        eigen_vector = propagate_knn(
            self.subgraph_eigen_vector,
            features,
            self.subgraph_features,
            knn,
            distance=self.distance,
            chunk_size=self.matmul_chunk_size,
            device=self.device,
            use_tqdm=self.verbose,
            move_output_to_cpu=self.move_output_to_cpu,
        )
        if self.make_orthogonal:
            eigen_vector = gram_schmidt(eigen_vector)
        return eigen_vector, self.eigen_value

    def fit_transform(self,
                      features: torch.Tensor,
                      precomputed_sampled_indices: torch.Tensor = None
                      ):
        """
        Args:
            features (torch.Tensor): input features, shape (n_samples, n_features)
            precomputed_sampled_indices (torch.Tensor): precomputed sampled indices, shape (num_sample,)
                override the sample_method, if not None
                
        Returns:
            (torch.Tensor): eigen_vectors, shape (n_samples, num_eig)
            (torch.Tensor): eigen_values, sorted in descending order, shape (num_eig,)
        """
        return self.fit(features, precomputed_sampled_indices=precomputed_sampled_indices).transform(features)


def nystrom_ncut(
    features: torch.Tensor,
    num_eig: int = 100,
    num_sample: int = 10000,
    knn: int = 10,
    sample_method: Literal["farthest", "random"] = "farthest",
    precomputed_sampled_indices: torch.Tensor = None,
    distance: Literal["cosine", "euclidean", "rbf"] = "cosine",
    affinity_focal_gamma: float = 1.0,
    indirect_connection: bool = True,
    indirect_pca_dim: int = 100,
    device: str = None,
    eig_solver: Literal["svd_lowrank", "lobpcg", "svd", "eigh"] = "svd_lowrank",
    normalize_features: bool = None,
    matmul_chunk_size: int = 8096,
    make_orthogonal: bool = True,
    verbose: bool = False,
    no_propagation: bool = False,
    move_output_to_cpu: bool = False,
):
    """PyTorch implementation of Faster Nystrom Normalized cut.
    Args:
        features (torch.Tensor): feature matrix, shape (n_samples, n_features)
        num_eig (int): default 100, number of top eigenvectors to return
        num_sample (int): default 10000, number of samples for Nystrom-like approximation
        knn (int): default 10, number of KNN for propagating eigenvectors from subgraph to full graph,
            smaller knn will result in more sharp eigenvectors,
        sample_method (str): sample method, 'farthest' (default) or 'random'
            'farthest' is recommended for better approximation
        precomputed_sampled_indices (torch.Tensor): precomputed sampled indices, shape (num_sample,)
            override the sample_method, if not None
        distance (str): distance metric, 'cosine' (default) or 'euclidean', 'rbf'
        affinity_focal_gamma (float): affinity matrix parameter, lower t reduce the weak edge weights,
            resulting in more sharp eigenvectors, default 1.0
        indirect_connection (bool): include indirect connection in the subgraph, default True
        indirect_pca_dim (int): default 100, PCA dimension to reduce the node dimension, only applied to
            the not sampled nodes, not applied to the sampled nodes
        device (str): device to use for computation, if None, will not change device
            a good practice is to pass features by CPU since it's usually large,
            and move subgraph affinity to GPU to speed up eigenvector computation
        eig_solver (str): eigen decompose solver, 'svd_lowrank' (default), 'lobpcg', 'svd', 'eigh'
            'svd_lowrank' is recommended for large scale graph, it's the fastest
            they correspond to torch.svd_lowrank, torch.lobpcg, torch.svd, torch.linalg.eigh
        normalize_features (bool): normalize input features before computing affinity matrix,
            default 'None' is True for cosine distance, False for euclidean distance and rbf
        matmul_chunk_size (int): chunk size for matrix multiplication
            large matrix multiplication is chunked to reduce memory usage,
            smaller chunk size will reduce memory usage but slower computation, default 8096
        make_orthogonal (bool): make eigenvectors orthogonal after propagation, default True
        verbose (bool): show progress bar when propagating eigenvectors from subgraph to full graph
        no_propagation (bool): if True, skip the eigenvector propagation step, only return the subgraph eigenvectors
        move_output_to_cpu (bool): move output to CPU, set to True if you have memory issue
    Returns:
        (torch.Tensor): eigenvectors, shape (n_samples, num_eig)
        (torch.Tensor): eigenvalues, sorted in descending order, shape (num_eig,)
        (torch.Tensor): sampled_indices used by Nystrom-like approximation subgraph, shape (num_sample,)
    """

    # check if features dimension greater than num_eig
    if eig_solver in ["svd_lowrank", "lobpcg"]:
        assert features.shape[0] > (
            num_eig * 2
        ), "number of nodes should be greater than 2*num_eig"
    if eig_solver in ["svd", "eigh"]:
        assert (
            features.shape[0] > num_eig
        ), "number of nodes should be greater than num_eig"

    assert distance in ["cosine", "euclidean", "rbf"], "distance should be 'cosine', 'euclidean', 'rbf'"

    if normalize_features:
        # features need to be normalized for affinity matrix computation (cosine distance)
        features = torch.nn.functional.normalize(features, dim=-1)

    if precomputed_sampled_indices is not None:
        sampled_indices = precomputed_sampled_indices
    else:
        sampled_indices = run_subgraph_sampling(
            features,
            num_sample=num_sample,
            sample_method=sample_method,
        )

    sampled_features = features[sampled_indices]
    # move subgraph gpu to speed up
    original_device = sampled_features.device
    device = original_device if device is None else device
    sampled_features = sampled_features.to(device)

    # compute affinity matrix on subgraph
    A = affinity_from_features(
        sampled_features, affinity_focal_gamma=affinity_focal_gamma,
        distance=distance,
    )

    # check if all nodes are sampled, if so, no need for Nystrom approximation
    not_sampled = torch.full((features.shape[0],), True)
    not_sampled[sampled_indices] = False
    _n_not_sampled = not_sampled.sum()

    if _n_not_sampled == 0:
        # if sampled all nodes, no need for nyström approximation
        eigen_vector, eigen_value = ncut(A, num_eig, eig_solver=eig_solver)
        return eigen_vector, eigen_value, sampled_indices

    # 1) PCA to reduce the node dimension for the not sampled nodes
    # 2) compute indirect connection on the PC nodes
    if _n_not_sampled > 0 and indirect_connection:
        indirect_pca_dim = min(indirect_pca_dim, *features.shape)
        U, S, V = torch.pca_lowrank(features[not_sampled].T, q=indirect_pca_dim)
        S = S / math.sqrt(_n_not_sampled)
        feature_B_T = U @ torch.diag(S)
        feature_B = feature_B_T.T
        feature_B = feature_B.to(device)

        B = affinity_from_features(
            sampled_features,
            feature_B,
            affinity_focal_gamma=affinity_focal_gamma,
            distance=distance,
            fill_diagonal=False,
        )
        # P is 1-hop random walk matrix
        B_row = B / B.sum(dim=1, keepdim=True)
        B_col = B / B.sum(dim=0, keepdim=True)
        P = B_row @ B_col.T
        P = (P + P.T) / 2
        # fill diagonal with 0
        P[torch.arange(P.shape[0]), torch.arange(P.shape[0])] = 0
        A = A + P

    # compute normalized cut on the subgraph
    eigen_vector, eigen_value = ncut(A, num_eig, eig_solver=eig_solver)
    eigen_vector = eigen_vector.to(dtype=features.dtype, device=original_device)
    eigen_value = eigen_value.to(dtype=features.dtype, device=original_device)

    if no_propagation:
        return eigen_vector, eigen_value, sampled_indices

    # propagate eigenvectors from subgraph to full graph
    eigen_vector = propagate_knn(
        eigen_vector,
        features,
        sampled_features,
        knn,
        distance=distance,
        chunk_size=matmul_chunk_size,
        device=device,
        use_tqdm=verbose,
        move_output_to_cpu=move_output_to_cpu,
    )

    # post-hoc orthogonalization
    if make_orthogonal:
        eigen_vector = gram_schmidt(eigen_vector)

    return eigen_vector, eigen_value, sampled_indices


def affinity_from_features(
    features: torch.Tensor,
    features_B: torch.Tensor = None,
    affinity_focal_gamma: float = 1.0,
    distance: Literal["cosine", "euclidean", "rbf"] = "cosine",
    fill_diagonal: bool = True,
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

    # if feature_B is not provided, compute affinity matrix on features x features
    # if feature_B is provided, compute affinity matrix on features x feature_B
    if features_B is not None:
        assert not fill_diagonal, "fill_diagonal should be False when feature_B is None"
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

    if fill_diagonal:
        A[torch.arange(A.shape[0]), torch.arange(A.shape[0])] = 0

    # torch.exp make affinity matrix positive definite,
    # lower affinity_focal_gamma reduce the weak edge weights
    if distance != "rbf":
        A = torch.exp(-A / affinity_focal_gamma)
    if distance == "rbf":
        sigma = 2 * affinity_focal_gamma * features.var(dim=0).sum()
        A = torch.exp(-A / sigma)
    return A


def ncut(
    A: torch.Tensor,
    num_eig: int = 100,
    eig_solver: Literal["svd_lowrank", "lobpcg", "svd", "eigh"] = "svd_lowrank",
):
    """PyTorch implementation of Normalized cut without Nystrom-like approximation.

    Args:
        A (torch.Tensor): affinity matrix, shape (n_samples, n_samples)
        num_eig (int): number of eigenvectors to return
        eig_solver (str): eigen decompose solver, ['svd_lowrank', 'lobpcg', 'svd', 'eigh']

    Returns:
        (torch.Tensor): eigenvectors corresponding to the eigenvalues, shape (n_samples, num_eig)
        (torch.Tensor): eigenvalues of the eigenvectors, sorted in descending order
    """

    # make sure A is symmetric
    A = (A + A.T) / 2

    # symmetrical normalization; A = D^(-1/2) A D^(-1/2)
    D = A.sum(dim=-1).detach().clone()
    A /= torch.sqrt(D)[:, None]
    A /= torch.sqrt(D)[None, :]

    # compute eigenvectors
    if eig_solver == "svd_lowrank":  # default
        # only top q eigenvectors, fastest
        eigen_vector, eigen_value, _ = torch.svd_lowrank(A, q=num_eig)
    elif eig_solver == "lobpcg":
        # only top k eigenvectors, fast
        eigen_value, eigen_vector = torch.lobpcg(A, k=num_eig)
    elif eig_solver == "svd":
        # all eigenvectors, slow
        eigen_vector, eigen_value, _ = torch.svd(A)
    elif eig_solver == "eigh":
        # all eigenvectors, slow
        eigen_value, eigen_vector = torch.linalg.eigh(A)
    else:
        raise ValueError(
            "eigen_solver should be 'lobpcg', 'svd_lowrank', 'svd' or 'eigh'"
        )

    # sort eigenvectors by eigenvalues, take top (descending order)
    eigen_value = eigen_value.real
    eigen_vector = eigen_vector.real
    sort_order = torch.argsort(eigen_value, descending=True)[:num_eig]
    eigen_value = eigen_value[sort_order]
    eigen_vector = eigen_vector[:, sort_order]

    # correct the random rotation (flipping sign) of eigenvectors
    eigen_vector = correct_rotation(eigen_vector)

    if eigen_value.min() < 0:
        logging.warning(
            "negative eigenvalues detected, please make sure the affinity matrix is positive definite"
        )

    return eigen_vector, eigen_value


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
    orthogonal_matrix = torch.zeros((m, n), dtype=matrix.dtype)

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


def correct_rotation(eigen_vector):
    # correct the random rotation (flipping sign) of eigenvectors
    rand_w = torch.ones(
        eigen_vector.shape[0], device=eigen_vector.device, dtype=eigen_vector.dtype
    )
    s = rand_w[None, :] @ eigen_vector
    s = s.sign()
    return eigen_vector * s


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


# Multiclass Spectral Clustering, SX Yu, J Shi, 2003
def _discretisation_eigenvector(eigen_vector):
    # Function that discretizes rotated eigenvectors
    n, k = eigen_vector.shape

    # Find the maximum index along each row
    _, J = torch.max(eigen_vector, dim=1)
    Y = torch.zeros(n, k, device=eigen_vector.device).scatter_(1, J.unsqueeze(1), 1)

    return Y


def kway_ncut(eigen_vectors: torch.Tensor, max_iter=300, return_rotation=False, return_continuous=False):
    """Multiclass Spectral Clustering, SX Yu, J Shi, 2003

    Args:
        eigen_vectors (torch.Tensor): continuous eigenvectors from NCUT, shape (n, k)
        max_iter (int, optional): Maximum number of iterations.

    Returns:
        torch.Tensor: Discretized eigenvectors, shape (n, k), each row is a one-hot vector.
    """
        
    # Normalize eigenvectors
    n, k = eigen_vectors.shape
    vm = torch.sqrt(torch.sum(eigen_vectors ** 2, dim=1))
    eigen_vectors = eigen_vectors / vm.unsqueeze(1)

    # Initialize R matrix with the first column from Farthest Point Sampling
    _sample_idx = farthest_point_sampling(eigen_vectors, k)
    R = eigen_vectors[_sample_idx].T
    
    # Iterative optimization loop
    last_objective_value = 0
    exit_loop = False
    nb_iterations_discretisation = 0

    while not exit_loop:
        nb_iterations_discretisation += 1

        # Discretize the projected eigenvectors
        eigenvectors_continuous = eigen_vectors @ R
        eigenvectors_discrete = _discretisation_eigenvector(eigenvectors_continuous)

        # SVD decomposition
        _out = eigenvectors_discrete.T @ eigen_vectors
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

    if return_rotation:
        return eigenvectors_discrete, R

    if return_continuous:
        return eigenvectors_discrete, eigenvectors_continuous

    return eigenvectors_discrete


def axis_align(eigen_vectors, max_iter=300):
    return kway_ncut(eigen_vectors, max_iter=max_iter, return_rotation=True)


## for backward compatibility ##

try:

    from .nystrom_utils import (
        propagate_nearest,
        propagate_eigenvectors,
        quantile_normalize,
        quantile_min_max,
        farthest_point_sampling,
    )
    from .visualize_utils import (
        eigenvector_to_rgb,
        rgb_from_tsne_3d,
        rgb_from_umap_sphere,
        rgb_from_tsne_2d,
        rgb_from_umap_3d,
        rgb_from_umap_2d,
        rotate_rgb_cube,
        convert_to_lab_color,
        _transform_heatmap,
        _clean_mask,
        get_mask,
    )

except ImportError:
    print("some of viualization and nystrom_utils are not imported")