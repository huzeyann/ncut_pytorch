import logging
from typing import Literal, Tuple

import torch

from .nystrom import (
    EigSolverOptions,
    OnlineKernel,
    OnlineNystrom,
    solve_eig,
)
from .propagation_utils import (
    affinity_from_features,
    run_subgraph_sampling,
)


DistanceOptions = Literal["cosine", "euclidean", "rbf"]


class LaplacianKernel(OnlineKernel):
    def __init__(
        self,
        affinity_focal_gamma: float,
        distance: DistanceOptions,
        eig_solver: EigSolverOptions,
    ):
        self.affinity_focal_gamma = affinity_focal_gamma
        self.distance: DistanceOptions = distance
        self.eig_solver: EigSolverOptions = eig_solver

        # Anchor matrices
        self.anchor_features: torch.Tensor = None               # [n x d]
        self.A: torch.Tensor = None                             # [n x n]
        self.Ainv: torch.Tensor = None                          # [n x n]

        # Updated matrices
        self.a_r: torch.Tensor = None                           # [n]
        self.b_r: torch.Tensor = None                           # [n]

    def fit(self, features: torch.Tensor) -> None:
        self.anchor_features = features                         # [n x d]
        self.A = affinity_from_features(
            self.anchor_features,                               # [n x d]
            affinity_focal_gamma=self.affinity_focal_gamma,
            distance=self.distance,
            fill_diagonal=False,
        )                                                       # [n x n]
        U, L = solve_eig(self.A, features.shape[-1], self.eig_solver)   # [n x d], [d]
        self.Ainv = U @ torch.diag(1 / L) @ U.mT                # [n x n]
        self.a_r = torch.sum(self.A, dim=-1)                    # [n]
        self.b_r = torch.zeros_like(self.a_r)                   # [n]

    def update(self, features: torch.Tensor) -> torch.Tensor:
        B = affinity_from_features(
            self.anchor_features,                               # [n x d]
            features,                                           # [m x d]
            affinity_focal_gamma=self.affinity_focal_gamma,
            distance=self.distance,
            fill_diagonal=False,
        )                                                       # [n x m]
        b_r = torch.sum(B, dim=-1)                              # [n]
        b_c = torch.sum(B, dim=-2)                              # [m]
        self.b_r = self.b_r + b_r                               # [n]

        rowscale = self.a_r + self.b_r                          # [n]
        colscale = b_c + B.mT @ self.Ainv @ self.b_r            # [m]
        scale = (rowscale[:, None] * colscale) ** -0.5          # [n x m]
        return (B * scale).mT                                   # [m x n]

    def transform(self, features: torch.Tensor = None) -> torch.Tensor:
        rowscale = self.a_r + self.b_r                          # [n]
        if features is None:
            B = self.A                                          # [n x n]
            colscale = rowscale                                 # [n]
        else:
            B = affinity_from_features(
                self.anchor_features,                           # [n x d]
                features,                                       # [m x d]
                affinity_focal_gamma=self.affinity_focal_gamma,
                distance=self.distance,
                fill_diagonal=False,
            )                                                   # [n x m]
            b_c = torch.sum(B, dim=-2)                          # [m]
            colscale = b_c + B.mT @ self.Ainv @ self.b_r        # [m]
        scale = (rowscale[:, None] * colscale) ** -0.5          # [n x m]
        return (B * scale).mT                                   # [m x n]


class NewNCUT(OnlineNystrom):
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
        """
        OnlineNystrom.__init__(
            self,
            num_eig,
            LaplacianKernel(affinity_focal_gamma, distance, eig_solver),
            indirect_pca_dim=indirect_pca_dim if indirect_connection else 0,
            eig_solver=eig_solver,
            chunk_size=matmul_chunk_size,
        )
        self.knn = knn
        self.num_sample = num_sample
        self.sample_method = sample_method
        self.distance = distance
        self.indirect_connection = indirect_connection
        self.normalize_features = normalize_features
        if self.normalize_features is None:
            if distance in ["cosine"]:
                self.normalize_features = True
            if distance in ["euclidean", "rbf"]:
                self.normalize_features = False
        self.make_orthogonal = make_orthogonal

        self.device = device
        self.move_output_to_cpu = move_output_to_cpu
        self.matmul_chunk_size = matmul_chunk_size
        self.verbose = verbose

    def _fit_helper(
        self,
        features: torch.Tensor,
        precomputed_sampled_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # move subgraph gpu to speed up
        original_device = features.device
        device = original_device if self.device is None else self.device

        _n = features.shape[0]
        if self.num_sample >= _n:
            logging.info(
                f"NCUT nystrom num_sample is larger than number of input samples, nyström approximation is not needed, setting num_sample={_n} and knn=1"
            )
            self.num_sample = _n
            self.knn = 1

        # check if features dimension greater than num_eig
        if self.eig_solver in ["svd_lowrank", "lobpcg"]:
            assert (
                _n >= self.n_components * 2
            ), "number of nodes should be greater than 2*num_eig"
        elif self.eig_solver in ["svd", "eigh"]:
            assert (
                _n >= self.n_components
            ), "number of nodes should be greater than num_eig"

        assert self.distance in ["cosine", "euclidean", "rbf"], "distance should be 'cosine', 'euclidean', 'rbf'"

        if self.normalize_features:
            # features need to be normalized for affinity matrix computation (cosine distance)
            features = torch.nn.functional.normalize(features, dim=-1)

        if precomputed_sampled_indices is not None:
            sampled_indices = precomputed_sampled_indices
        else:
            sampled_indices = run_subgraph_sampling(
                features,
                num_sample=self.num_sample,
                sample_method=self.sample_method,
            )
        sampled_features = features[sampled_indices].to(device)
        OnlineNystrom.fit(self, sampled_features)

        _n_not_sampled = _n - len(sampled_features)
        if _n_not_sampled > 0:
            unsampled_indices = torch.full((_n,), True).scatter(0, sampled_indices, False)
            unsampled_features = features[unsampled_indices].to(device)
            if self.indirect_connection:
                V_unsampled, _ = OnlineNystrom.update(self, unsampled_features)
            else:
                V_unsampled, _ = OnlineNystrom.transform(self, unsampled_features)
        else:
            unsampled_indices = V_unsampled = None
        return unsampled_indices, V_unsampled

    def fit(
        self,
        features: torch.Tensor,
        precomputed_sampled_indices: torch.Tensor = None,
    ):
        """Fit Nystrom Normalized Cut on the input features.
        Args:
            features (torch.Tensor): input features, shape (n_samples, n_features)
            precomputed_sampled_indices (torch.Tensor): precomputed sampled indices, shape (num_sample,)
                override the sample_method, if not None
        Returns:
            (NCUT): self
        """
        NewNCUT._fit_helper(self, features, precomputed_sampled_indices)
        return self

    def fit_transform(
        self,
        features: torch.Tensor,
        precomputed_sampled_indices: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features (torch.Tensor): input features, shape (n_samples, n_features)
            precomputed_sampled_indices (torch.Tensor): precomputed sampled indices, shape (num_sample,)
                override the sample_method, if not None

        Returns:
            (torch.Tensor): eigen_vectors, shape (n_samples, num_eig)
            (torch.Tensor): eigen_values, sorted in descending order, shape (num_eig,)
        """
        unsampled_indices, V_unsampled = NewNCUT._fit_helper(self, features, precomputed_sampled_indices)
        V_sampled, L = OnlineNystrom.transform(self)

        if unsampled_indices is not None:
            V = torch.zeros((len(unsampled_indices), self.n_components))
            V[~unsampled_indices] = V_sampled
            V[unsampled_indices] = V_unsampled
        else:
            V = V_sampled
        return V, L


