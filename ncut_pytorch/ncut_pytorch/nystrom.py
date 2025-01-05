from typing import Any, Callable, Dict, Literal, Optional, Tuple

import torch


class OnlineKernel:
    def fit(self, features: torch.Tensor) -> None:                          # [n x d]
        raise NotImplementedError()

    def update(self, features: torch.Tensor) -> torch.Tensor:               # [m x d] -> [m x n]
        raise NotImplementedError()

    def transform(self, featuers: torch.Tensor = None) -> torch.Tensor:     # [m x d] -> [m x n]
        raise NotImplementedError()


class OnlineNystrom:
    SolverOptions = Literal["svd_lowrank", "lobpcg", "svd", "eigh"]

    def __init__(
        self,
        n_components: int,
        kernel: OnlineKernel,
        indirect_pca_dim: int = None,
        eig_solver: SolverOptions = "svd_lowrank",
        make_orthogonal: bool = True,
        device: str = None,
        verbose: bool = False,
    ):
        """
        Args:
            n_components (int): number of top eigenvectors to return
            kernel (OnlineKernel): Online kernel that computes pairwise matrix entries from input features and allows updates
            indirect_pca_dim (int): when compute indirect connection, PCA to reduce the node dimension,
            eig_solver (str): eigen decompose solver, ['svd_lowrank', 'lobpcg', 'svd', 'eigh'].
            make_orthogonal (bool): make eigenvectors orthogonal post-hoc
            device (str): device to use for eigen computation, move to GPU to speeds up a bit (~5x faster)
            verbose (bool): progress bar
        """
        self.n_components: int = n_components
        self.kernel: OnlineKernel = kernel
        self.indirect_pca_dim: int = indirect_pca_dim
        self.eig_solver: OnlineNystrom.SolverOptions = eig_solver
        self.make_orthogonal = make_orthogonal

        self.device: str = device
        self.verbose: bool = verbose

        # Anchor matrices
        self.anchor_features: torch.Tensor = None   # [n x d]
        self.A: torch.Tensor = None                 # [n x n]
        self.Ahinv: torch.Tensor = None             # [n x n]
        self.Ahinv_UL: torch.Tensor = None          # [n x indirect_pca_dim]
        self.Ahinv_VT: torch.Tensor = None          # [indirect_pca_dim x n]

        # Updated matrices
        self.S: torch.Tensor = None                 # [n x n]
        self.transform_matrix: torch.Tensor = None  # [n x n_components]

    def fit(self, features: torch.Tensor) -> None:
        self.anchor_features = features

        self.kernel.fit(self.anchor_features)
        self.A = self.S = self.kernel.transform()                                               # [n x n]

        U, L = solve_eig(self.A, max(self.indirect_pca_dim, self.n_components), self.eig_solver)# [n x ?], [?]
        self.Ahinv_UL = (U * (L ** -0.5))[:, :self.indirect_pca_dim]                            # [n x indirect_pca_dim]
        self.Ahinv_VT = U[:, :self.indirect_pca_dim].mT                                         # [indirect_pca_dim x n]
        self.Ahinv = self.Ahinv_UL @ self.Ahinv_VT                                              # [n x n]

        self.transform_matrix = (U / L)[:, :self.n_components]                                  # [n x n_components]

    def update(self, features: torch.Tensor) -> torch.Tensor:
        B = self.kernel.update(features).mT                                                     # [n x m]
        compressed_B = self.Ahinv_VT @ B                                                        # [indirect_pca_dim x m]
        self.S = self.S + self.Ahinv_UL @ (compressed_B @ compressed_B.mT) @ self.Ahinv_UL.mT   # [n x n]

        US, LS = solve_eig(self.S, self.n_components, self.eig_solver)                          # [n x n_components], [n_components]
        self.transform_matrix = self.Ahinv @ US * (LS ** -0.5)                                  # [n x n_components]
        return B.mT @ self.transform_matrix                                                     # [m x n_components]

    def transform(self, features: torch.Tensor = None) -> torch.Tensor:
        if features is None:
            B = self.A                                                                          # [n x n]
        else:
            B = self.kernel.transform(features).mT                                              # [n x m]
        return B.mT @ self.transform_matrix                                                     # [m x n_components]


def solve_eig(
    A: torch.Tensor,
    num_eig: int,
    eig_solver: Literal["svd_lowrank", "lobpcg", "svd", "eigh"],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch implementation of Eigensolver cut without Nystrom-like approximation.

    Args:
        A (torch.Tensor): input matrix, shape (n_samples, n_samples)
        num_eig (int): number of eigenvectors to return
        eig_solver (str): eigen decompose solver, ['svd_lowrank', 'lobpcg', 'svd', 'eigh']

    Returns:
        (torch.Tensor): eigenvectors corresponding to the eigenvalues, shape (n_samples, num_eig)
        (torch.Tensor): eigenvalues of the eigenvectors, sorted in descending order
    """
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
    return eigen_vector, eigen_value


def correct_rotation(eigen_vector):
    # correct the random rotation (flipping sign) of eigenvectors
    rand_w = torch.ones(
        eigen_vector.shape[0], device=eigen_vector.device, dtype=eigen_vector.dtype
    )
    s = rand_w[None, :] @ eigen_vector
    s = s.sign()
    return eigen_vector * s



