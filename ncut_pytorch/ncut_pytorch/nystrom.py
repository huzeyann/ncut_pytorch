from typing import Literal, Tuple

import torch


EigSolverOptions = Literal["svd_lowrank", "lobpcg", "svd", "eigh"]


class OnlineKernel:
    def fit(self, features: torch.Tensor) -> None:                          # [n x d]
        raise NotImplementedError()

    def update(self, features: torch.Tensor) -> torch.Tensor:               # [m x d] -> [m x n]
        raise NotImplementedError()

    def transform(self, features: torch.Tensor = None) -> torch.Tensor:     # [m x d] -> [m x n]
        raise NotImplementedError()


class OnlineNystrom:
    def __init__(
        self,
        n_components: int,
        kernel: OnlineKernel,
        eig_solver: EigSolverOptions,
        chunk_size: int = 8192,
    ):
        """
        Args:
            n_components (int): number of top eigenvectors to return
            kernel (OnlineKernel): Online kernel that computes pairwise matrix entries from input features and allows updates
            eig_solver (str): eigen decompose solver, ['svd_lowrank', 'lobpcg', 'svd', 'eigh'].
        """
        self.n_components: int = n_components
        self.kernel: OnlineKernel = kernel
        self.eig_solver: EigSolverOptions = eig_solver
        self.inverse_approximation_dim: int = None

        self.chunk_size = chunk_size

        # Anchor matrices
        self.anchor_features: torch.Tensor = None   # [n x d]
        self.A: torch.Tensor = None                 # [n x n]
        self.Ahinv: torch.Tensor = None             # [n x n]
        self.Ahinv_UL: torch.Tensor = None          # [n x indirect_pca_dim]
        self.Ahinv_VT: torch.Tensor = None          # [indirect_pca_dim x n]

        # Updated matrices
        self.S: torch.Tensor = None                 # [n x n]
        self.transform_matrix: torch.Tensor = None  # [n x n_components]
        self.LS: torch.Tensor = None                # [n]

    def fit(self, features: torch.Tensor):
        OnlineNystrom.fit_transform(self, features)
        return self

    def fit_transform(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.anchor_features = features

        self.kernel.fit(self.anchor_features)
        self.A = self.S = self.kernel.transform()                                                   # [n x n]

        self.inverse_approximation_dim = max(self.n_components, features.shape[-1]) + 1
        U, L = solve_eig(
            self.A,
            num_eig=self.inverse_approximation_dim,
            eig_solver=self.eig_solver,
        )                                                                                           # [n x (? + 1)], [? + 1]
        self.Ahinv_UL = U * (L ** -0.5)                                                             # [n x (? + 1)]
        self.Ahinv_VT = U.mT                                                                        # [(? + 1) x n]
        self.Ahinv = self.Ahinv_UL @ self.Ahinv_VT                                                  # [n x n]

        self.transform_matrix = (U / L)[:, :self.n_components]                                      # [n x n_components]
        self.LS = L[:self.n_components]                                                             # [n_components]
        return U[:, :self.n_components], L[:self.n_components]                                      # [n x n_components], [n_components]

    def update(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_chunks = -(-len(features) // self.chunk_size)
        if n_chunks > 1:
            """ Chunked version """
            chunks = torch.chunk(features, n_chunks, dim=0)
            for chunk in chunks:
                self.kernel.update(chunk)

            compressed_BBT = torch.zeros((self.inverse_approximation_dim, self.inverse_approximation_dim))  # [(? + 1) x (? + 1))]
            for i, chunk in enumerate(chunks):
                _B = self.kernel.transform(chunk).mT                                                    # [n x _m]
                _compressed_B = self.Ahinv_VT @ _B                                                      # [(? + 1) x _m]
                compressed_BBT = compressed_BBT + _compressed_B @ _compressed_B.mT                      # [(? + 1) x (? + 1)]
            self.S = self.S + self.Ahinv_UL @ compressed_BBT @ self.Ahinv_UL.mT                         # [n x n]
            US, self.LS = solve_eig(self.S, self.n_components, self.eig_solver)                         # [n x n_components], [n_components]
            self.transform_matrix = self.Ahinv @ US * (self.LS ** -0.5)                                 # [n x n_components]

            VS = []
            for chunk in chunks:
                VS.append(self.kernel.transform(chunk) @ self.transform_matrix)                         # [_m x n_components]
            VS = torch.cat(VS, dim=0)
            return VS, self.LS                                                                          # [m x n_components], [n_components]
        else:
            """ Unchunked version """
            B = self.kernel.update(features).mT                                                         # [n x m]
            compressed_B = self.Ahinv_VT @ B                                                            # [indirect_pca_dim x m]

            self.S = self.S + self.Ahinv_UL @ (compressed_B @ compressed_B.mT) @ self.Ahinv_UL.mT       # [n x n]
            US, self.LS = solve_eig(self.S, self.n_components, self.eig_solver)                         # [n x n_components], [n_components]
            self.transform_matrix = self.Ahinv @ US * (self.LS ** -0.5)                                 # [n x n_components]

            return B.mT @ self.transform_matrix, self.LS                                                # [m x n_components], [n_components]

    def transform(self, features: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if features is None:
            VS = self.A @ self.transform_matrix                                                     # [n x n_components]
        else:
            n_chunks = -(-len(features) // self.chunk_size)
            if n_chunks > 1:
                """ Chunked version """
                chunks = torch.chunk(features, n_chunks, dim=0)
                VS = []
                for chunk in chunks:
                    VS.append(self.kernel.transform(chunk) @ self.transform_matrix)                     # [_m x n_components]
                VS = torch.cat(VS, dim=0)
            else:
                """ Unchunked version """
                VS = self.kernel.transform(features) @ self.transform_matrix                            # [m x n_components]
        return VS, self.LS                                                                          # [m x n_components], [n_components]


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
    eigen_value, indices = torch.topk(eigen_value, k=num_eig, dim=0)
    eigen_vector = eigen_vector[:, indices]

    # correct the random rotation (flipping sign) of eigenvectors
    eigen_vector = eigen_vector * torch.sum(eigen_vector, dim=0).sign()
    return eigen_vector, eigen_value
