# %%
import torch

from .nystrom_utils import (
    run_subgraph_sampling,
    nystrom_propagate,
    farthest_point_sampling,
    auto_divice,
    _NYSTROM_CONFIG,
)
from .math_utils import (
    get_affinity,
    normalize_affinity,
    svd_lowrank,
    gram_schmidt,
    correct_rotation,
)
from .gamma import find_gamma_by_degree_after_fps


class Ncut:

    def __init__(
        self,
        n_eig: int = 100,
        track_grad: bool = False,
        degree: float = 0.1,
        gamma: float = None,
        device: str = 'auto',
        make_orthogonal: bool = False,
        **kwargs,
    ):
        """
        Nystrom Normalized Cut.
        Args:
            n_eig (int): number of eigenvectors
            track_grad (bool): keep track of pytorch gradients
            degree (float): degree for automatic gamma search,
                lower degree results in sharper eigenvectors
            gamma (float): affinity parameter, default None (auto search based on degree)
            device (str): device, default 'auto' (auto detect GPU)
            make_orthogonal (bool): make eigenvectors orthogonal

        Examples:
            >>> from ncut_pytorch import Ncut
            >>> import torch
            >>> features = torch.rand(10000, 100)
            >>> ncut = Ncut(n_eig=20)
            >>> ncut.fit(features)
            >>> eigvec = ncut.transform(features)
            >>> eigval = ncut.eigval
            >>> print(eigvec.shape, eigval.shape)
            >>> # (10000, 20) (20,)

            >>> # transform new features
            >>> new_features = torch.rand(500, 100)
            >>> new_eigvec =  ncut.transform(new_features)
            >>> print(new_eigvec.shape)
            >>> # (500, 20)
        """
        self.n_eig = n_eig
        self.gamma = gamma
        self.degree = degree
        self.device = device
        self.track_grad = track_grad
        self.make_orthogonal = make_orthogonal

        self._config = _NYSTROM_CONFIG.copy()
        self._config.update(kwargs)

        self._nystrom_x = None
        self._nystrom_eigvec = None
        self._eigval = None
    
    @property
    def eigval(self):
        return self._eigval

    def fit(self, X: torch.Tensor) -> "Ncut":
        """
        Args:
            X (torch.Tensor): input features, shape (N, D)
        Returns:
            (NCUT): self
        """
        self._nystrom_eigvec, self._eigval, nystrom_indices, self.gamma = \
            nystrom_ncut(
                X,
                n_eig=self.n_eig,
                gamma=self.gamma,
                degree=self.degree,
                device=self.device,
                track_grad=self.track_grad,
                make_orthogonal=self.make_orthogonal,
                no_propagation=True,
                **self._config
            )
        self._nystrom_x = X[nystrom_indices]
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): input features, shape (N, D)
        Returns:
            (torch.Tensor): eigenvectors, shape (N, n_eig)
        """
    
        # propagate eigenvectors from subgraph to full graph
        eigvec = nystrom_propagate(
            self._nystrom_eigvec,
            X,
            self._nystrom_x,
            n_neighbors=self._config['n_neighbors'],
            n_sample=self._config['n_sample2'],
            gamma=self.gamma,
            device=self.device,
            move_output_to_cpu=self._config['move_output_to_cpu'],
            track_grad=self.track_grad,
        )
        if self.make_orthogonal:
            eigvec = gram_schmidt(eigvec)
        return eigvec

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): input features, shape (N, D)
        Returns:
            (torch.Tensor): eigenvectors, shape (N, n_eig)
        """
        return self.fit(X).transform(X)


def nystrom_ncut(
    X: torch.Tensor,
    n_eig: int = 100,
    track_grad: bool = False,
    degree: float = 0.1,
    gamma: float = None,
    device: str = 'auto',
    make_orthogonal: bool = False,
    no_propagation: bool = False,
    **kwargs,
):
    """Nystrom Normalized Cut.
    Args:
        X (torch.Tensor): input features, shape (N, D)
        n_eig (int): number of eigenvectors
        track_grad (bool): keep track of pytorch gradients
        degree (float): degree for automatic gamma search,
            lower degree results in sharper eigenvectors
        gamma (float): affinity parameter, default None (auto search based on degree)
        device (str): device, default 'auto' (auto detect GPU)
        make_orthogonal (bool): make eigenvectors orthogonal
    Returns:
        (torch.Tensor): eigenvectors, shape (n_samples, num_eig)
        (torch.Tensor): eigenvalues, sorted in descending order, shape (num_eig,)
    Examples:
        >>> from ncut_pytorch import nystrom_ncut
        >>> import torch
        >>> features = torch.rand(10000, 100)
        >>> eigvec, eigval = nystrom_ncut(features, n_eig=20)
        >>> print(eigvec.shape, eigval.shape)
        >>> # (10000, 20) (20,)
    """
    config = _NYSTROM_CONFIG.copy()
    config.update(kwargs)

    # use GPU if available
    device = auto_divice(X.device, device)

    # skip pytorch gradient computation if track_grad is False
    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(track_grad)

    # sub-sample for nystrom approximation
    nystrom_indices = run_subgraph_sampling(X, n_sample=config['n_sample'], sample_method=config['sample_method'])
    nystrom_X = X[nystrom_indices].to(device)

    # find optimal gamma for affinity matrix
    if gamma is None:
        gamma = find_gamma_by_degree_after_fps(nystrom_X, degree)

    # compute Ncut on the nystrom sampled subgraph
    A = get_affinity(nystrom_X, gamma=gamma)
    nystrom_eigvec, eigval = _plain_ncut(A, n_eig)

    if no_propagation:
        torch.set_grad_enabled(prev_grad_state)
        return nystrom_eigvec, eigval, nystrom_indices, gamma

    # propagate eigenvectors from subgraph to full graph
    eigvec = nystrom_propagate(
        nystrom_eigvec,
        X,
        nystrom_X,
        n_neighbors=config['n_neighbors'],
        n_sample=config['n_sample2'],
        gamma=gamma,
        chunk_size=config['matmul_chunk_size'],
        device=device,
        move_output_to_cpu=config['move_output_to_cpu'],
        track_grad=track_grad,
    )

    # post-hoc orthogonalization
    if make_orthogonal:
        eigvec = gram_schmidt(eigvec)

    torch.set_grad_enabled(prev_grad_state)

    return eigvec, eigval


def _plain_ncut(
    A: torch.Tensor,
    n_eig: int = 100,
):
    """Normalized Cut.

    Args:
        A (torch.Tensor): affinity matrix, shape (n_samples, n_samples)
        n_eig (int): number of eigenvectors to return

    Returns:
        (torch.Tensor): eigenvectors corresponding to the eigenvalues, shape (n_samples, num_eig)
        (torch.Tensor): eigenvalues of the eigenvectors, sorted in descending order
    """

    # normalization; A = D^(-1/2) A D^(-1/2)
    A = normalize_affinity(A)

    eigvec, eigval, _ = svd_lowrank(A, n_eig)

    # correct the random rotation (flipping sign) of eigenvectors
    eigvec = correct_rotation(eigvec)

    return eigvec, eigval

