from typing import Callable, Union

import torch

from ncut_pytorch.ncuts.ncut_nystrom import ncut_fn, nystrom_propagate
from ncut_pytorch.utils.math import rbf_affinity, cosine_affinity, chunked_matmul
from ncut_pytorch.ncuts.ncut_kway import quick_kway
from ncut_pytorch.utils.device import auto_device

class Ncut:
    """
    Class interface for Normalized Cut, save states of nystrom approximation, can be used to transform new data.
    """
    
    def __init__(
            self,
            n_eig: int = 100,
            quantile_sigma: float = 0.25,
            quantile_sigma_repulsion: float = 0.20,
            sigma: float | None = None,
            repulsion_sigma: float | None = None,
            repulsion_weight: float | None = None,
            affinity_fn: Union["rbf_affinity", "cosine_affinity"] = rbf_affinity,
            extrapolation_factor: float = 1.0,
            exact_gradient: bool = False,
            device: str | None = None,
            **kwargs,
    ):
        """
        
        Args:       
            n_eig (int): number of eigenvectors
            n_eig (int): number of eigenvectors
            quantile_sigma (float): quantile of affinity sigma parameter, lower quantile_sigma results in sharper eigenvectors
            quantile_sigma_repulsion (float): quantile of repulsion sigma parameter, lower quantile_sigma_repulsion results in sharper eigenvectors
            sigma (float): affinity parameter, override d_sigma if provided
            repulsion_sigma (float): (if use repulsion) repulsion sigma parameter, default None (no repulsion)
            repulsion_weight (float): (if use repulsion) repulsion weight, default 0.2
            affinity_fn (callable): affinity function, default rbf_affinity. Should accept (X1, X2=None, sigma=float) and return affinity matrix
            extrapolation_factor (float): control how far can we extrapolate, larger extrapolation_factor means we can extrapolate further, default 1.0
            exact_gradient (bool): use full spectrum and exact gradient, can be slower and unstable, default False            device (str): device, default 'auto' (auto detect GPU)
            
        Examples:
            >>> from ncut_pytorch import Ncut
            >>> import torch
            >>> X = torch.rand(10000, 100)
            >>> ncut = Ncut(n_eig=20)
            >>> eigvec = ncut.fit_transform(X)
            >>> eigval = ncut.eigval
            >>> print(eigvec.shape, eigval.shape)  # (10000, 20) (20,)
            >>> 
            >>> # transform new data
            >>> new_X = torch.rand(500, 100)
            >>> new_eigvec = ncut.transform(new_X)
            >>> print(new_eigvec.shape)  # (500, 20)
        """
        self.n_eig = n_eig
        self.quantile_sigma = quantile_sigma
        self.quantile_sigma_repulsion = quantile_sigma_repulsion
        self.sigma = sigma
        self.repulsion_sigma = repulsion_sigma
        self.repulsion_weight = repulsion_weight
        self.extrapolation_factor = extrapolation_factor
        self.exact_gradient = exact_gradient
        self.device = device
        self.affinity_fn = affinity_fn
        self.kwargs = kwargs

        self._nystrom_x = None
        self._nystrom_eigvec = None
        self._eigval = None

        self._kway_R: dict[tuple[int, int], torch.Tensor] = {}

    @property
    def eigval(self) -> torch.Tensor:
        return self._eigval

    def fit(self, X: torch.Tensor) -> "Ncut":
        """
        Fit the Ncut model to the input features. save states of nystrom approximation.
        
        Args:
            X (torch.Tensor): input features, shape (N, D)
        Returns:
            ncut (Ncut): Ncut instance
        """
        eigvec, eigval, indices, sigma = \
            ncut_fn(
                X,
                n_eig=self.n_eig,
                quantile_sigma=self.quantile_sigma,
                quantile_sigma_repulsion=self.quantile_sigma_repulsion,
                sigma=self.sigma,
                repulsion_sigma=self.repulsion_sigma,
                repulsion_weight=self.repulsion_weight,
                device=self.device,
                exact_gradient=self.exact_gradient,
                no_propagation=True,
                affinity_fn=self.affinity_fn,
                **self.kwargs
            )
        # store Ncut state to use in transform()
        self._nystrom_x = X[indices]
        self._nystrom_eigvec = eigvec
        self._eigval = eigval
        self.sigma = sigma
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transform new data using the fitted Ncut model and it's saved states of nystrom approximation.
        
        Args:
            X (torch.Tensor): input features, shape (N, D)
            
        Returns:
            eigvec (torch.Tensor): eigenvectors, shape (N, n_eig)
        """
        self._check_is_fitted()

        # propagate eigenvectors from subgraph to full graph
        eigvec = nystrom_propagate(
            self._nystrom_eigvec,
            X,
            self._nystrom_x,
            extrapolation_factor=self.extrapolation_factor,
            device=self.device,
            **self.kwargs
        )
        return eigvec

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            X (torch.Tensor): input features, shape (N, D)

        Returns:
            eigvec (torch.Tensor): eigenvectors, shape (N, n_eig)
        """
        return self.fit(X).transform(X)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.fit_transform(X)

    def _check_is_fitted(self) -> None:
        if self._nystrom_x is None or self._nystrom_eigvec is None:
            raise ValueError("Ncut has not been fitted yet. Call fit() first.")

    def _validate_kway_params(self, n_clusters: int, n_eig: int) -> None:
        self._check_is_fitted()

        if not isinstance(n_clusters, int) or not isinstance(n_eig, int):
            raise TypeError("n_clusters and n_eig must be integers.")
        if n_clusters <= 0 or n_eig <= 0:
            raise ValueError("n_clusters and n_eig must be positive.")
        if n_eig < 2:
            raise ValueError("n_eig must be at least 2 for k-way discretization.")
        if n_eig > self._nystrom_eigvec.shape[1]:
            raise ValueError(
                f"n_eig={n_eig} exceeds fitted eigenvector count {self._nystrom_eigvec.shape[1]}."
            )

    def kway_fit(self, n_clusters: int, n_eig: int, kmeans_iter: int = 10) -> "Ncut":
        """
        Fit and cache a k-way rotation matrix for the fitted eigenvectors.

        Args:
            n_clusters (int): number of output clusters.
            n_eig (int): number of leading eigenvectors to use.
            kmeans_iter (int): number of k-means refinement iterations.

        Returns:
            Ncut: current instance.
        """
        self._validate_kway_params(n_clusters=n_clusters, n_eig=n_eig)

        R = quick_kway(
            self._nystrom_eigvec[:, :n_eig],
            n_clusters=n_clusters,
            n_eig=n_eig,
            n_sample=self._nystrom_eigvec.shape[0],
            device=self.device,
            kmeans_iter=kmeans_iter,
            ret_R=True,
        )
        self._kway_R[(n_clusters, n_eig)] = R.cpu()
        return self

    def kway_transform(self, X: torch.Tensor, n_clusters: int, n_eig: int) -> torch.Tensor:
        """
        Transform data with a previously fitted k-way rotation matrix.

        Args:
            X (torch.Tensor): input features, shape (N, D).
            n_clusters (int): number of output clusters.
            n_eig (int): number of leading eigenvectors to use.

        Returns:
            torch.Tensor: rotated eigenvectors, shape (N, n_clusters).
        """
        self._validate_kway_params(n_clusters=n_clusters, n_eig=n_eig)

        cache_key = (n_clusters, n_eig)
        if cache_key not in self._kway_R:
            raise ValueError(
                "K-way rotation has not been fitted for this configuration. "
                "Call kway_fit() with the same n_clusters and n_eig first."
            )

        eigvec = self.transform(X)[:, :n_eig]
        R = self._kway_R[cache_key]
        device = auto_device(self.device)

        return chunked_matmul(eigvec, R, device=device, large_device=eigvec.device)
