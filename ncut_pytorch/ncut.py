from typing import Callable, Union

import torch

from ncut_pytorch.ncuts.ncut_nystrom import ncut_fn, nystrom_propagate
from ncut_pytorch.utils.math import rbf_affinity


class Ncut:

    def __init__(
            self,
            n_eig: int = 100,
            track_grad: bool = False,
            d_gamma: float = None,
            device: str = None,
            affinity_fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor] = rbf_affinity,
            **kwargs,
    ):
        """
        Normalized Cut, balanced sampling and nystrom approximation.
        
        Args:       
            n_eig (int): number of eigenvectors
            track_grad (bool): keep track of pytorch gradients
            d_gamma (float): affinity gamma parameter, lower d_gamma results in a sharper eigenvectors
            device (str): device, default 'auto'

        Examples:
            >>> from ncut_pytorch import Ncut
            >>> import torch
            >>> X = torch.rand(10000, 100)
            >>> 
            >>> # Method 1: Function-like call
            >>> eigvec = Ncut(X, n_eig=20)
            >>> print(eigvec.shape)  # (10000, 20)
            >>> 
            >>> # Method 2: Create instance and call
            >>> ncut = Ncut(n_eig=20)
            >>> eigvec = ncut(X)  # returns just eigvec
            >>> eigval = ncut.eigval  # access eigenvalues separately
            >>> print(eigvec.shape, eigval.shape)  # (10000, 20) (20,)
            >>> 
            >>> # Method 2.1: Transform new data after fitting
            >>> new_X = torch.rand(500, 100)
            >>> new_eigvec = ncut.transform(new_X)
            >>> print(new_eigvec.shape)  # (500, 20)
        """
        self.n_eig = n_eig
        self.d_gamma = d_gamma
        self.device = device
        self.track_grad = track_grad
        self.affinity_fn = affinity_fn
        self.kwargs = kwargs

        self._nystrom_x = None
        self._nystrom_eigvec = None
        self._eigval = None

    @property
    def eigval(self) -> torch.Tensor:
        return self._eigval

    def fit(self, X: torch.Tensor) -> "Ncut":
        """
        Args:
            X (torch.Tensor): input features, shape (N, D)
        Returns:
            self: Ncut instance
        """
        eigvec, eigval, indices, gamma = \
            ncut_fn(
                X,
                n_eig=self.n_eig,
                d_gamma=self.d_gamma,
                device=self.device,
                track_grad=self.track_grad,
                no_propagation=True,
                affinity_fn=self.affinity_fn,
                **self.kwargs
            )
        # store Ncut state to use in transform()
        self._nystrom_x = X[indices]
        self._nystrom_eigvec = eigvec
        self._eigval = eigval
        self.gamma = gamma
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): input features, shape (N, D)
        Returns:
            (torch.Tensor): eigenvectors, shape (N, n_eig)
        """
        # check if fit has been called
        if self._nystrom_x is None:
            raise ValueError("Ncut has not been fitted yet. Call fit() first.")

        # propagate eigenvectors from subgraph to full graph
        eigvec = nystrom_propagate(
            self._nystrom_eigvec,
            X,
            self._nystrom_x,
            gamma=self.gamma,
            device=self.device,
            track_grad=self.track_grad,
            affinity_fn=self.affinity_fn,
            **self.kwargs
        )
        return eigvec

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        return self.fit(X).transform(X)

    def __new__(cls, X: torch.Tensor = None, n_eig: int = 100, track_grad: bool = False, d_gamma: float = None,
                device: str = None, affinity_fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor] = rbf_affinity, 
                **kwargs) -> Union["Ncut", torch.Tensor]:
        if X is not None:
            # function-like behavior
            eigvec, eigval = ncut_fn(X, n_eig=n_eig, track_grad=track_grad, d_gamma=d_gamma, device=device, affinity_fn=affinity_fn, **kwargs)
            return eigvec
        # normal class instantiation
        return super().__new__(cls)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.fit_transform(X)
