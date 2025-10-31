from typing import Callable, Union

import torch

from ncut_pytorch.ncuts.ncut_nystrom import ncut_fn, nystrom_propagate
from ncut_pytorch.utils.math import rbf_affinity, cosine_affinity


class Ncut:
    """
    Class interface for Normalized Cut, save states of nystrom approximation, can be used to transform new data.
    """
    
    def __init__(
            self,
            n_eig: int = 100,
            track_grad: bool = False,
            d_gamma: float = None,
            gamma: float = None,
            repulsion_gamma: float = None,
            repulsion_weight: float = 0.2,
            extrapolation_factor: float = 1.0,
            device: str = None,
            affinity_fn: Union["rbf_affinity", "cosine_affinity"] = rbf_affinity,
            **kwargs,
    ):
        """
        
        Args:       
            n_eig (int): number of eigenvectors
            track_grad (bool): keep track of pytorch gradients
            d_gamma (float): affinity gamma parameter, lower d_gamma results in a sharper eigenvectors
            gamma (float): affinity parameter, override d_gamma if provided
            repulsion_gamma (float): (if use repulsion) repulsion gamma parameter, default None (no repulsion)
            repulsion_weight (float): (if use repulsion) repulsion weight, default 0.2
            extrapolation_factor (float): control how far can we extrapolate, larger extrapolation_factor means we can extrapolate further, default 1.0
            device (str): device, default 'auto' (auto detect GPU)
            affinity_fn (callable): affinity function, default rbf_affinity. Should accept (X1, X2=None, gamma=float) and return affinity matrix
            
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
        self.d_gamma = d_gamma
        self.gamma = gamma
        self.repulsion_gamma = repulsion_gamma
        self.repulsion_weight = repulsion_weight
        self.extrapolation_factor = extrapolation_factor
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
        Fit the Ncut model to the input features. save states of nystrom approximation.
        
        Args:
            X (torch.Tensor): input features, shape (N, D)
        Returns:
            ncut (Ncut): Ncut instance
        """
        eigvec, eigval, indices, gamma = \
            ncut_fn(
                X,
                n_eig=self.n_eig,
                d_gamma=self.d_gamma,
                gamma=self.gamma,
                repulsion_gamma=self.repulsion_gamma,
                repulsion_weight=self.repulsion_weight,
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
        Transform new data using the fitted Ncut model and it's saved states of nystrom approximation.
        
        Args:
            X (torch.Tensor): input features, shape (N, D)
            
        Returns:
            eigvec (torch.Tensor): eigenvectors, shape (N, n_eig)
        """
        # check if fit has been called
        if self._nystrom_x is None:
            raise ValueError("Ncut has not been fitted yet. Call fit() first.")

        # propagate eigenvectors from subgraph to full graph
        eigvec = nystrom_propagate(
            self._nystrom_eigvec,
            X,
            self._nystrom_x,
            extrapolation_factor=self.extrapolation_factor,
            device=self.device,
            track_grad=self.track_grad,
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
