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
    farthest_point_sampling,
    which_device,
)
from .math_utils import (
    affinity_from_features,
    normalize_affinity,
    svd_lowrank,
    gram_schmidt,
    correct_rotation,
)
from .affinity_gamma import find_gamma_by_degree_after_fps


def nystrom_ncut(
    features: torch.Tensor,
    num_eig: int = 100,
    degree: float = 0.05,
    distance: Literal["cosine", "euclidean", "rbf"] = "rbf",
    num_sample: int = 10240,
    num_sample2: int = 1024,
    knn: int = 10,
    matmul_chunk_size: int = 16384,
    sample_method: Literal["farthest", "random"] = "farthest",
    precomputed_sampled_indices: torch.Tensor = None,
    affinity_focal_gamma: float = None,
    device: str = None,
    make_orthogonal: bool = False,
    no_propagation: bool = False,
    move_output_to_cpu: bool = None,
    **kwargs,
):
    """PyTorch implementation of Faster Nystrom Normalized cut.
    Args:
        features (torch.Tensor): feature matrix, shape (n_samples, n_features)
        num_eig (int): default 100, number of top eigenvectors to return
        degree (float): target degree to search for optimal gamma, default 0.05. 
            lower degree will result in more sharp eigenvectors
        distance (str): distance metric, 'cosine' (default) or 'euclidean', 'rbf'
        num_sample (int): default 10240, number of samples for Nystrom-like approximation
        num_sample2 (int): default 1024, number of samples for eigenvector propagation
        knn (int): default 10, number of KNN for propagating eigenvectors from subgraph to full graph,
            smaller knn will result in more sharp eigenvectors,
        sample_method (str): sample method, 'farthest' (default) or 'random'
            'farthest' is recommended for better approximation
        precomputed_sampled_indices (torch.Tensor): precomputed sampled indices, shape (num_sample,)
            override the sample_method, if not None
        affinity_focal_gamma (float): affinity matrix parameter, lower t reduce the weak edge weights,
            resulting in more sharp eigenvectors, default None (auto search)
        device (str): device to use for computation, if None, will not change device
            a good practice is to pass features by CPU since it's usually large,
            and move subgraph affinity to GPU to speed up eigenvector computation
        make_orthogonal (bool): make eigenvectors orthogonal after propagation, default True
        no_propagation (bool): if True, skip the eigenvector propagation step, only return the subgraph eigenvectors
        move_output_to_cpu (bool): move output to CPU, set to True if output is too large to fit in GPU memory
    Returns:
        (torch.Tensor): eigenvectors, shape (n_samples, num_eig)
        (torch.Tensor): eigenvalues, sorted in descending order, shape (num_eig,)
        (torch.Tensor): sampled_indices used by Nystrom-like approximation subgraph, shape (num_sample,)
    Examples:
        >>> from ncut_pytorch import nystrom_ncut
        >>> import torch
        >>> features = torch.rand(10000, 100)
        >>> eigenvectors, eigenvalues = nystrom_ncut(features, num_eig=20)
        >>> print(eigenvectors.shape, eigenvalues.shape)
        >>> # (10000, 20) (20,)
    """

    if precomputed_sampled_indices is not None:
        sampled_indices = precomputed_sampled_indices
    else:
        sampled_indices = run_subgraph_sampling(
            features,
            num_sample=num_sample,
            sample_method=sample_method,
        )

    if move_output_to_cpu is None:
        move_output_to_cpu = True if features.device.type == "cpu" else False

    sampled_features = features[sampled_indices]
    device = which_device(sampled_features.device, device)
    sampled_features = sampled_features.to(device)

    # compute affinity matrix on subgraph
    if affinity_focal_gamma is None:
        with torch.no_grad():
            affinity_focal_gamma = find_gamma_by_degree_after_fps(sampled_features, degree, distance=distance)
    
    A = affinity_from_features(sampled_features, affinity_focal_gamma=affinity_focal_gamma, distance=distance)

    # compute normalized cut on the subgraph
    eigen_vector, eigen_value = ncut(A, num_eig)

    if no_propagation:
        return eigen_vector, eigen_value, sampled_indices, affinity_focal_gamma

    # propagate eigenvectors from subgraph to full graph
    eigen_vector = propagate_knn(
        eigen_vector,
        features,
        features[sampled_indices],
        knn,
        num_sample=num_sample2,
        distance=distance,
        affinity_focal_gamma=affinity_focal_gamma,
        chunk_size=matmul_chunk_size,
        device=device,
        move_output_to_cpu=move_output_to_cpu,
    )

    # post-hoc orthogonalization
    if make_orthogonal:
        eigen_vector = gram_schmidt(eigen_vector)

    return eigen_vector, eigen_value


class NCUT:
    """Nystrom Normalized Cut."""

    def __init__(
        self,
        num_eig: int = 100,
        knn: int = 10,
        affinity_focal_gamma: float = None,
        degree: float = 0.05,
        num_sample: int = 10240,
        num_sample2: int = 1024,
        sample_method: Literal["farthest", "random"] = "farthest",
        distance: Literal["cosine", "euclidean", "rbf"] = "rbf",
        device: str = None,
        move_output_to_cpu: bool = False,
        make_orthogonal: bool = False,
        **kwargs,
    ):
        """
        Args:
            num_eig (int): default 100, number of top eigenvectors to return
            num_sample (int): default 10240, number of samples for Nystrom-like approximation
            num_sample2 (int): default 1024, number of samples for eigenvector propagation
            knn (int): default 10, number of KNN for propagating eigenvectors from subgraph to full graph,
                smaller knn will result in more sharp eigenvectors,
            sample_method (str): sample method, 'farthest' (default) or 'random'
                'farthest' is recommended for better approximation
            distance (str): distance metric, 'cosine' (default) or 'euclidean', 'rbf'
            affinity_focal_gamma (float): affinity matrix parameter, lower t reduce the weak edge weights,
                resulting in more sharp eigenvectors, default None (auto search)
            degree (float): target degree to search for optimal gamma, default 0.05. 
                lower degree will result in more sharp eigenvectors
            device (str): device to use for computation, if None, will not change device
                a good practice is to pass features by CPU since it's usually large,
                and move subgraph affinity to GPU to speed up eigenvector computation
            make_orthogonal (bool): make eigenvectors orthogonal after propagation, default True
            move_output_to_cpu (bool): move output to CPU, set to True if you have memory issue

        Examples:
            >>> from ncut_pytorch import NCUT
            >>> import torch
            >>> features = torch.rand(10000, 100)
            >>> ncut = NCUT(num_eig=20)
            >>> ncut.fit(features)
            >>> eigenvectors, eigenvalues = ncut.transform(features)
            >>> print(eigenvectors.shape, eigenvalues.shape)
            >>> # (10000, 20) (20,)

            >>> # transform new features
            >>> new_features = torch.rand(500, 100)
            >>> new_eigenvectors, _ = ncut.transform(new_features)
            >>> print(new_eigenvectors.shape)
            >>> # (500, 20)
        """
        self.num_eig = num_eig
        self.num_sample = num_sample
        self.num_sample2 = num_sample2
        self.knn = knn
        self.sample_method = sample_method
        self.distance = distance
        self.affinity_focal_gamma = affinity_focal_gamma
        self.degree = degree
        self.device = device
        self.move_output_to_cpu = move_output_to_cpu
        self.make_orthogonal = make_orthogonal

        self.subgraph_eigen_vector = None
        self.eigen_value = None
        self.subgraph_sample_indices = None
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
        (self.subgraph_eigen_vector, self.eigen_value,
        self.subgraph_sample_indices, self.affinity_focal_gamma) = nystrom_ncut(
                                features,
                                num_eig=self.num_eig,
                                num_sample=self.num_sample,
                                num_sample2=self.num_sample2,
                                sample_method=self.sample_method,
                                precomputed_sampled_indices=precomputed_sampled_indices,
                                distance=self.distance,
                                affinity_focal_gamma=self.affinity_focal_gamma,
                                degree=self.degree,
                                device=self.device,
                                move_output_to_cpu=self.move_output_to_cpu,
                                no_propagation=True,
                            )
        self.subgraph_features = features[self.subgraph_sample_indices]
        return self

    def transform(self, features: torch.Tensor):
        """Transform new features using the fitted Nystrom Normalized Cut.
        Args:
            features (torch.Tensor): new features, shape (n_samples, n_features)
        Returns:
            (torch.Tensor): eigen_vectors, shape (n_samples, num_eig)
            (torch.Tensor): eigen_values, sorted in descending order, shape (num_eig,)
        """
    
        # propagate eigenvectors from subgraph to full graph
        eigen_vector = propagate_knn(
            self.subgraph_eigen_vector,
            features,
            self.subgraph_features,
            self.knn,
            num_sample=self.num_sample2,
            distance=self.distance,
            affinity_focal_gamma=self.affinity_focal_gamma,
            device=self.device,
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


def ncut(
    A: torch.Tensor,
    num_eig: int = 100,
):
    """PyTorch implementation of Normalized cut without Nystrom-like approximation.

    Args:
        A (torch.Tensor): affinity matrix, shape (n_samples, n_samples)
        num_eig (int): number of eigenvectors to return

    Returns:
        (torch.Tensor): eigenvectors corresponding to the eigenvalues, shape (n_samples, num_eig)
        (torch.Tensor): eigenvalues of the eigenvectors, sorted in descending order
    """

    # normalization; A = D^(-1/2) A D^(-1/2)
    A = normalize_affinity(A)

    eigen_vector, eigen_value, _ = svd_lowrank(A, num_eig)

    # correct the random rotation (flipping sign) of eigenvectors
    eigen_vector = correct_rotation(eigen_vector)

    return eigen_vector, eigen_value

    

## dirty hack for backward compatibility ##

try:
    from .math_utils import (
        quantile_normalize,
        quantile_min_max,
    )
    from .affinity_gamma import (
        find_gamma_by_degree,
        find_gamma_by_degree_after_fps,
    )
    from .visualize_utils import (
        rgb_from_tsne_3d,
        rgb_from_umap_sphere,
        rgb_from_tsne_2d,
        rgb_from_umap_3d,
        rgb_from_umap_2d,
        rotate_rgb_cube,
        convert_to_lab_color,
    )
    from .kway_ncut import kway_ncut, axis_align
except ImportError:
    print("some of viualization and nystrom_utils are not imported")