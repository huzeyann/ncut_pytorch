import logging
from typing import Any, Callable, Dict, Literal

import numpy as np
import torch
from sklearn.base import BaseEstimator

from .ncut_pytorch import propagate_knn, run_subgraph_sampling
from .nystrom_utils import propagate_eigenvectors, quantile_normalize


def _identity(X: torch.Tensor) -> torch.Tensor:
    return X


def eigenvector_to_rgb(
    eigen_vector: torch.Tensor,
    method: Literal["tsne_2d", "tsne_3d", "umap_sphere", "umap_2d", "umap_3d"] = "tsne_3d",
    num_sample: int = 300,
    perplexity: int = 150,
    n_neighbors: int = 150,
    min_distance: float = 0.1,
    metric: Literal["cosine", "euclidean"] = "cosine",
    device: str = None,
    q: float = 0.95,
    knn: int = 10,
    seed: int = 0,
):
    """Use t-SNE or UMAP to convert eigenvectors (more than 3) to RGB color (3D RGB CUBE).

    Args:
        eigen_vector (torch.Tensor): eigenvectors, shape (n_samples, num_eig)
        method (str): method to convert eigenvectors to RGB,
            choices are: ['tsne_2d', 'tsne_3d', 'umap_sphere', 'umap_2d', 'umap_3d']
        num_sample (int): number of samples for Nystrom-like approximation, increase for better approximation
        perplexity (int): perplexity for t-SNE, increase for more global structure
        n_neighbors (int): number of neighbors for UMAP, increase for more global structure
        min_distance (float): minimum distance for UMAP
        metric (str): distance metric, default 'cosine'
        device (str): device to use for computation, if None, will not change device
        q (float): quantile for RGB normalization, default 0.95. lower q results in more sharp colors
        knn (int): number of KNN for propagating eigenvectors from subgraph to full graph,
            smaller knn result in more sharp colors, default 1. knn>1 will smooth-out the embedding
            in the t-SNE or UMAP space.
        seed (int): random seed for t-SNE or UMAP

    Examples:
        >>> from ncut_pytorch import eigenvector_to_rgb
        >>> X_3d, rgb = eigenvector_to_rgb(eigenvectors, method='tsne_3d')
        >>> print(X_3d.shape, rgb.shape)
        >>> # (10000, 3) (10000, 3)

    Returns:
        (torch.Tensor): t-SNE or UMAP embedding, shape (n_samples, 2) or (n_samples, 3)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    kwargs = {
        "num_sample": num_sample,
        "perplexity": perplexity,
        "n_neighbors": n_neighbors,
        "min_distance": min_distance,
        "metric": metric,
        "device": device,
        "q": q,
        "knn": knn,
        "seed": seed,
    }

    if method == "tsne_2d":
        embed, rgb = rgb_from_tsne_2d(eigen_vector, **kwargs)
    elif method == "tsne_3d":
        embed, rgb = rgb_from_tsne_3d(eigen_vector, **kwargs)
    elif method == "umap_sphere":
        embed, rgb = rgb_from_umap_sphere(eigen_vector, **kwargs)
    elif method == "umap_2d":
        embed, rgb = rgb_from_umap_2d(eigen_vector, **kwargs)
    elif method == "umap_3d":
        embed, rgb = rgb_from_umap_3d(eigen_vector, **kwargs)
    else:
        raise ValueError("method should be 'tsne_2d', 'tsne_3d' or 'umap_sphere'")

    return embed, rgb


def _rgb_with_dimensionality_reduction(
    features: torch.Tensor,
    num_sample: int,
    metric: Literal["cosine", "euclidean"],
    rgb_func: Callable[[torch.Tensor, float], torch.Tensor],
    q: float, knn: int,
    seed: int, device: str,
    reduction: Callable[..., BaseEstimator],
    reduction_dim: int,
    reduction_kwargs: Dict[str, Any],
    transform_func: Callable[[torch.Tensor], torch.Tensor] = _identity,
):
    subgraph_indices = run_subgraph_sampling(
        features,
        num_sample=num_sample,
        sample_method="farthest",
    )

    _inp = features[subgraph_indices].cpu().numpy()
    _subgraph_embed = reduction(
        n_components=reduction_dim,
        metric=metric,
        random_state=seed,
        **reduction_kwargs
    ).fit_transform(_inp)

    _subgraph_embed = torch.tensor(_subgraph_embed, dtype=torch.float32)
    X_nd = transform_func(propagate_knn(
        _subgraph_embed,
        features,
        features[subgraph_indices],
        distance=metric,
        knn=knn,
        device=device,
        move_output_to_cpu=True,
    ))
    rgb = rgb_func(X_nd, q)
    return X_nd.numpy(force=True), rgb


def rgb_from_tsne_2d(
    features: torch.Tensor,
    num_sample: int = 300,
    perplexity: int = 150,
    metric: Literal["cosine", "euclidean"] = "cosine",
    device: str = None,
    seed: int = 0,
    q: float = 0.95,
    knn: int = 10,
    **kwargs: Any,
):
    """
    Returns:
        (torch.Tensor): Embedding in 2D, shape (n_samples, 2)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError(
            "sklearn import failed, please install `pip install scikit-learn`"
        )
    num_sample = min(num_sample, features.shape[0])
    if perplexity > num_sample // 2:
        logging.warning(
            f"perplexity is larger than num_sample, set perplexity to {num_sample // 2}"
        )
        perplexity = num_sample // 2

    return _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        metric=metric,
        rgb_func=rgb_from_2d_colormap,
        q=q, knn=knn,
        seed=seed, device=device,
        reduction=TSNE, reduction_dim=2, reduction_kwargs={
            "perplexity": perplexity,
        },
    )


def rgb_from_tsne_3d(
    features: torch.Tensor,
    num_sample: int = 300,
    perplexity: int = 150,
    metric: Literal["cosine", "euclidean"] = "cosine",
    device: str = None,
    seed: int = 0,
    q: float = 0.95,
    knn: int = 10,
    **kwargs: Any,
):
    """
    Returns:
        (torch.Tensor): Embedding in 3D, shape (n_samples, 3)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError(
            "sklearn import failed, please install `pip install scikit-learn`"
        )
    num_sample = min(num_sample, features.shape[0])
    if perplexity > num_sample // 2:
        logging.warning(
            f"perplexity is larger than num_sample, set perplexity to {num_sample // 2}"
        )
        perplexity = num_sample // 2

    return _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        metric=metric,
        rgb_func=rgb_from_3d_rgb_cube,
        q=q, knn=knn,
        seed=seed, device=device,
        reduction=TSNE, reduction_dim=3, reduction_kwargs={
            "perplexity": perplexity,
        },
    )


def rgb_from_umap_2d(
    features: torch.Tensor,
    num_sample: int = 300,
    n_neighbors: int = 150,
    min_dist: float = 0.1,
    metric: Literal["cosine", "euclidean"] = "cosine",
    device: str = None,
    seed: int = 0,
    q: float = 0.95,
    knn: int = 10,
    **kwargs: Any,
):
    """
    Returns:
        (torch.Tensor): Embedding in 2D, shape (n_samples, 2)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap import failed, please install `pip install umap-learn`")

    return _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        metric=metric,
        rgb_func=rgb_from_2d_colormap,
        q=q, knn=knn,
        seed=seed, device=device,
        reduction=UMAP, reduction_dim=2, reduction_kwargs={
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
        },
    )


def rgb_from_umap_sphere(
    features: torch.Tensor,
    num_sample: int = 300,
    n_neighbors: int = 150,
    min_dist: float = 0.1,
    metric: Literal["cosine", "euclidean"] = "cosine",
    device: str = None,
    seed: int = 0,
    q: float = 0.95,
    knn: int = 10,
    **kwargs: Any,
):
    """
    Returns:
        (torch.Tensor): Embedding in 2D, shape (n_samples, 2)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap import failed, please install `pip install umap-learn`")

    def transform_func(X: torch.Tensor) -> torch.Tensor:
        return torch.stack((
            torch.sin(X[:, 0]) * torch.cos(X[:, 1]),
            torch.sin(X[:, 0]) * torch.sin(X[:, 1]),
            torch.cos(X[:, 0]),
        ), dim=1)

    return _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        metric=metric,
        rgb_func=rgb_from_3d_rgb_cube,
        q=q, knn=knn,
        seed=seed, device=device,
        reduction=UMAP, reduction_dim=2, reduction_kwargs={
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "output_metric": "haversine",
        },
        transform_func=transform_func
    )


def rgb_from_umap_3d(
    features: torch.Tensor,
    num_sample: int = 300,
    n_neighbors: int = 150,
    min_dist: float = 0.1,
    metric: Literal["cosine", "euclidean"] = "cosine",
    device: str = None,
    seed: int = 0,
    q: float = 0.95,
    knn: int = 10,
    **kwargs: Any,
):
    """
    Returns:
        (torch.Tensor): Embedding in 2D, shape (n_samples, 2)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap import failed, please install `pip install umap-learn`")

    return _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        metric=metric,
        rgb_func=rgb_from_3d_rgb_cube,
        q=q, knn=knn,
        seed=seed, device=device,
        reduction=UMAP, reduction_dim=3, reduction_kwargs={
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
        },
    )


def flatten_sphere(X_3d):
    x = np.arctan2(X_3d[:, 0], X_3d[:, 1])
    y = -np.arccos(X_3d[:, 2])
    X_2d = np.stack([x, y], axis=1)
    return X_2d


def rotate_rgb_cube(rgb, position=1):
    """rotate RGB cube to different position

    Args:
        rgb (torch.Tensor): RGB color space [0, 1], shape (*, 3)
        position (int): position to rotate, 0, 1, 2, 3, 4, 5, 6

    Returns:
        torch.Tensor: RGB color space, shape (n_samples, 3)
    """
    assert position in range(0, 7), "position should be 0, 1, 2, 3, 4, 5, 6"
    rotation_matrix = torch.tensor(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]
    ).float()
    n_mul = position % 3
    rotation_matrix = torch.matrix_power(rotation_matrix, n_mul)
    rgb = rgb @ rotation_matrix
    if position > 3:
        rgb = 1 - rgb
    return rgb


def rgb_from_3d_rgb_cube(X_3d, q=0.95):
    """convert 3D t-SNE to RGB color space

    Args:
        X_3d (torch.Tensor): 3D t-SNE embedding, shape (n_samples, 3)
        q (float): quantile, default 0.95

    Returns:
        torch.Tensor: RGB color space, shape (n_samples, 3)
    """
    assert X_3d.shape[1] == 3, "input should be (n_samples, 3)"
    assert len(X_3d.shape) == 2, "input should be (n_samples, 3)"
    rgb = []
    for i in range(3):
        rgb.append(quantile_normalize(X_3d[:, i], q=q))
    rgb = torch.stack(rgb, dim=-1)
    return rgb


def convert_to_lab_color(rgb, full_range=True):
    from skimage import color
    import copy

    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
    _rgb = copy.deepcopy(rgb)
    _rgb[..., 0] = _rgb[..., 0] * 100
    if full_range:
        _rgb[..., 1] = _rgb[..., 1] * 255 - 128
        _rgb[..., 2] = _rgb[..., 2] * 255 - 128
    else:
        _rgb[..., 1] = _rgb[..., 1] * 100 - 50
        _rgb[..., 2] = _rgb[..., 2] * 100 - 50
    lab_rgb = color.lab2rgb(_rgb)
    return lab_rgb


def rgb_from_2d_colormap(X_2d, q=0.95):
    xy = X_2d.clone()
    for i in range(2):
        xy[:, i] = quantile_normalize(xy[:, i], q=q)

    try:
        from pycolormap_2d import (
            ColorMap2DBremm,
            ColorMap2DZiegler,
            ColorMap2DCubeDiagonal,
            ColorMap2DSchumann,
        )
    except ImportError:
        raise ImportError(
            "pycolormap_2d import failed, please install `pip install pycolormap-2d`"
        )

    cmap = ColorMap2DCubeDiagonal()
    xy = xy.cpu().numpy()
    len_x, len_y = cmap._cmap_data.shape[:2]
    x = (xy[:, 0] * (len_x - 1)).astype(int)
    y = (xy[:, 1] * (len_y - 1)).astype(int)
    rgb = cmap._cmap_data[x, y]
    rgb = torch.tensor(rgb, dtype=torch.float32) / 255
    return rgb


def propagate_rgb_color(
    rgb: torch.Tensor,
    eigenvectors: torch.Tensor,
    new_eigenvectors: torch.Tensor,
    knn: int = 10,
    num_sample: int = 300,
    sample_method: Literal["farthest", "random"] = "farthest",
    chunk_size: int = 8096,
    device: str = None,
    use_tqdm: bool = False,
):
    """Propagate RGB color to new nodes using KNN.
    Args:
        rgb (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
        features (torch.Tensor): features from existing nodes, shape (n_samples, n_features)
        new_features (torch.Tensor): features from new nodes, shape (n_new_samples, n_features)
        knn (int): number of KNN to propagate RGB color, default 1
        num_sample (int): number of samples for subgraph sampling, default 50000
        sample_method (str): sample method, 'farthest' (default) or 'random'
        chunk_size (int): chunk size for matrix multiplication, default 8096
        device (str): device to use for computation, if None, will not change device
        use_tqdm (bool): show progress bar when propagating RGB color from subgraph to full graph

    Returns:
        torch.Tensor: propagated RGB color for each data sample, shape (n_new_samples, 3)

    Examples:
        >>> old_rgb = torch.randn(3000, 3)
        >>> old_eigenvectors = torch.randn(3000, 20)
        >>> new_eigenvectors = torch.randn(200, 20)
        >>> new_rgb = propagate_rgb_color(old_rgb, new_eigenvectors, old_eigenvectors)
        >>> # new_eigenvectors.shape = (200, 3)
    """
    return propagate_eigenvectors(
        eigenvectors=rgb,
        features=eigenvectors,
        new_features=new_eigenvectors,
        knn=knn,
        num_sample=num_sample,
        sample_method=sample_method,
        chunk_size=chunk_size,
        device=device,
        use_tqdm=use_tqdm,
    )
