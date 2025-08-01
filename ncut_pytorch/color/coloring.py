__all__ = ["mspace_color", "tsne_color", "umap_color", "umap_sphere_color", "rotate_rgb_cube", "convert_to_lab_color"]

import warnings
from typing import Any, Callable, Dict, Literal, Tuple, Optional

import numpy as np
import torch

from .mspace import mspace_viz_transform
from ncut_pytorch.ncuts.ncut_nystrom import nystrom_propagate
from ncut_pytorch.utils.math import quantile_normalize
from ncut_pytorch.utils.sample import farthest_point_sampling


def _identity(X: torch.Tensor) -> torch.Tensor:
    return X


def mspace_color(
        X: torch.Tensor,
        q: float = 0.95,
        n_eig: Optional[int] = 32,
        n_dim: int = 3,
        training_steps: int = 100,
        progress_bar: bool = False,
        **kwargs: Any,
):
    """
    Returns:
        (torch.Tensor): Embedding in 2D, shape (n_samples, 2)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """

    low_dim_embedding = mspace_viz_transform(
        X=X,
        n_eig=n_eig,
        mood_dim=n_dim,
        training_steps=training_steps,
        progress_bar=progress_bar,
        eigvec_loss=1.0,
        recon_loss=0.0,
        decoder_training_steps=0,
        boundary_loss=100.,
        zero_center_loss=0.0,
        repulsion_loss=1.0,
        attraction_loss=100.,
        axis_align_loss=100.,
        degree=['auto'],
        **kwargs)

    rgb = rgb_from_nd_colormap(low_dim_embedding, q=q)

    return rgb


def tsne_color(
        X: torch.Tensor,
        num_sample: int = 1000,
        perplexity: int = 150,
        n_dim: int = 2,
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
    num_sample = min(num_sample, X.shape[0])
    if perplexity > num_sample // 2:
        warnings.warn(
            f"perplexity is larger than num_sample, set perplexity to {num_sample // 2}",
            stacklevel=2,
            category=UserWarning,
        )
        perplexity = num_sample // 2

    low_dim_embedding, rgb = _nystrom_dimension_reduction(
        X=X,
        num_sample=num_sample,
        metric=metric,
        rgb_func=rgb_from_nd_colormap,
        q=q, knn=knn,
        seed=seed, device=device,
        reduction=TSNE, reduction_dim=n_dim, reduction_kwargs={
            "perplexity": perplexity,
        },
    )

    return rgb


def umap_color(
        X: torch.Tensor,
        num_sample: int = 1000,
        n_neighbors: int = 150,
        min_dist: float = 0.1,
        n_dim: int = 2,
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

    low_dim_embedding, rgb = _nystrom_dimension_reduction(
        X=X,
        num_sample=num_sample,
        metric=metric,
        rgb_func=rgb_from_nd_colormap,
        q=q, knn=knn,
        seed=seed, device=device,
        reduction=UMAP, reduction_dim=n_dim, reduction_kwargs={
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
        },
    )

    return rgb


def umap_sphere_color(
        X: torch.Tensor,
        num_sample: int = 1000,
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

    low_dim_embedding, rgb = _nystrom_dimension_reduction(
        X=X,
        num_sample=num_sample,
        metric=metric,
        rgb_func=rgb_from_nd_colormap,
        q=q, knn=knn,
        seed=seed, device=device,
        reduction=UMAP, reduction_dim=2, reduction_kwargs={
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "output_metric": "haversine",
        },
        transform_func=transform_func
    )

    return rgb


def _nystrom_dimension_reduction(
        X: torch.Tensor,
        num_sample: int,
        metric: Literal["cosine", "euclidean"],
        rgb_func: Callable[[torch.Tensor, float], torch.Tensor],
        q: float,
        knn: int,
        seed: int,
        device: str,
        reduction: Callable[..., "sklearn.base.BaseEstimator"],
        reduction_dim: int,
        reduction_kwargs: Dict[str, Any],
        transform_func: Callable[[torch.Tensor], torch.Tensor] = _identity,
) -> Tuple[torch.Tensor, torch.Tensor]:
    subgraph_indices = farthest_point_sampling(X, n_sample=num_sample, device=device)

    _inp = X[subgraph_indices].cpu().numpy()
    _subgraph_embed = reduction(
        n_components=reduction_dim,
        metric=metric,
        random_state=seed,
        **reduction_kwargs
    ).fit_transform(_inp)

    _subgraph_embed = torch.tensor(_subgraph_embed, dtype=torch.float32)
    X_nd = transform_func(nystrom_propagate(
        _subgraph_embed,
        X,
        X[subgraph_indices],
        n_neighbors=knn,
        device=device,
        move_output_to_cpu=True,
    ))
    rgb = rgb_func(X_nd, q)
    return X_nd, rgb


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
    x = (xy[:, 0] * (len_x - 1)).astype(np.int64)
    y = (xy[:, 1] * (len_y - 1)).astype(np.int64)
    x = np.clip(x, 0, len_x - 1)
    y = np.clip(y, 0, len_y - 1)
    rgb = cmap._cmap_data[x, y]
    rgb = torch.tensor(rgb, dtype=torch.float32) / 255
    return rgb


def rgb_from_nd_colormap(X_nd, q=0.95):
    """
    Returns:
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    d = X_nd.shape[1]
    if d == 2:
        return rgb_from_2d_colormap(X_nd, q=q)
    elif d == 3:
        return rgb_from_3d_rgb_cube(X_nd, q=q)
    else:
        raise ValueError(f"Unsupported dimensionality: {d}")


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
