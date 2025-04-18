import logging
from typing import Any, Callable, Dict, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator

from .nystrom_utils import (
    run_subgraph_sampling,
    propagate_knn,
    which_device,
)
from .math_utils import (
    quantile_normalize,
    quantile_min_max,
    check_if_normalized,
)


def _identity(X: torch.Tensor) -> torch.Tensor:
    return X


def _rgb_with_dimensionality_reduction(
    features: torch.Tensor,
    num_sample: int,
    metric: Literal["cosine", "euclidean"],
    rgb_func: Callable[[torch.Tensor, float], torch.Tensor],
    q: float, 
    knn: int,
    seed: int, 
    device: str,
    reduction: Callable[..., BaseEstimator],
    reduction_dim: int,
    reduction_kwargs: Dict[str, Any],
    transform_func: Callable[[torch.Tensor], torch.Tensor] = _identity,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def rgb_from_mspace_2d(
    features: torch.Tensor,
    q: float = 0.95,
    n_eig: int = 32,
    training_steps: int = 500,
    progress_bar: bool = True,
    **kwargs: Any,
):
    from .mspace import mspace_viz_transform
    """
    Returns:
        (torch.Tensor): Embedding in 2D, shape (n_samples, 2)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """

    x2d = mspace_viz_transform(features, n_eig=n_eig, mood_dim=2, training_steps=training_steps, progress_bar=progress_bar, **kwargs)

    rgb = rgb_from_2d_colormap(x2d, q=q)

    return x2d, rgb


def rgb_from_mspace_3d(
    features: torch.Tensor,
    q: float = 0.95,
    n_eig: int = 32,
    training_steps: int = 500,
    progress_bar: bool = True,
    **kwargs: Any,
):
    from .mspace import mspace_viz_transform
    """
    Returns:
        (torch.Tensor): Embedding in 3D, shape (n_samples, 3)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """

    x3d = mspace_viz_transform(features, n_eig=n_eig, mood_dim=3, training_steps=training_steps, progress_bar=progress_bar, **kwargs)

    rgb = rgb_from_3d_rgb_cube(x3d, q=q)

    return x3d, rgb



def rgb_from_tsne_2d(
    features: torch.Tensor,
    num_sample: int = 1000,
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

    x2d, rgb = _rgb_with_dimensionality_reduction(
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
    
    return x2d, rgb


def rgb_from_tsne_3d(
    features: torch.Tensor,
    num_sample: int = 1000,
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

    x3d, rgb = _rgb_with_dimensionality_reduction(
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

    return x3d, rgb


def rgb_from_cosine_tsne_3d(
    features: torch.Tensor,
    num_sample: int = 1000,
    perplexity: int = 150,
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


    def cosine_to_rbf(X: torch.Tensor) -> torch.Tensor:                                 # [B... x N x 3]
        normalized_X = X / torch.norm(X, p=2, dim=-1, keepdim=True)                     # [B... x N x 3]
        D = 1 - normalized_X @ normalized_X.mT                                          # [B... x N x N]

        G = (D[..., :1, 1:] ** 2 + D[..., 1:, :1] ** 2 - D[..., 1:, 1:] ** 2) / 2       # [B... x (N - 1) x (N - 1)]
        L, V = torch.linalg.eigh(G)                                                     # [B... x (N - 1)], [B... x (N - 1) x (N - 1)]
        sqrtG = V[..., -3:] * (L[..., None, -3:] ** 0.5)                                # [B... x (N - 1) x 3]

        Y = torch.cat((torch.zeros_like(sqrtG[..., :1, :]), sqrtG), dim=-2)             # [B... x N x 3]
        Y = Y - torch.mean(Y, dim=-2, keepdim=True)
        return Y

    def rgb_from_cosine(X_3d: torch.Tensor, q: float) -> torch.Tensor:
        return rgb_from_3d_rgb_cube(cosine_to_rbf(X_3d), q=q)

    x3d, rgb = _rgb_with_dimensionality_reduction(
        features=features,
        num_sample=num_sample,
        metric="cosine",
        rgb_func=rgb_from_cosine,
        q=q, knn=knn,
        seed=seed, device=device,
        reduction=TSNE, reduction_dim=3, reduction_kwargs={
            "perplexity": perplexity,
        },
    )
    
    return x3d, rgb


def rgb_from_umap_2d(
    features: torch.Tensor,
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

    x2d, rgb = _rgb_with_dimensionality_reduction(
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
    
    return x2d, rgb


def rgb_from_umap_sphere(
    features: torch.Tensor,
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

    x3d, rgb = _rgb_with_dimensionality_reduction(
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
    
    return x3d, rgb


def rgb_from_umap_3d(
    features: torch.Tensor,
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

    x3d, rgb = _rgb_with_dimensionality_reduction(
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
    
    return x3d, rgb


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
    x = (xy[:, 0] * (len_x - 1)).astype(np.int64)
    y = (xy[:, 1] * (len_y - 1)).astype(np.int64)
    x = np.clip(x, 0, len_x - 1)
    y = np.clip(y, 0, len_y - 1)
    rgb = cmap._cmap_data[x, y]
    rgb = torch.tensor(rgb, dtype=torch.float32) / 255
    return rgb

def rgb_from_nd_colormap(X_nd, q=0.95, lab_color=False):
    """
    Returns:
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    d = X_nd.shape[1]
    if d == 2:
        return rgb_from_2d_colormap(X_nd, q=q)
    elif d == 3:
        rgb = rgb_from_3d_rgb_cube(X_nd, q=q)
        if lab_color:
            rgb = convert_to_lab_color(rgb)
            rgb = torch.from_numpy(rgb)
        return rgb
    else:
        raise ValueError(f"Unsupported dimensionality: {d}")
