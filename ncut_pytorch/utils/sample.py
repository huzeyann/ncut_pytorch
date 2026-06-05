__all__ = ["farthest_point_sampling"]

import numpy as np
import torch
import warnings

from .device import auto_device
from .math import pca_lowrank, random_orthogonal_projection

_TORCH_QUICKFPS_IMPORT_ERROR: Exception | None = None
try:
    from torch_quickfps import sample_idx as _torch_quickfps_sample_idx

    _TORCH_QUICKFPS_OP = "torch_quickfps::_sample_idx_impl"
    _HAS_CUDA_KERNEL = torch._C._dispatch_has_kernel_for_dispatch_key(_TORCH_QUICKFPS_OP, "CUDA")
    _HAS_TORCH_QUICKFPS = True
    sample_idx = _torch_quickfps_sample_idx
except Exception as exc:
    _torch_quickfps_sample_idx = None
    _HAS_CUDA_KERNEL = False
    _HAS_TORCH_QUICKFPS = False
    _TORCH_QUICKFPS_IMPORT_ERROR = exc
    sample_idx = None

_FPSAMPLE_IMPORT_ERROR: Exception | None = None
try:
    import fpsample as _fpsample

    _HAS_FPSAMPLE = True
except Exception as exc:
    _fpsample = None
    _HAS_FPSAMPLE = False
    _FPSAMPLE_IMPORT_ERROR = exc

_HAS_FPSAMPLE_BUCKET_FPS = _HAS_FPSAMPLE and hasattr(_fpsample, "bucket_fps_kdline_sampling")
_HAS_FPSAMPLE_KDTREE_FPS = _HAS_FPSAMPLE and hasattr(_fpsample, "fps_npdu_kdtree_sampling")
_WARNED_ABOUT_LEGACY_FPSAMPLE = False

_DEFAULT_MAX_DRAW_RATIO = 4.0
_DEFAULT_FPSAMPLE_H = 7
# Edit this variable in code to switch the FPS pre-projection method.
FPS_DIMENSION_REDUCTION_METHOD = "random_orthogonal"


def _reduce_fps_dimension(
    X: torch.Tensor,
    *,
    max_dim: int,
) -> torch.Tensor:
    reduction_method = FPS_DIMENSION_REDUCTION_METHOD
    if X.shape[1] <= max_dim:
        return X

    if reduction_method == "random_orthogonal":
        return random_orthogonal_projection(X, q=max_dim)

    if reduction_method == "pca":
        return pca_lowrank(X, q=max_dim)

    raise ValueError(
        "Unsupported FPS reduction method: "
        f"{reduction_method!r}. Expected 'random_orthogonal' or 'pca'."
    )


def _stratified_presample_indices(
    num_data: int,
    num_draw: int,
) -> torch.Tensor:
    """Draw evenly spread candidate indices without materializing randperm(num_data)."""
    if num_draw >= num_data:
        return torch.arange(num_data)

    step = num_data / num_draw
    offset = torch.rand((), dtype=torch.float64) * step
    base = torch.arange(num_draw, dtype=torch.float64)
    draw_indices = torch.floor(offset + base * step).to(torch.long)
    draw_indices.clamp_(max=num_data - 1)
    return draw_indices


def _prepare_fps_input(
    X: torch.Tensor | np.ndarray,
    *,
    max_dim: int,
    device: str | None,
    force_cpu: bool = False,
) -> torch.Tensor:
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    target_device = "cpu" if force_cpu else auto_device(X.device, device)
    X = X.to(target_device)

    X = _reduce_fps_dimension(
        X,
        max_dim=max_dim,
    )

    assert X.ndim == 2, "X should be a 2D tensor"
    assert X.shape[0] > 0, "X should have at least 1 data point"
    assert X.shape[1] > 0, "X should have at least 1 dimension"
    assert not torch.any(torch.isnan(X)), "X contains NaN"
    assert not torch.any(torch.isinf(X)), "X contains Inf"

    return X


def _raise_missing_fps_backend() -> None:
    message = (
        "No FPS backend available. Install `fpsample>=0.2.0`. "
        "`torch_quickfps` is optional and only used when available."
    )
    if _FPSAMPLE_IMPORT_ERROR is not None:
        message = f"{message} fpsample import error: {_FPSAMPLE_IMPORT_ERROR!r}"
    raise ImportError(message) from (_FPSAMPLE_IMPORT_ERROR or _TORCH_QUICKFPS_IMPORT_ERROR)


def _warn_about_legacy_fpsample() -> None:
    global _WARNED_ABOUT_LEGACY_FPSAMPLE
    if _WARNED_ABOUT_LEGACY_FPSAMPLE:
        return

    warnings.warn(
        "package `fpsample` does not provide `bucket_fps_kdline_sampling`; "
        "falling back to the slower `fps_npdu_kdtree_sampling` implementation. "
        "Upgrade `fpsample` for the faster path.",
        RuntimeWarning,
        stacklevel=2,
    )
    _WARNED_ABOUT_LEGACY_FPSAMPLE = True


def _sample_idx_with_torch_quickfps(
    X: torch.Tensor,
    n_sample: int,
    *,
    max_dim: int,
    device: str | None,
) -> torch.Tensor:
    X = _prepare_fps_input(
        X,
        max_dim=max_dim,
        device=device,
        force_cpu=not _HAS_CUDA_KERNEL,
    )
    return sample_idx(X, n_sample).cpu()


def _sample_idx_with_fpsample(
    X: torch.Tensor | np.ndarray,
    n_sample: int,
    *,
    max_dim: int,
    device: str | None,
) -> torch.Tensor:
    if not _HAS_FPSAMPLE:
        _raise_missing_fps_backend()

    X = _prepare_fps_input(
        X,
        max_dim=max_dim,
        device=device,
    )
    X_np = X.cpu().numpy()

    if _HAS_FPSAMPLE_BUCKET_FPS:
        h = min(_DEFAULT_FPSAMPLE_H, max(1, int(np.log2(X_np.shape[0]))))
        samples_idx = _fpsample.bucket_fps_kdline_sampling(X_np, n_sample, h)
    elif _HAS_FPSAMPLE_KDTREE_FPS:
        _warn_about_legacy_fpsample()
        samples_idx = _fpsample.fps_npdu_kdtree_sampling(X_np, n_sample)
    else:
        raise ImportError(
            "Installed `fpsample` does not provide a supported FPS API. "
            "Please install `fpsample>=0.2.0`."
        )

    return torch.as_tensor(np.asarray(samples_idx, dtype=np.int64), dtype=torch.long)


@torch.no_grad()
def farthest_point_sampling(
    X: torch.Tensor,          # [N,D]
    n_sample: int,
    max_draw_ratio: float = _DEFAULT_MAX_DRAW_RATIO,
    max_dim: int = 8,
    device: str | None = None,
) -> torch.Tensor:             # [n_sample]
    """Farthest point sampling with optional stratified pre-sampling for large datasets."""
    num_data = X.shape[0]
    num_draw = min(num_data, max(n_sample, int(n_sample * max_draw_ratio)))

    if num_draw >= num_data:
        return _farthest_point_sampling(
            X,
            n_sample,
            device=device,
            max_dim=max_dim,
        )

    draw_indices = _stratified_presample_indices(num_data, num_draw)
    subset_indices = draw_indices if X.device.type == "cpu" else draw_indices.to(X.device)
    sampled_indices = _farthest_point_sampling(
        X.index_select(0, subset_indices),
        n_sample=n_sample,
        max_dim=max_dim,
        device=device,
    )
    return draw_indices[sampled_indices.cpu()]


@torch.no_grad()
def _farthest_point_sampling(
    X: torch.Tensor,          # [N,D]
    n_sample: int,
    max_dim: int = 8,
    device: str | None = None,
) -> torch.Tensor:             # [n_sample]
    """Internal farthest point sampling implementation with optional torch-quickfps acceleration."""
    num_data = X.shape[0]
    if n_sample >= num_data:
        return torch.arange(num_data)

    if _HAS_TORCH_QUICKFPS:
        return _sample_idx_with_torch_quickfps(
            X,
            n_sample,
            max_dim=max_dim,
            device=device,
        )

    if _HAS_FPSAMPLE:
        return _sample_idx_with_fpsample(
            X,
            n_sample,
            max_dim=max_dim,
            device=device,
        )

    _raise_missing_fps_backend()
