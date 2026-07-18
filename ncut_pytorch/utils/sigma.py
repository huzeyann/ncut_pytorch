__all__ = ["find_sigma", "find_sigma_by_degree"]

import math
import warnings

import torch

from .math import rbf_affinity
from .sample import farthest_point_sampling

_DEPRECATION_WARNED = False


@torch.no_grad()
def _find_sigma(
    anchor: torch.Tensor,
    n_sample: int = 1000,
    A_degree: float | None = None,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> float:
    """Find sigma for RBF affinity.

    Base sigma = median pairwise distance (FPS subsample -> cdist -> median).
    At median, mean(W) ~= 0.61, which already gives good dynamic range.

    If A_degree is given (e.g. 0.5), refines via binary search
    starting from median so that mean(W) = A_degree. The search
    is optional because the data shows median alone gives nearly
    identical spectral gap and W std.
    """
    if A_degree is not None and (A_degree <= 0 or A_degree >= 1):
        raise ValueError(f"A_degree must be between 0 and 1, got {A_degree}")

    n = anchor.shape[0]
    if n > n_sample:
        indices = farthest_point_sampling(anchor, n_sample)
        sub = anchor[indices]
    else:
        sub = anchor

    dist = torch.cdist(sub, sub, p=2)
    triu = torch.triu_indices(dist.shape[0], dist.shape[0], offset=1)
    pairwise = dist[triu[0], triu[1]]

    if pairwise.numel() == 0:
        return 1.0

    sigma = pairwise.median().item()

    if A_degree is None:
        return sigma

    dist_nonzero = dist[dist > 0]
    log_lo = math.log(dist_nonzero.min().item()) - 1.0
    log_hi = math.log(dist_nonzero.max().item()) + 1.0
    log_lo = min(log_lo, math.log(sigma))
    log_hi = max(log_hi, math.log(sigma))

    for _ in range(max_iter):
        log_sigma = (log_lo + log_hi) / 2
        sigma = math.exp(log_sigma)
        W = rbf_affinity(sub, sub, sigma=sigma)
        mean_aff = W.mean().item()

        if abs(mean_aff - A_degree) < tol:
            break
        if mean_aff > A_degree:
            log_hi = log_sigma
        else:
            log_lo = log_sigma

    return sigma


@torch.no_grad()
def find_sigma(
    anchor: torch.Tensor,
    n_sample: int = 1000,
    A_degree: float | None = None,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> float:
    """Find sigma for RBF affinity (public entry point).

    See ``_find_sigma`` for details. ``A_degree`` is the desired mean
    affinity in (0, 1); when ``None``, returns the median pairwise distance.
    """
    return _find_sigma(
        anchor,
        n_sample=n_sample,
        A_degree=A_degree,
        max_iter=max_iter,
        tol=tol,
    )


@torch.no_grad()
def find_sigma_by_degree(
    X: torch.Tensor,                    # [n_samples, n_features]
    quantile_sigma: float = 0.25,
    affinity_fn: callable = rbf_affinity,
    X2: torch.Tensor | None = None,
    init_sigma: float = 0.5,
    r_tol: float = 1e-2,
    max_iter: int = 100,
    n_sample: int = 1000,
) -> float:
    """Deprecated. Use :func:`find_sigma` instead.

    ``quantile_sigma`` (a quantile of per-node degrees) is mapped to
    ``A_degree`` (a target mean affinity). Note the semantics differ;
    this alias only preserves the parameter name for call-site compatibility.
    """
    global _DEPRECATION_WARNED
    if not _DEPRECATION_WARNED:
        warnings.warn(
            "find_sigma_by_degree is deprecated; use find_sigma instead. "
            "quantile_sigma is mapped to A_degree (semantics changed "
            "from a quantile of per-node degrees to a target mean affinity).",
            DeprecationWarning,
            stacklevel=2,
        )
        _DEPRECATION_WARNED = True
    return find_sigma(X, n_sample=n_sample, A_degree=quantile_sigma, max_iter=max_iter)


def _pop_deprecated_sigma_kwargs(kwargs: dict) -> tuple[float | None, float | None]:
    """Pop deprecated quantile_sigma / quantile_sigma_repulsion from kwargs.

    Returns (A_degree, A_rep_degree) where each entry is
    ``None`` if the corresponding deprecated key was absent.
    """
    global _DEPRECATION_WARNED
    A_degree: float | None = None
    A_rep_degree: float | None = None
    if "quantile_sigma" in kwargs:
        if not _DEPRECATION_WARNED:
            warnings.warn(
                "quantile_sigma is deprecated; use A_degree instead. "
                "Semantics changed from a quantile of per-node degrees to a target mean affinity.",
                DeprecationWarning,
                stacklevel=2,
            )
            _DEPRECATION_WARNED = True
        A_degree = kwargs.pop("quantile_sigma")
    if "quantile_sigma_repulsion" in kwargs:
        if not _DEPRECATION_WARNED:
            warnings.warn(
                "quantile_sigma_repulsion is deprecated; use A_rep_degree instead. "
                "Semantics changed from a quantile of per-node degrees to a target mean affinity.",
                DeprecationWarning,
                stacklevel=2,
            )
            _DEPRECATION_WARNED = True
        A_rep_degree = kwargs.pop("quantile_sigma_repulsion")
    return A_degree, A_rep_degree
