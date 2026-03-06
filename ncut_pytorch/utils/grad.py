__all__ = ["rbf_eigvec_manual_grad", "grad_manager"]

import torch
from contextlib import contextmanager


@torch.no_grad()
def rbf_eigvec_manual_grad(
    X: torch.Tensor,          # [N,D]
    W: torch.Tensor,          # [N,N]  (RBF affinity, e.g. W_ij = exp(-||xi-xj||^2 / (2 * sigma^2)))
    U: torch.Tensor,          # [N,K]  eigenvectors of A = S W S (K can be k or N)
    evals: torch.Tensor,      # [K]    eigenvalues corresponding to U (same ordering)
    sigma: float,             # RBF sigma in exp(-||xi-xj||^2 / (2 * sigma^2))
    r: int = 1,               # target eigenvector index in U (assumed "largest-first" if you pass top-k)
    I: torch.Tensor | None = None,   # [m] indices; if None -> all points
    eps: float = 1e-12,
    squared_objective: bool = True,
):
    """
    Computes grad[t] = ∂ u_r[i] / ∂ x_i  (a D-vector), for i = I[t],
    where u_r is the r-th eigenvector of A = S W S with S = diag(D^{-1/2}), D = W 1.

    If squared_objective=True, returns gradient of (u_r[i])^2 instead:
        ∂(u_r[i]^2)/∂x_i = 2 u_r[i] * ∂u_r[i]/∂x_i

    Assumptions:
      - W is symmetric, nonnegative.
      - U, evals are eigenpairs of A = S W S (NOT of W).
      - U columns are M-orthonormal in the generalized derivation; here A is symmetric so standard orthonormal.
      - evals and U must be aligned (same ordering). If you pass top-k (largest-first), r is in that indexing.
    """
    assert X.ndim == 2
    assert W.ndim == 2 and W.shape[0] == W.shape[1] == X.shape[0]
    assert U.ndim == 2 and U.shape[0] == X.shape[0]
    assert evals.ndim == 1 and evals.shape[0] == U.shape[1]
    assert evals.shape[0] == X.shape[0], "all eigenvalues must be provided"

    N, Ddim = X.shape
    K = U.shape[1]
    assert 0 <= r < K

    device, dtype = X.device, X.dtype
    W = W.to(device=device, dtype=dtype)
    U = U.to(device=device, dtype=dtype)
    evals = evals.to(device=device, dtype=dtype)

    if I is None:
        I = torch.arange(N, device=device, dtype=torch.long)
    else:
        I = I.to(device=device, dtype=torch.long)
    m = I.numel()

    # ---- symmetric normalization A = S W S, S=diag(s), s = D^{-1/2} ----
    Deg = W.sum(dim=1).clamp_min(eps)          # [N]
    s = torch.rsqrt(Deg)                       # [N]
    s3 = s**3                                  # [N]

    lam_r = evals[r]
    u = U[:, r]                                # [N]

    # precompute
    z = s * u                                  # [N]
    Wz = W @ z                                 # [N]

    denom = (lam_r - evals).clone()            # [K]
    denom[r] = torch.inf

    UI = U[I, :]                               # [m,K]
    uI = u[I]                                  # [m]
    zI = z[I]                                  # [m]

    # ---- vectorized over d ----
    # diff[t,j,d] = x_i - x_j for i=I[t]
    diff = X[I, None, :] - X[None, :, :]       # [m,N,D]

    # P[t,j,d] = ∂W_{i,j} / ∂x_{i,d}, for W_ij = exp(-||xi-xj||^2 / (2 * sigma^2))
    # d/dxi exp(-||xi-xj||^2 / (2 * sigma^2)) = exp(-...)*(-2/sigma^2)*(xi-xj)
    P = -(diff / (sigma * sigma)) * W[I, :, None] # [m,N,D]

    # dD[t,j,d] = ∂Deg_j / ∂x_{i,d}
    rs = P.sum(dim=1)                          # [m,D] = sum_j ∂W_{i,j}/∂x_i
    dDeg = P.clone()                           # start with only row i contributions
    rows = torch.arange(m, device=device)
    dDeg[rows, I, :] = rs                      # add the "column i" contribution to Deg_i

    # ds[t,j,d] = ∂s_j / ∂x_{i,d} = -1/2 * s_j^3 * ∂Deg_j/∂x_{i,d}
    ds = -0.5 * dDeg * s3[None, :, None]       # [m,N,D]

    # term1: dS * (W (S u))
    term1 = ds * Wz[None, :, None]             # [m,N,D]

    # term2: S * (dW (S u))
    # q[t,d] = sum_j P[t,j,d] * z[j]  (row i contribution)
    q = torch.einsum('mnd,n->md', P, z)         # [m,D]
    dWz = P * zI[:, None, None]                 # [m,N,D] (column i contribution)
    dWz[rows, I, :] = q                         # set j=i entries
    term2 = dWz * s[None, :, None]              # [m,N,D]

    # term3: S * (W (dS u))
    tmp = ds * u[None, :, None]                 # [m,N,D]
    Wu = torch.einsum('jk,mkd->mjd', W, tmp)    # [m,N,D]
    term3 = Wu * s[None, :, None]               # [m,N,D]

    dAu = term1 + term2 + term3                 # [m,N,D]

    # beta[t,s,d] = u_s^T (dAu[t,:,d])
    beta = torch.einsum('mnd,ns->msd', dAu, U)  # [m,K,D]
    coeff = beta / denom[None, :, None]         # [m,K,D]

    # grad_u[t,d] = sum_{s!=r} u_s[i] * coeff[t,s,d]
    grad_u = torch.einsum('ms,msd->md', UI, coeff)  # [m,D]

    if squared_objective:
        grad_u = 2.0 * uI[:, None] * grad_u

    return grad_u



@contextmanager
def grad_manager(enabled: bool):
    """Context manager to temporarily set gradient computation mode.
    
    This context manager allows you to control gradient computation for a block
    of code, and automatically restores the previous gradient state when exiting
    the context.
    
    Args:
        enabled (bool): If True, enables gradient tracking within the context.
                        If False, disables gradient tracking within the context.
    
    Yields:
        None
        
    Examples:
        >>> import torch
        >>> from ncut_pytorch.utils.grad import set_grad_enabled
        >>> 
        >>> # Disable gradients for inference
        >>> with set_grad_enabled(False):
        ...     result = model(input_tensor)
        >>> 
        >>> # Enable gradients for training
        >>> with set_grad_enabled(True):
        ...     loss = criterion(model(input_tensor), target)
        ...     loss.backward()
    """
    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(enabled)
    try:
        yield
    finally:
        torch.set_grad_enabled(prev_grad_state)

