__all__ = ["rbf_eigvec_manual_grad"]

import torch

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


class MultiSpectralProjectorFromMasks(torch.autograd.Function):
    """
    A (symmetric) -> {P_b}_b, where P_b = U_{S_b} U_{S_b}^T and S_b is specified by a boolean mask.

    Computes eigh(A) ONCE, sorts eigenpairs DESCENDING (largest-first), then forms projectors
    for each mask.

    Inputs:
      A:     [N,N] (float), symmetric (or will be symmetrized if symmetrize=True)
      masks: [B,N] (bool), masks[b,i]=True selects eigenvector i in DESCENDING eigen-order.

    Output:
      P: [B,N,N]
    """

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,          # [N,N]
        masks: torch.Tensor,      # [B,N] bool (in DESCENDING eigen-order)
        gap_eps: float = 0.0,
        symmetrize: bool = True,
    ):
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square [N,N], got {tuple(A.shape)}")
        if masks.ndim != 2:
            raise ValueError(f"masks must be [B,N], got {tuple(masks.shape)}")
        if masks.dtype != torch.bool:
            raise ValueError("masks must be boolean")
        N = A.shape[0]
        B, N2 = masks.shape
        if N2 != N:
            raise ValueError(f"masks second dim must equal N={N}, got {N2}")
        if (masks.sum(dim=1) == 0).any():
            raise ValueError("Each mask row must select at least one eigenvector.")

        device = A.device
        masks = masks.to(device=device)

        A_used = 0.5 * (A + A.T) if symmetrize else A

        # eigh ascending -> flip to descending
        evals_asc, U_asc = torch.linalg.eigh(A_used)
        evals = torch.flip(evals_asc, dims=[0])   # [N] descending
        U = torch.flip(U_asc, dims=[1])           # [N,N] descending columns

        # Build projectors
        P_out = []
        for b in range(B):
            U_S = U[:, masks[b]]                  # [N,p_b]
            P_b = U_S @ U_S.T                     # [N,N]
            P_out.append(P_b)
        P = torch.stack(P_out, dim=0)             # [B,N,N]

        ctx.save_for_backward(U, evals, masks)
        ctx.gap_eps = float(gap_eps)
        ctx.symmetrize = bool(symmetrize)

        return P

    @staticmethod
    def backward(ctx, grad_P: torch.Tensor):
        U, evals, masks = ctx.saved_tensors
        gap_eps = ctx.gap_eps
        symmetrize = ctx.symmetrize

        if grad_P.ndim != 3:
            raise ValueError(f"grad_P must be [B,N,N], got {tuple(grad_P.shape)}")
        B, N, N2 = grad_P.shape
        if N != N2:
            raise ValueError("grad_P must be square per batch")

        grad_A_used = torch.zeros((N, N), device=grad_P.device, dtype=grad_P.dtype)

        for b in range(B):
            mask = masks[b]                       # [N]
            U_S = U[:, mask]                      # [N,p]
            U_perp = U[:, ~mask]                  # [N,N-p]
            lam_S = evals[mask]                   # [p]
            lam_perp = evals[~mask]               # [N-p]

            # symmetric part only matters
            G = grad_P[b]
            Gs = 0.5 * (G + G.T)

            # H = U_perp^T Gs U_S
            H = U_perp.T @ (Gs @ U_S)             # [N-p,p]

            denom = lam_S[None, :] - lam_perp[:, None]  # [N-p,p]
            if gap_eps > 0.0:
                denom = torch.sign(denom) * torch.clamp(denom.abs(), min=gap_eps)

            Q = H / denom                          # [N-p,p]

            Bmat = U_perp @ (Q @ U_S.T)            # [N,N]
            grad_A_used = grad_A_used + (Bmat + Bmat.T)

        grad_A = 0.5 * (grad_A_used + grad_A_used.T) if symmetrize else grad_A_used
        return grad_A, None, None, None


def spectral_projectors_from_masks(
    A: torch.Tensor,
    masks: torch.Tensor,
    gap_eps: float = 0.0,
    symmetrize: bool = True,
):
    """
    Convenience wrapper.

    masks: [B,N] bool in DESCENDING eigen-order (0 = largest eigenvalue).
    returns P: [B,N,N]
    """
    return MultiSpectralProjectorFromMasks.apply(A, masks, gap_eps, symmetrize)


if __name__ == "__main__":
    B = 2
    N = 1000
    masks = torch.zeros(B, N, dtype=torch.bool)
    masks[0, :3] = True          # top-3 eigenvectors (largest-first)
    masks[1, 3:6] = True         # next-3


    A1 = torch.randn(N, N)
    A1 = 0.5 * (A1 + A1.T)
    A1.requires_grad_(True)
    P1 = spectral_projectors_from_masks(A1, masks)

    A2 = torch.randn(N, N)
    A2 = 0.5 * (A2 + A2.T)
    A2.requires_grad_(True)
    P2 = spectral_projectors_from_masks(A2, masks)

    loss = torch.norm(P1 - P2, p=2, dim=(0, 1)).sum()
    loss.backward()
    print(A1.grad.shape)    
    print(A2.grad.shape)
