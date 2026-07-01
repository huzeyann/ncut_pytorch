import torch
from ncut_pytorch.utils.math import rbf_affinity, normalize_affinity

@torch.no_grad()
def _ritz_extrapolate(
        points,
        extrapolate_points,
        evals,
        evecs,
        degrees,
        n_eig=64,  # 增大 n_eig 可以提高精度
        p_subspace=128,  # p1: T_x 子空间大小
        p_spectral=256,  # p2: 近似 S ≈ U2 Lambda2 U2^T 的谱基大小
        gamma=1.0,
        chunk_size=512,
        self_affinity=1.0,
        eps=1e-8,
):
    device = points.device
    dtype = points.dtype

    idx_all = torch.argsort(evals, descending=True)

    # U1: Ritz / projected T_x subspace.
    idx_sub = idx_all[:p_subspace]
    U1 = evecs[:, idx_sub]  # [n, p1]

    # U2: spectral approximation basis for S.
    idx_spec = idx_all[:p_spectral]
    lam2 = evals[idx_spec]  # [p2]
    U2 = evecs[:, idx_spec]  # [n, p2]

    p1 = U1.shape[1]
    p2 = U2.shape[1]

    outs = []

    for start in range(0, len(extrapolate_points), chunk_size):
        x = extrapolate_points[start:start + chunk_size]
        bsz = x.shape[0]

        K = rbf_affinity(x, points, gamma=gamma)  # [b, n]
        d_x = (self_affinity + K.sum(dim=1)).clamp_min(eps)  # [b]
        d_old_aug = (degrees[None, :] + K).clamp_min(eps)  # [b, n]

        alpha = self_affinity / d_x  # [b]
        beta = K / torch.sqrt(d_x[:, None] * d_old_aug)  # [b, n]

        R = torch.sqrt(degrees[None, :] / d_old_aug)  # [b, n]
        z = beta @ U1  # [b, p1]
        C = torch.einsum("nr,bn,nq->brq", U2, R, U1)  # [b, p2, p1]
        M = torch.matmul(
            C.transpose(1, 2),  # [b, p1, p2]
            lam2[None, :, None] * C  # [b, p2, p1]
        )  # [b, p1, p1]
        M = 0.5 * (M + M.transpose(1, 2))

        # Build T_x.
        T = torch.zeros((bsz, p1 + 1, p1 + 1), device=device, dtype=dtype)
        T[:, 0, 0] = alpha
        T[:, 0, 1:] = z
        T[:, 1:, 0] = z
        T[:, 1:, 1:] = M

        theta, Qsmall = torch.linalg.eigh(T)
        order = torch.argsort(theta, dim=1, descending=True)[:, :n_eig]
        Qsel = torch.gather(
            Qsmall,
            dim=2,
            index=order[:, None, :].expand(-1, p1 + 1, -1),
        )

        # New-node coordinate.
        y = Qsel[:, 0, :]  # [b, n_eig]
        # Old-node coefficients in U1 basis.
        coeffs = Qsel[:, 1:, :]  # [b, p1, n_eig]

        # Sign alignment:
        diag_coeffs = torch.diagonal(coeffs, dim1=1, dim2=2)  # [b, n_eig]
        signs = torch.where(
            diag_coeffs >= 0,
            torch.ones_like(diag_coeffs),
            -torch.ones_like(diag_coeffs),
        )

        outs.append(y * signs)

    return torch.cat(outs, dim=0)


def _check_args(n, n_eig, p_subspace, p_spectral):
    if p_subspace is None:
        p_subspace = n_eig * 2
    if p_spectral is None:
        p_spectral = p_subspace * 2
    p_subspace = max(p_subspace, 2 * n_eig)
    p_spectral = max(p_spectral, 4 * n_eig)
    p_subspace = min(p_subspace, n)
    p_spectral = min(p_spectral, n)
    if p_subspace < n_eig or p_spectral < n_eig:
        raise ValueError(
            f"p_subspace and p_spectral must be >= n_eig for output/sign alignment. "
            f"Got p_subspace={p_subspace}, p_spectral={p_spectral} n_eig={n_eig}."
        )
    return p_subspace, p_spectral


@torch.no_grad()
def ritz_extrapolate(
        points,
        extrapolate_points,
        n_eig=64,  # 增大 n_eig 可以提高精度
        p_subspace=128,  # p1: T_x 子空间大小
        p_spectral=256,  # p2: 近似 S ≈ U2 Lambda2 U2^T 的谱基大小
        gamma=1.0,
        chunk_size=512,
        self_affinity=1.0,
        eps=1e-8,
):
    """Rizt Approximation
    args:
        points: [n, d]
        extrapolate_points: [N, d]
    return:
        extrapolate_eigvec: [N, n_eig]
    """

    n = points.shape[0]
    p_subspace, p_spectral = _check_args(n, n_eig, p_subspace, p_spectral)

    W = rbf_affinity(points, points, gamma=gamma)
    degrees = W.sum(dim=1).clamp_min(eps)
    S = normalize_affinity(W)
    evals, evecs = torch.linalg.eigh(S)  # 担心evecs会有sign flipping，不同次call结果会不一样

    out_eigvec = _ritz_extrapolate(
        points,
        extrapolate_points,
        evals,
        evecs,
        degrees,
        n_eig,
        p_subspace,
        p_spectral,
        gamma,
        chunk_size,
        self_affinity,
        eps
    )

    # return out_eigvec, evals, evecs, degrees
    return out_eigvec