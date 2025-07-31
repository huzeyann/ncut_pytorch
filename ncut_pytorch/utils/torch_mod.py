# Mod from torch/_lowrank.py
# because QR and SVD is not implemented for MPS (apple silicon)

# TODO: remove this file after torch have updated MPS support

"""Implement various linear algebra algorithms for low rank matrices."""
from typing import Optional

import torch
from torch import _linalg_utils as _utils, Tensor
from torch.overrides import handle_torch_function, has_torch_function


def qr_fallback_cpu(X: torch.Tensor):
    device = X.device
    if device.type == 'mps':
        X = X.cpu()
        Q = torch.linalg.qr(X).Q
        Q = Q.to(device)
        return Q
    return torch.linalg.qr(X).Q


def svd_fallback_cpu(X: torch.Tensor):
    device = X.device
    if device.type == 'mps':
        X = X.cpu()
        U, S, V = torch.linalg.svd(X, full_matrices=False)
        U = U.to(device)
        S = S.to(device)
        V = V.to(device)
        return U, S, V
    return torch.linalg.svd(X, full_matrices=False)


def get_approximate_basis(
        A: Tensor,
        q: int,
        niter: Optional[int] = 2,
        M: Optional[Tensor] = None,
) -> Tensor:
    """Return tensor :math:`Q` with :math:`q` orthonormal columns such
    that :math:`Q Q^H A` approximates :math:`A`. If :math:`M` is
    specified, then :math:`Q` is such that :math:`Q Q^H (A - M)`
    approximates :math:`A - M`. without instantiating any tensors
    of the size of :math:`A` or :math:`M`.

    .. note:: The implementation is based on the Algorithm 4.4 from
              Halko et al., 2009.

    .. note:: For an adequate approximation of a k-rank matrix
              :math:`A`, where k is not known in advance but could be
              estimated, the number of :math:`Q` columns, q, can be
              chosen according to the following criteria: in general,
              :math:`k <= q <= min(2*k, m, n)`. For large low-rank
              matrices, take :math:`q = k + 5..10`.  If k is
              relatively small compared to :math:`min(m, n)`, choosing
              :math:`q = k + 0..2` may be sufficient.

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Args::
        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int): the dimension of subspace spanned by :math:`Q`
                 columns.

        niter (int, optional): the number of subspace iterations to
                               conduct; ``niter`` must be a
                               nonnegative integer. In most cases, the
                               default value 2 is more than enough.

        M (Tensor, optional): the input tensor's mean of size
                              :math:`(*, m, n)`.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """

    niter = 2 if niter is None else niter
    dtype = _utils.get_floating_dtype(A) if not A.is_complex() else A.dtype
    matmul = _utils.matmul

    R = torch.randn(A.shape[-1], q, dtype=dtype, device=A.device)

    # The following code could be made faster using torch.geqrf + torch.ormqr
    # but geqrf is not differentiable

    X = matmul(A, R)
    if M is not None:
        X = X - matmul(M, R)
    Q = qr_fallback_cpu(X)
    for _ in range(niter):
        X = matmul(A.mH, Q)
        if M is not None:
            X = X - matmul(M.mH, Q)
        Q = qr_fallback_cpu(X)
        X = matmul(A, Q)
        if M is not None:
            X = X - matmul(M, Q)
        Q = qr_fallback_cpu(X)
    return Q


def svd_lowrank(
        A: Tensor,
        q: Optional[int] = 6,
        niter: Optional[int] = 2,
        M: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Return the singular value decomposition ``(U, S, V)`` of a matrix,
    batches of matrices, or a sparse matrix :math:`A` such that
    :math:`A \approx U \operatorname{diag}(S) V^{\text{H}}`. In case :math:`M` is given, then
    SVD is computed for the matrix :math:`A - M`.

    .. note:: The implementation is based on the Algorithm 5.1 from
              Halko et al., 2009.

    .. note:: For an adequate approximation of a k-rank matrix
              :math:`A`, where k is not known in advance but could be
              estimated, the number of :math:`Q` columns, q, can be
              chosen according to the following criteria: in general,
              :math:`k <= q <= min(2*k, m, n)`. For large low-rank
              matrices, take :math:`q = k + 5..10`.  If k is
              relatively small compared to :math:`min(m, n)`, choosing
              :math:`q = k + 0..2` may be sufficient.

    .. note:: This is a randomized method. To obtain repeatable results,
              set the seed for the pseudorandom number generator

    .. note:: In general, use the full-rank SVD implementation
              :func:`torch.linalg.svd` for dense matrices due to its 10x
              higher performance characteristics. The low-rank SVD
              will be useful for huge sparse matrices that
              :func:`torch.linalg.svd` cannot handle.

    Args::
        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int, optional): a slightly overestimated rank of A.

        niter (int, optional): the number of subspace iterations to
                               conduct; niter must be a nonnegative
                               integer, and defaults to 2

        M (Tensor, optional): the input tensor's mean of size
                              :math:`(*, m, n)`, which will be broadcasted
                              to the size of A in this function.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <https://arxiv.org/abs/0909.4061>`_).

    """
    if not torch.jit.is_scripting():
        tensor_ops = (A, M)
        if not set(map(type, tensor_ops)).issubset(
                (torch.Tensor, type(None))
        ) and has_torch_function(tensor_ops):
            return handle_torch_function(
                svd_lowrank, tensor_ops, A, q=q, niter=niter, M=M
            )
    return _svd_lowrank(A, q=q, niter=niter, M=M)


def _svd_lowrank(
        A: Tensor,
        q: Optional[int] = 6,
        niter: Optional[int] = 2,
        M: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    # Algorithm 5.1 in Halko et al., 2009

    q = 6 if q is None else q
    m, n = A.shape[-2:]
    matmul = _utils.matmul
    if M is not None:
        M = M.broadcast_to(A.size())

    # Assume that A is tall
    if m < n:
        A = A.mH
        if M is not None:
            M = M.mH

    Q = get_approximate_basis(A, q, niter=niter, M=M)
    B = matmul(Q.mH, A)
    if M is not None:
        B = B - matmul(Q.mH, M)
    U, S, Vh = svd_fallback_cpu(B)
    V = Vh.mH
    U = Q.matmul(U)

    if m < n:
        U, V = V, U

    return U, S, V
