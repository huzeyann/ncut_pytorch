import torch
import numpy as np


def chunked_matmul(
    A,
    B,
    chunk_size=8096,
    device="cuda:0",
    large_device="cpu",
    transform=lambda x: x,
    use_tqdm=True,
):
    A = A.to(large_device)
    B = B.to(large_device)
    C = torch.zeros(A.shape[0], B.shape[1], device=large_device)
    iterator = range(0, A.shape[0], chunk_size)
    for i in iterator:
        end_i = min(i + chunk_size, A.shape[0])
        for j in range(0, B.shape[1], chunk_size):
            end_j = min(j + chunk_size, B.shape[1])
            _A = A[i:end_i]
            _B = B[:, j:end_j]
            _C_ij = None
            for k in range(0, A.shape[1], chunk_size):
                end_k = min(k + chunk_size, A.shape[1])
                __A = _A[:, k:end_k].to(device)
                __B = _B[k:end_k].to(device)
                _C = __A @ __B
                _C_ij = _C if _C_ij is None else _C_ij + _C
            _C_ij = transform(_C_ij)

            _C_ij = _C_ij.to(large_device)
            C[i:end_i, j:end_j] = _C_ij
    return C


def get_d(A, B, n_inv=-1, device="cuda:0", chunk_size=8096):
    # A : (m, m)
    # B : (m, n)
    d_A, d_B = A.sum(1), B.sum(1)
    d1 = d_A + d_B

    A_inv = truncate_inv(A.to(device).float(), n_inv=n_inv)
    A_inv = A_inv.to(dtype=B.dtype)
    B_sum = d_B.to(device)
    _right_mat = A_inv @ B_sum.reshape(-1, 1)
    # _right_mat = _right_mat.abs() # FIXME: this is a hack
    d2 = chunked_matmul(B.T, _right_mat, chunk_size=chunk_size, device=device)
    d2 = d2.reshape(-1)
    d2 = d2 + B.sum(0)

    d = torch.cat([d1, d2])
    print("degree", d.min(), d.max())
    if d.min() <= 0 and d.min() > -1e-1:
        print("negative degree found, forcing to be positive")
        d = d.abs()
    d += 1e-5
    assert torch.all(d > 0), f"negative degree found, {d.min()}"
    print("degree", d.min(), d.max())
    return d


def _nystrom_ncut(A, B, num_eig=20, n_inv=100, device="cuda:0", chunk_size=8096):
    # ei, ev = torch.linalg.eigh(A)
    A = A.to(device)
    ev, ei, _ = torch.svd_lowrank(A, num_eig)
    ei, ev = ei.to(B.dtype), ev.to(B.dtype)
    c = ei.sum() / torch.trace(A)
    print("get_topk_V, c", c)
    print("get_topk_V, ei", ei.min(), ei.max())
    A_sqrt_inv = ev @ torch.diag(1 / torch.sqrt(ei.abs() + 1e-8)) @ ev.T

    BB_T = chunked_matmul(B, B.T, chunk_size=chunk_size, device=device)
    BB_T = BB_T.to(device)
    print("BB_T", BB_T.min(), BB_T.max())
    S = A + A_sqrt_inv @ BB_T @ A_sqrt_inv
    
    indirect_A = S - A

    ev_s, ei_s, _ = torch.svd_lowrank(S, num_eig)
    arg_sort = torch.argsort(ei_s, descending=True)
    ei_s, ev_s = ei_s[arg_sort], ev_s[:, arg_sort]
    ei_s, ev_s = ei_s.to(B.dtype), ev_s.to(B.dtype)

    V_left = torch.cat([A.cpu(), B.T.cpu()], dim=0)
    # V = V_left @ A_sqrt_inv @ ev_s @ torch.diag(1 / torch.sqrt(ei_s.abs() + 1e-8))
    # the following code is memory efficient to the line above
    __r = chunked_matmul(A_sqrt_inv, ev_s, chunk_size=chunk_size, device=device)
    _ei_s = torch.diag(1 / torch.sqrt(ei_s.abs() + 1e-8))
    _right_mat = chunked_matmul(__r, _ei_s, chunk_size=chunk_size, device=device)
    V = chunked_matmul(V_left, _right_mat, chunk_size=chunk_size, device=device)
    return V, ei_s, indirect_A


def truncate_inv(A, n_inv=-1):

    if n_inv > 0:
        ev, ei, _ = torch.svd_lowrank(A, n_inv)
        c = ei.sum() / torch.trace(A)
        # ei *= c
        print("eig_inv, c", c)
    else:
        ei, ev = torch.linalg.eigh(A)
        print("eig_inv, ei", ei.min(), ei.max())
    A_inv = ev @ torch.diag(1 / (ei.abs() + 1e-8)) @ ev.T
    return A_inv


def symmetric_normalize(A, B, D):
    n_sample = A.shape[0]
    A /= torch.sqrt(D)[:n_sample, None]
    A /= torch.sqrt(D)[None, :n_sample]
    B /= torch.sqrt(D)[:n_sample, None]
    B /= torch.sqrt(D)[None, n_sample:]
    return A, B


from ncut_pytorch.ncut_pytorch import correct_rotation, get_affinity
@torch.no_grad()
def nystrom_ncut(
    features,
    sample_indices,
    distance="rbf",
    num_eig=50,
    n_inv=50,
    device="cuda:0",
    chunk_size=8096,
):
    A = get_affinity(features[sample_indices], distance=distance)
    not_sample_indices = np.setdiff1d(np.arange(features.shape[0]), sample_indices)
    B = get_affinity(features[sample_indices], features[not_sample_indices], distance=distance, fill_diagonal=False)

    indices = np.concatenate([sample_indices, not_sample_indices])
    reverse_indices = np.argsort(indices)


    D = get_d(A, B, n_inv=n_inv, device=device, chunk_size=chunk_size)
    A, B = symmetric_normalize(A, B, D)

    V, L, _ = _nystrom_ncut(
        A, B, num_eig=num_eig, n_inv=n_inv, device=device, chunk_size=chunk_size
    )
    V = V[reverse_indices]
    V = correct_rotation(V)

    return V, L
    # eigenvectors, eigenvalues