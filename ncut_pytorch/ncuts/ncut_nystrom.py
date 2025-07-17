import torch

from ncut_pytorch.utils.gamma import find_gamma_by_degree_after_fps
from ncut_pytorch.utils.math_utils import get_affinity, gram_schmidt, normalize_affinity, svd_lowrank, correct_rotation, \
    keep_topk_per_row
from ncut_pytorch.utils.sample_utils import auto_divice, farthest_point_sampling

# internal configuration for nystrom approximation, can be overridden by kwargs
# values are optimized based on empirical experiments, no need to change the values
_NYSTROM_CONFIG = {
    'n_sample': 10240,  # number of samples for nystrom approximation, 10240 is large enough for most cases
    'n_sample2': 1024,  # number of samples for eigenvector propagation, 1024 is large enough for most cases
    'n_neighbors': 10,  # number of neighbors for eigenvector propagation, 10 is large enough for most cases
    'matmul_chunk_size': 16384,  # chunk size for matrix multiplication, larger chunk size is faster but requires more memory
    'sample_method': "farthest",  # sample method for nystrom approximation, 'farthest' is FPS(Farthest Point Sampling)
    'move_output_to_cpu': True,  # if True, will move output to cpu, which saves memory but loses gradients
}

def ncut_fn(
        X: torch.Tensor,
        n_eig: int = 100,
        track_grad: bool = False,
        d_gamma: float = 0.1,
        device: str = 'auto',
        gamma: float = None,
        make_orthogonal: bool = False,
        no_propagation: bool = False,
        **kwargs,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Normalized Cut, balanced sampling and nystrom approximation.
    Function interface that returns both eigenvectors and eigenvalues.

    Args:
        X (torch.Tensor): input features, shape (N, D)
        n_eig (int): number of eigenvectors
        track_grad (bool): keep track of pytorch gradients
        d_gamma (float): affinity gamma parameter, lower d_gamma results in sharper eigenvectors
        device (str): device, default 'auto' (auto detect GPU)
        gamma (float): affinity parameter, override d_gamma if provided
        make_orthogonal (bool): make eigenvectors orthogonal
        no_propagation (bool): if True, return intermediate results without propagation
    Returns:
        (torch.Tensor): eigenvectors, shape (N, n_eig)
        (torch.Tensor): eigenvalues, sorted in descending order, shape (n_eig,)
    Examples:
        >>> from ncut_pytorch import ncut_fn
        >>> import torch
        >>> features = torch.rand(10000, 100)
        >>> eigvec, eigval = ncut_fn(features, n_eig=20)
        >>> print(eigvec.shape, eigval.shape)  # (10000, 20) (20,)
    """
    _config = _NYSTROM_CONFIG.copy()
    _config.update(kwargs)

    # use GPU if available
    device = auto_divice(X.device, device)

    # skip pytorch gradient computation if track_grad is False
    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(track_grad)

    # subsample for nystrom approximation
    nystrom_indices = farthest_point_sampling(X, n_sample=_config['n_sample'], device=device)
    nystrom_X = X[nystrom_indices].to(device)

    # find optimal gamma for affinity matrix
    if gamma is None:
        gamma = find_gamma_by_degree_after_fps(nystrom_X, d_gamma)

    # compute Ncut on the nystrom sampled subgraph
    A = get_affinity(nystrom_X, gamma=gamma)
    nystrom_eigvec, eigval = _plain_ncut(A, n_eig)

    if no_propagation:
        torch.set_grad_enabled(prev_grad_state)
        return nystrom_eigvec, eigval, nystrom_indices, gamma

    # propagate eigenvectors from subgraph to full graph
    eigvec = _nystrom_propagate(
        nystrom_eigvec,
        X,
        nystrom_X,
        n_neighbors=_config['n_neighbors'],
        n_sample=_config['n_sample2'],
        gamma=gamma,
        chunk_size=_config['matmul_chunk_size'],
        device=device,
        move_output_to_cpu=_config['move_output_to_cpu'],
        track_grad=track_grad,
    )

    # post-hoc orthogonalization
    if make_orthogonal:
        eigvec = gram_schmidt(eigvec)

    torch.set_grad_enabled(prev_grad_state)

    return eigvec, eigval


def _plain_ncut(
        A: torch.Tensor,
        n_eig: int = 100,
):
    """Normalized Cut.

    Args:
        A (torch.Tensor): affinity matrix, shape (N, N)
        n_eig (int): number of eigenvectors to return

    Returns:
        (torch.Tensor): eigenvectors corresponding to the eigenvalues, shape (N, n_eig)
        (torch.Tensor): eigenvalues of the eigenvectors, sorted in descending order
    """

    # normalization; A = D^(-1/2) A D^(-1/2)
    A = normalize_affinity(A)

    eigvec, eigval, _ = svd_lowrank(A, n_eig)

    # correct the random rotation (flipping sign) of eigenvectors
    eigvec = correct_rotation(eigvec)

    return eigvec, eigval


def _nystrom_propagate(
        nystrom_out: torch.Tensor,
        X: torch.Tensor,
        nystrom_X: torch.Tensor,
        gamma: float = 1.0,
        track_grad: bool = False,
        device: str = 'auto',
        **kwargs,
):
    """propagate output from nystrom sampled nodes to all nodes,
    use a weighted sum of the nearest neighbors to propagate the output.

    Args:
        nystrom_out (torch.Tensor): output from nystrom sampled nodes, shape (m, D)
        X (torch.Tensor): input features for all nodes, shape (N, D)
        nystrom_X (torch.Tensor): input features from nystrom sampled nodes, shape (m, D)
        gamma (float): affinity parameter, default 1.0
        track_grad (bool): keep track of pytorch gradients, default False
        device (str): device to use for computation, if 'auto', will detect GPU automatically
        _config (dict): configuration for nystrom approximation, default _NYSTROM_CONFIG

    Returns:
        torch.Tensor: output propagated by nearest neighbors, shape (N, D)
    """

    _config = _NYSTROM_CONFIG.copy()
    _config.update(kwargs)

    # skip pytorch gradient computation if track_grad is False
    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(track_grad)

    device = auto_divice(nystrom_out.device, device)
    indices = farthest_point_sampling(nystrom_out, _config['n_sample2'], device=device)
    nystrom_out = nystrom_out[indices].to(device)
    nystrom_X = nystrom_X[indices].to(device)

    all_outs = []
    n_chunk = _config['matmul_chunk_size']
    n_neighbors = _config['n_neighbors']
    for i in range(0, X.shape[0], n_chunk):
        end = min(i + n_chunk, X.shape[0])

        _Ai = get_affinity(X[i:end].to(device), nystrom_X, gamma=gamma)
        _Ai, _indices = keep_topk_per_row(_Ai, n_neighbors)  # (n, n_neighbors)

        weights = _Ai[..., None]  # (n, n_neighbors, 1)
        neighbors = nystrom_out[_indices.flatten()].reshape(-1, n_neighbors, nystrom_out.shape[-1])  # (n, n_neighbors, d)
        out = weights * neighbors  # (n, n_neighbors, d)
        out = out.sum(dim=1)  # (n, d)

        if _config['move_output_to_cpu'] and not track_grad:
            out = out.cpu()
        all_outs.append(out)

    all_outs = torch.cat(all_outs, dim=0)

    torch.set_grad_enabled(prev_grad_state)

    return all_outs



