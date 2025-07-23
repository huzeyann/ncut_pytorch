# %%
from sympy import Q
import torch

from ncut_pytorch.utils.gamma import find_gamma_by_degree_after_fps, find_gamma_by_degree
from ncut_pytorch.utils.math_utils import get_affinity, normalize_affinity, svd_lowrank, correct_rotation
from ncut_pytorch.utils.sample_utils import farthest_point_sampling, auto_divice
from .ncut_kway import kway_ncut
from .ncut_nystrom import _nystrom_propagate
from .ncut_nystrom import _plain_ncut
from .ncut_nystrom import _NYSTROM_CONFIG


def ncut_click_prompt(
        X: torch.Tensor,
        fg_indices: torch.Tensor,
        bg_indices: torch.Tensor = None,
        click_weight: float = 0.5,
        bg_weight: float = 0.1,
        n_eig: int = 2,
        track_grad: bool = False,
        d_gamma: float = 0.1,
        device: str = 'auto',
        gamma: float = None,
        no_propagation: bool = False,
        **kwargs,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:

    _config = _NYSTROM_CONFIG.copy()
    _config.update(kwargs)

    # use GPU if available
    device = auto_divice(X.device, device)

    # skip pytorch gradient computation if track_grad is False
    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(track_grad)
    
    if bg_indices is None:
        bg_indices = torch.tensor([], dtype=torch.long)

    # subsample for nystrom approximation
    nystrom_indices = farthest_point_sampling(X, n_sample=_config['n_sample'], device=device)
    nystrom_indices = torch.tensor(nystrom_indices, dtype=torch.long)
    # remove fg and bg from fps_idx
    nystrom_indices = nystrom_indices[~torch.isin(nystrom_indices, torch.cat([fg_indices, bg_indices]))]
    # add fg and bg to fps_idx
    nystrom_indices = torch.cat([fg_indices, bg_indices, nystrom_indices])
    fg_indices = torch.arange(len(fg_indices))
    bg_indices = torch.arange(len(bg_indices)) + len(fg_indices)
    n_fgbg = len(fg_indices) + len(bg_indices)
    
    nystrom_X = X[nystrom_indices].to(device)
    
    # find optimal gamma for affinity matrix
    if gamma is None:
        gamma = find_gamma_by_degree_after_fps(nystrom_X, d_gamma)

    # compute Ncut on the nystrom sampled subgraph
    A = get_affinity(nystrom_X, gamma=gamma)
    A = normalize_affinity(A)

    # modify the affinity from the clicks
    X_click = 1 * A[fg_indices].mean(0)
    if len(bg_indices) > 0:
        X_click = X_click - bg_weight * A[bg_indices].mean(0)
    
    X_click = X_click * A.shape[0]
    
    # gamma2 = find_gamma_by_degree(X_click.unsqueeze(1), d_gamma)
    # A_click = get_affinity(X_click.unsqueeze(1), gamma=gamma2)
    A_click = get_affinity(X_click.unsqueeze(1), gamma=0.5)
    # A_click = - torch.cdist(X_click.unsqueeze(1), X_click.unsqueeze(1))
    A_click = normalize_affinity(A_click)
    
    _A = click_weight * A_click + (1 - click_weight) * A
    # _A = _A[n_fgbg:, n_fgbg:]
    # nystrom_indices = nystrom_indices[n_fgbg:]
    # nystrom_X = nystrom_X[n_fgbg:]
        
    nystrom_eigvec, eigval = _plain_ncut(_A, n_eig)
    
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
    
    torch.set_grad_enabled(prev_grad_state)    

    return eigvec, eigval


def get_mask_and_heatmap(eigvecs, fg_indices, num_cluster=2, device='auto'):
    device = auto_divice(eigvecs.device, device)
    eigvecs = eigvecs[:, :num_cluster]

    eigvecs = kway_ncut(eigvecs[:, :num_cluster], device=device)
    # find which cluster is the foreground
    fg_eigvecs = eigvecs[fg_indices]
    fg_idx = fg_eigvecs.mean(0).argmax().item()
    bg_idx = 1 if fg_idx == 0 else 0
    
    # discretize the eigvecs
    mask = eigvecs.argmax(dim=-1) == fg_idx

    heatmap = eigvecs[:, fg_idx] - eigvecs[:, bg_idx]
    
    return mask, heatmap



from ncut_pytorch.utils.math_utils import keep_topk_per_row

def ncut_click_prompt_cached(
        nystrom_indices: torch.Tensor,
        gamma: float,
        X: torch.Tensor,
        fg_indices: torch.Tensor,
        bg_indices: torch.Tensor = None,
        click_weight: float = 0.5,
        bg_weight: float = 0.1,
        n_eig: int = 2,
        track_grad: bool = False,
        device: str = 'auto',
        **kwargs,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:

    _config = _NYSTROM_CONFIG.copy()
    _config.update(kwargs)

    # use GPU if available
    device = auto_divice(X.device, device)

    # skip pytorch gradient computation if track_grad is False
    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(track_grad)
    
    if bg_indices is None:
        bg_indices = torch.tensor([], dtype=torch.long)

    # subsample for nystrom approximation
    nystrom_indices = torch.tensor(nystrom_indices, dtype=torch.long)
    # add fg and bg to fps_idx
    nystrom_indices = torch.cat([fg_indices, bg_indices, nystrom_indices])
    fg_indices = torch.arange(len(fg_indices))
    bg_indices = torch.arange(len(bg_indices)) + len(fg_indices)
    n_fgbg = len(fg_indices) + len(bg_indices)
    
    nystrom_X = X[nystrom_indices].to(device)
    
    # compute Ncut on the nystrom sampled subgraph
    A = get_affinity(nystrom_X, gamma=gamma)
    A = normalize_affinity(A)

    # modify the affinity from the clicks
    X_click = 1 * A[fg_indices].mean(0)
    if len(bg_indices) > 0:
        X_click = X_click - bg_weight * A[bg_indices].mean(0)
    
    X_click = X_click * A.shape[0]
    
    A_click = get_affinity(X_click.unsqueeze(1), gamma=0.5)
    A_click = normalize_affinity(A_click)
    
    _A = click_weight * A_click + (1 - click_weight) * A
    _A = _A[n_fgbg:, n_fgbg:]
    nystrom_indices = nystrom_indices[n_fgbg:]
    nystrom_X = nystrom_X[n_fgbg:]
        
    nystrom_eigvec, eigval = _plain_ncut(_A, n_eig)
    
    torch.set_grad_enabled(prev_grad_state)
    return nystrom_eigvec, eigval


def _build_nystrom_graph(
        X: torch.Tensor,
        nystrom_X: torch.Tensor,
        gamma: float = 1.0,
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

    device = auto_divice(X.device, device)
    nystrom_X = nystrom_X.to(device)

    all_outs = []
    n_chunk = _config['matmul_chunk_size']
    n_neighbors = _config['n_neighbors']
    cached_weights = torch.zeros((X.shape[0], nystrom_X.shape[0]), 
                                 device=device, dtype=X.dtype)
    for i in range(0, X.shape[0], n_chunk):
        end = min(i + n_chunk, X.shape[0])

        _Ai = get_affinity(X[i:end].to(device), nystrom_X, gamma=gamma)
        _Ai, _indices = keep_topk_per_row(_Ai, n_neighbors)  # (n, n_neighbors)
        row_indices = torch.arange(i, end).unsqueeze(1).expand(-1, n_neighbors)  # shape (N, 10)
        cached_weights[row_indices, _indices] = _Ai
        print((cached_weights[i] > 0).sum())

    return cached_weights
