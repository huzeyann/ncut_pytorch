# %%
from sympy import Q
import torch

from ncut_pytorch.utils.gamma import find_gamma_by_degree_after_fps, find_gamma_by_degree
from ncut_pytorch.utils.math_utils import get_affinity, normalize_affinity, svd_lowrank, correct_rotation
from ncut_pytorch.utils.sample_utils import farthest_point_sampling, auto_divice
from .ncut_kway import kway_ncut
from .ncut_nystrom import _nystrom_propagate
from .ncut_nystrom import _plain_ncut
from .ncut_nystrom import NystromConfig


#TODO: automatically optimize click_weight based on the iou of fg and bg
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
        return_indices_and_gamma: bool = False,
        **kwargs,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:

    config = NystromConfig()
    config.update(kwargs)

    # use GPU if available
    device = auto_divice(X.device, device)

    # skip pytorch gradient computation if track_grad is False
    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(track_grad)
    
    if bg_indices is None:
        bg_indices = torch.tensor([], dtype=torch.long)

    # subsample for nystrom approximation
    nystrom_indices = farthest_point_sampling(X, n_sample=config.n_sample, device=device)
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
    eigvec, nystrom_indices2 = _nystrom_propagate(
        nystrom_eigvec,
        X,
        nystrom_X,
        n_neighbors=config.n_neighbors,
        n_sample=config.n_sample2,
        gamma=gamma,
        matmul_chunk_size=config.matmul_chunk_size,
        device=device,
        move_output_to_cpu=config.move_output_to_cpu,
        track_grad=track_grad,
        return_indices=True,
    )
    
    torch.set_grad_enabled(prev_grad_state)    

    if return_indices_and_gamma:
        indices = nystrom_indices[nystrom_indices2]
        return eigvec, eigval, indices, gamma

    return eigvec, eigval


def get_mask_and_heatmap(eigvecs, fg_indices, n_cluster=2, device='auto'):
    device = auto_divice(eigvecs.device, device)
    eigvecs = eigvecs[:, :n_cluster]

    eigvecs = kway_ncut(eigvecs, device=device)
    # find which cluster is the foreground
    fg_eigvecs = eigvecs[fg_indices]
    fg_idx = fg_eigvecs.mean(0).argmax().item()
    bg_idx = 1 if fg_idx == 0 else 0
    
    # discretize the eigvecs
    mask = eigvecs.argmax(dim=-1) == fg_idx

    heatmap = eigvecs[:, fg_idx] - eigvecs[:, bg_idx]
    
    return mask, heatmap

