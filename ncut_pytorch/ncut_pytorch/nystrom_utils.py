import logging
import math
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def run_subgraph_sampling(
    features: torch.Tensor,
    num_sample: int = 300,
    max_draw: int = 1000000,
    sample_method: Literal["farthest", "random"] = "farthest",
):
    if num_sample >= features.shape[0]:
        # if too many samples, use all samples and bypass Nystrom-like approximation
        logging.info(
            "num_sample is larger than total, bypass Nystrom-like approximation"
        )
        sampled_indices = torch.arange(features.shape[0])
    else:
        # sample subgraph
        if sample_method == "farthest":  # default
            if num_sample > max_draw:
                logging.warning(
                    f"num_sample is larger than max_draw, apply farthest point sampling on random sampled {max_draw} samples"
                )
                draw_indices = torch.randperm(features.shape[0])[:max_draw]
                sampled_indices = farthest_point_sampling(
                    features[draw_indices].detach(),
                    num_sample=num_sample,
                )
                sampled_indices = draw_indices[sampled_indices]
            else:
                sampled_indices = farthest_point_sampling(
                    features.detach(),
                    num_sample=num_sample,
                )
        elif sample_method == "random":  # not recommended
            sampled_indices = torch.randperm(features.shape[0])[:num_sample]
        else:
            raise ValueError("sample_method should be 'farthest' or 'random'")
    return sampled_indices


def farthest_point_sampling(
    features: torch.Tensor,
    num_sample: int = 300,
    h: int = 9,
):
    try:
        import fpsample
    except ImportError:
        raise ImportError(
            "fpsample import failed, please install `pip install fpsample`"
        )

    # PCA to reduce the dimension
    if features.shape[1] > 8:
        u, s, v = torch.pca_lowrank(features, q=8)
        _n = features.shape[0]
        s /= math.sqrt(_n)
        features = u @ torch.diag(s)

    h = min(h, int(np.log2(features.shape[0])))

    kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(
        features.cpu().numpy(), num_sample, h
    ).astype(np.int64)
    return kdline_fps_samples_idx


def propagate_knn(
    subgraph_output: torch.Tensor,
    inp_features: torch.Tensor,
    subgraph_features: torch.Tensor,
    knn: int = 10,
    distance: Literal["cosine", "euclidean", "rbf"] = "cosine",
    chunk_size: int = 8096,
    device: str = None,
    use_tqdm: bool = False,
    move_output_to_cpu: bool = False,
):
    """A generic function to propagate new nodes using KNN.

    Args:
        subgraph_output (torch.Tensor): output from subgraph, shape (num_sample, D)
        inp_features (torch.Tensor): features from existing nodes, shape (new_num_samples, n_features)
        subgraph_features (torch.Tensor): features from subgraph, shape (num_sample, n_features)
        knn (int): number of KNN to propagate eigenvectors
        distance (str): distance metric, 'cosine' (default) or 'euclidean', 'rbf'
        chunk_size (int): chunk size for matrix multiplication
        device (str): device to use for computation, if None, will not change device
        use_tqdm (bool): show progress bar when propagating eigenvectors from subgraph to full graph

    Returns:
        torch.Tensor: propagated eigenvectors, shape (new_num_samples, D)

    Examples:
        >>> old_eigenvectors = torch.randn(3000, 20)
        >>> old_features = torch.randn(3000, 100)
        >>> new_features = torch.randn(200, 100)
        >>> new_eigenvectors = propagate_knn(old_eigenvectors, new_features, old_features, knn=3)
        >>> # new_eigenvectors.shape = (200, 20)

    """
    device = subgraph_output.device if device is None else device
    if distance == 'cosine':
        if not check_if_normalized(inp_features):
            inp_features = F.normalize(inp_features, dim=-1)
        if not check_if_normalized(subgraph_features):
            subgraph_features = F.normalize(subgraph_features, dim=-1)

    if knn == 1:
        return propagate_nearest(
            subgraph_output,
            inp_features,
            subgraph_features,
            chunk_size=chunk_size,
            device=device,
            move_output_to_cpu=move_output_to_cpu,
        )

    # used in nystrom_ncut
    # propagate eigen_vector from subgraph to full graph
    subgraph_output = subgraph_output.to(device)
    V_list = []
    if use_tqdm:
        try:
            from tqdm import tqdm
        except ImportError:
            use_tqdm = False
    if use_tqdm:
        iterator = tqdm(range(0, inp_features.shape[0], chunk_size), "propagate by KNN")
    else:
        iterator = range(0, inp_features.shape[0], chunk_size)

    subgraph_features = subgraph_features.to(device)
    for i in iterator:
        end = min(i + chunk_size, inp_features.shape[0])
        _v = inp_features[i:end].to(device)
        #TODO: fix this, add expential for rbf
        if distance == 'cosine':
            _A = _v @ subgraph_features.T
        elif distance == 'euclidean':
            _A = - torch.cdist(_v, subgraph_features, p=2)
        elif distance == 'rbf':
            _A = - torch.cdist(_v, subgraph_features, p=2) ** 2
        else:
            raise ValueError("distance should be 'cosine' or 'euclidean', 'rbf'")

        # keep topk KNN for each row
        topk_sim, topk_idx = _A.topk(knn, dim=-1, largest=True)
        row_id = torch.arange(topk_idx.shape[0], device=_A.device)[:, None].expand(
            -1, topk_idx.shape[1]
        )
        _A = torch.sparse_coo_tensor(
            torch.stack([row_id, topk_idx], dim=-1).reshape(-1, 2).T,
            topk_sim.reshape(-1),
            size=(_A.shape[0], _A.shape[1]),
            device=_A.device,
        )
        _A = _A.to_dense().to(dtype=subgraph_output.dtype)
        # _A is KNN graph

        _D = _A.sum(-1)
        _A /= _D[:, None]

        _V = _A @ subgraph_output

        if move_output_to_cpu:
            _V = _V.cpu()
        V_list.append(_V)

    subgraph_output = torch.cat(V_list, dim=0)

    return subgraph_output


def propagate_nearest(
    subgraph_output,
    inp_features,
    subgraph_features,
    distance="cosine",
    chunk_size=8096,
    device=None,
    move_output_to_cpu=False,
):
    device = subgraph_output.device if device is None else device
    if distance == 'cosine':
        if not check_if_normalized(inp_features):
            inp_features = F.normalize(inp_features, dim=-1)
        if not check_if_normalized(subgraph_features):
            subgraph_features = F.normalize(subgraph_features, dim=-1)

    # used in nystrom_tsne, equivalent to propagate_by_knn with knn=1
    # propagate tSNE from subgraph to full graph
    V_list = []
    subgraph_features = subgraph_features.to(device)
    for i in range(0, inp_features.shape[0], chunk_size):
        end = min(i + chunk_size, inp_features.shape[0])
        _v = inp_features[i:end].to(device)
        if distance == 'cosine':
            _A = _v @ subgraph_features.T
        elif distance == 'euclidean':
            _A = - torch.cdist(_v, subgraph_features, p=2)
        elif distance == 'rbf':
            _A = - torch.cdist(_v, subgraph_features, p=2) ** 2
        else:
            raise ValueError("distance should be 'cosine' or 'euclidean', 'rbf'")
        # keep top1 for each row
        top_idx = _A.argmax(dim=-1).cpu()
        _V = subgraph_output[top_idx]
        if move_output_to_cpu:
            _V = _V.cpu()
        V_list.append(_V)

    subgraph_output = torch.cat(V_list, dim=0)

    return subgraph_output


# wrapper functions for adding new nodes to existing graph
def propagate_eigenvectors(
    eigenvectors: torch.Tensor,
    features: torch.Tensor,
    new_features: torch.Tensor,
    knn: int,
    num_sample: int,
    sample_method: Literal["farthest", "random"],
    chunk_size: int,
    device: str,
    use_tqdm: bool,
):
    """Propagate eigenvectors to new nodes using KNN. Note: this is equivalent to the class API `NCUT.tranform(new_features)`, expect for the sampling is re-done in this function.
    Args:
        eigenvectors (torch.Tensor): eigenvectors from existing nodes, shape (num_sample, num_eig)
        features (torch.Tensor): features from existing nodes, shape (n_samples, n_features)
        new_features (torch.Tensor): features from new nodes, shape (n_new_samples, n_features)
        knn (int): number of KNN to propagate eigenvectors, default 3
        num_sample (int): number of samples for subgraph sampling, default 50000
        sample_method (str): sample method, 'farthest' (default) or 'random'
        chunk_size (int): chunk size for matrix multiplication, default 8096
        device (str): device to use for computation, if None, will not change device
        use_tqdm (bool): show progress bar when propagating eigenvectors from subgraph to full graph

    Returns:
        torch.Tensor: propagated eigenvectors, shape (n_new_samples, num_eig)

    Examples:
        >>> old_eigenvectors = torch.randn(3000, 20)
        >>> old_features = torch.randn(3000, 100)
        >>> new_features = torch.randn(200, 100)
        >>> new_eigenvectors = propagate_eigenvectors(old_eigenvectors, new_features, old_features, knn=3)
        >>> # new_eigenvectors.shape = (200, 20)
    """

    device = eigenvectors.device if device is None else device

    # sample subgraph
    subgraph_indices = run_subgraph_sampling(
        features,
        num_sample=num_sample,
        sample_method=sample_method,
    )

    subgraph_eigenvectors = eigenvectors[subgraph_indices].to(device)
    subgraph_features = features[subgraph_indices].to(device)
    new_features = new_features.to(device)

    # propagate eigenvectors from subgraph to new nodes
    new_eigenvectors = propagate_knn(
        subgraph_eigenvectors,
        new_features,
        subgraph_features,
        knn=knn,
        chunk_size=chunk_size,
        device=device,
        use_tqdm=use_tqdm,
    )

    return new_eigenvectors


def check_if_normalized(x, n=1000):
    """check if the input tensor is normalized (unit norm)"""
    n = min(n, x.shape[0])
    random_indices = torch.randperm(x.shape[0])[:n]
    _x = x[random_indices]
    flag = torch.allclose(torch.norm(_x, dim=-1), torch.ones(n, device=x.device))
    return flag


def quantile_min_max(x, q1=0.01, q2=0.99, n_sample=10000):
    if x.shape[0] > n_sample:
        np.random.seed(0)
        random_idx = np.random.choice(x.shape[0], n_sample, replace=False)
        vmin, vmax = x[random_idx].quantile(q1), x[random_idx].quantile(q2)
    else:
        vmin, vmax = x.quantile(q1), x.quantile(q2)
    return vmin, vmax


def quantile_normalize(x, q=0.95):
    """normalize each dimension of x to [0, 1], take 95-th percentage, this robust to outliers
        </br> 1. sort x
        </br> 2. take q-th quantile
        </br>     min_value -> (1-q)-th quantile
        </br>     max_value -> q-th quantile
        </br> 3. normalize
        </br> x = (x - min_value) / (max_value - min_value)

    Args:
        x (torch.Tensor): input tensor, shape (n_samples, n_features)
            normalize each feature to 0-1 range
        q (float): quantile, default 0.95

    Returns:
        torch.Tensor: quantile normalized tensor
    """
    # normalize x to 0-1 range, max value is q-th quantile
    # quantile makes the normalization robust to outliers
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    vmax, vmin = quantile_min_max(x, q, 1 - q)
    x = (x - vmin) / (vmax - vmin)
    x = x.clamp(0, 1)
    return x
