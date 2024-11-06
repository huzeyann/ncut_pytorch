# %%
import logging
import math
from typing import Literal
import torch
import torch.nn.functional as F
import numpy as np


class NCUT:
    """Nystrom Normalized Cut for large scale graph."""

    def __init__(
        self,
        num_eig : int = 100,
        knn : int = 10,
        affinity_focal_gamma : float = 1.0,
        num_sample : int = 10000,
        sample_method : Literal["farthest", "random"] = "farthest",
        distance : Literal["cosine", "euclidean", "rbf"] = "cosine",
        indirect_connection : bool = True,
        indirect_pca_dim : int = 100,
        device : str = None,
        move_output_to_cpu : bool = False,
        eig_solver : Literal["svd_lowrank", "lobpcg", "svd", "eigh"] = "svd_lowrank",
        normalize_features : bool = None,
        matmul_chunk_size : int = 8096,
        make_orthogonal : bool = False,
        verbose : bool = False,
    ):
        """

        Args:
            num_eig (int): number of top eigenvectors to return
            knn (int): number of KNN for propagating eigenvectors from subgraph to full graph,
                smaller knn result in more sharp eigenvectors.
            affinity_focal_gamma (float): affinity matrix temperature, lower t reduce the not-so-connected edge weights,
                smaller t result in more sharp eigenvectors.
            num_sample (int): number of samples for Nystrom-like approximation,
                reduce only if memory is not enough, increase for better approximation
            sample_method (str): subgraph sampling, ['farthest', 'random'].
                farthest point sampling is recommended for better Nystrom-approximation accuracy
            distance (str): distance metric for affinity matrix, ['cosine', 'euclidean', 'rbf'].
            indirect_connection (bool): include indirect connection in the Nystrom-like approximation
            indirect_pca_dim (int): when compute indirect connection, PCA to reduce the node dimension,
            device (str): device to use for eigen computation,
                move to GPU to speeds up a bit (~5x faster)
            move_output_to_cpu (bool): move output to CPU, set to True if you have memory issue
            eig_solver (str): eigen decompose solver, ['svd_lowrank', 'lobpcg', 'svd', 'eigh'].
            normalize_features (bool): normalize input features before computing affinity matrix,
                default 'None' is True for cosine distance, False for euclidean distance and rbf
            matmul_chunk_size (int): chunk size for large-scale matrix multiplication
            make_orthogonal (bool): make eigenvectors orthogonal post-hoc
            verbose (bool): progress bar

        Examples:
            >>> from ncut_pytorch import NCUT
            >>> import torch
            >>> features = torch.rand(10000, 100)
            >>> ncut = NCUT(num_eig=20)
            >>> ncut.fit(features)
            >>> eigenvectors, eigenvalues = ncut.transform(features)
            >>> print(eigenvectors.shape, eigenvalues.shape)
            >>> # (10000, 20) (20,)

            >>> from ncut_pytorch import eigenvector_to_rgb
            >>> # use t-SNE or UMAP to convert eigenvectors to RGB
            >>> X_3d, rgb = eigenvector_to_rgb(eigenvectors, method='tsne_3d')
            >>> print(X_3d.shape, rgb.shape)
            >>> # (10000, 3) (10000, 3)

            >>> # transform new features
            >>> new_features = torch.rand(500, 100)
            >>> new_eigenvectors, _ = ncut.transform(new_features)
            >>> print(new_eigenvectors.shape)
            >>> # (500, 20)
        """
        self.num_eig = num_eig
        self.num_sample = num_sample
        self.knn = knn
        self.sample_method = sample_method
        self.distance = distance
        self.affinity_focal_gamma = affinity_focal_gamma
        self.indirect_connection = indirect_connection
        self.indirect_pca_dim = indirect_pca_dim
        self.device = device
        self.move_output_to_cpu = move_output_to_cpu
        self.eig_solver = eig_solver
        self.normalize_features = normalize_features
        self.matmul_chunk_size = matmul_chunk_size
        self.make_orthogonal = make_orthogonal
        self.verbose = verbose

    def fit(self, 
            features : torch.Tensor, 
            precomputed_sampled_indices : torch.Tensor = None
            ):
        """Fit Nystrom Normalized Cut on the input features.

        Args:
            features (torch.Tensor): input features, shape (n_samples, n_features)
            precomputed_sampled_indices (torch.Tensor): precomputed sampled indices, shape (num_sample,)
                override the sample_method, if not None
        Returns:
            (NCUT): self
        """
        _n = features.shape[0]
        if self.num_sample >= _n:
            logging.warning(
                f"NCUT nystrom num_sample is larger than number of input samples, nyström approximation is not needed, settting num_sample={_n} and knn=1"
            )
            self.num_sample = _n
            self.knn = 1
        
        # save the eigenvectors solution on the sub-sampled graph, do not propagate to full graph yet
        self.subgraph_eigen_vector, self.eigen_value, self.subgraph_indices = (
            nystrom_ncut(
                features,
                num_eig=self.num_eig,
                num_sample=self.num_sample,
                sample_method=self.sample_method,
                precomputed_sampled_indices=precomputed_sampled_indices,
                distance=self.distance,
                affinity_focal_gamma=self.affinity_focal_gamma,
                indirect_connection=self.indirect_connection,
                indirect_pca_dim=self.indirect_pca_dim,
                device=self.device,
                eig_solver=self.eig_solver,
                normalize_features=self.normalize_features,
                matmul_chunk_size=self.matmul_chunk_size,
                verbose=self.verbose,
                no_propagation=True,
            )
        )
        self.subgraph_features = features[self.subgraph_indices]
        return self

    def transform(self, features : torch.Tensor, knn : int = None):
        """Transform new features using the fitted Nystrom Normalized Cut.

        Args:
            features (torch.Tensor): new features, shape (n_samples, n_features)
            knn (int): number of KNN for propagating eigenvectors from subgraph to full graph,
        Returns:
            (torch.Tensor): eigen_vectors, shape (n_samples, num_eig)
            (torch.Tensor): eigen_values, sorted in descending order, shape (num_eig,)
        """
        
        knn = self.knn if knn is None else knn
        
        # propagate eigenvectors from subgraph to full graph
        eigen_vector = propagate_knn(
            self.subgraph_eigen_vector,
            features,
            self.subgraph_features,
            knn,
            distance=self.distance,
            chunk_size=self.matmul_chunk_size,
            device=self.device,
            use_tqdm=self.verbose,
            move_output_to_cpu=self.move_output_to_cpu,
        )
        if self.make_orthogonal:
            eigen_vector = gram_schmidt(eigen_vector)
        return eigen_vector, self.eigen_value

    def fit_transform(self, 
                    features : torch.Tensor,
                    precomputed_sampled_indices : torch.Tensor = None
                    ):
        """

        Args:
            features (torch.Tensor): input features, shape (n_samples, n_features)
            precomputed_sampled_indices (torch.Tensor): precomputed sampled indices, shape (num_sample,)
                override the sample_method, if not None
                
        Returns:
            (torch.Tensor): eigen_vectors, shape (n_samples, num_eig)
            (torch.Tensor): eigen_values, sorted in descending order, shape (num_eig,)
        """
        return self.fit(features, precomputed_sampled_indices=precomputed_sampled_indices).transform(features)


def eigenvector_to_rgb(
    eigen_vector : torch.Tensor,
    method : Literal["tsne_2d", "tsne_3d", "umap_sphere", "umap_2d", "umap_3d"] = "tsne_3d",
    num_sample : int = 300,
    perplexity : int = 150,
    n_neighbors : int = 150,
    min_distance : float = 0.1,
    metric : Literal["cosine", "euclidean"] = "cosine",
    device : str = None,
    q : float = 0.95,
    knn : int = 10,
    seed : int = 0,
):
    """Use t-SNE or UMAP to convert eigenvectors (more than 3) to RGB color (3D RGB CUBE).

    Args:
        eigen_vector (torch.Tensor): eigenvectors, shape (n_samples, num_eig)
        method (str): method to convert eigenvectors to RGB,
            choices are: ['tsne_2d', 'tsne_3d', 'umap_sphere', 'umap_2d', 'umap_3d']
        num_sample (int): number of samples for Nystrom-like approximation, increase for better approximation
        perplexity (int): perplexity for t-SNE, increase for more global structure
        n_neighbors (int): number of neighbors for UMAP, increase for more global structure
        min_distance (float): minimum distance for UMAP
        metric (str): distance metric, default 'cosine'
        device (str): device to use for computation, if None, will not change device
        q (float): quantile for RGB normalization, default 0.95. lower q results in more sharp colors
        knn (int): number of KNN for propagating eigenvectors from subgraph to full graph,
            smaller knn result in more sharp colors, default 1. knn>1 will smooth-out the embedding
            in the t-SNE or UMAP space.
        seed (int): random seed for t-SNE or UMAP

    Examples:
        >>> from ncut_pytorch import eigenvector_to_rgb
        >>> X_3d, rgb = eigenvector_to_rgb(eigenvectors, method='tsne_3d')
        >>> print(X_3d.shape, rgb.shape)
        >>> # (10000, 3) (10000, 3)

    Returns:
        (torch.Tensor): t-SNE or UMAP embedding, shape (n_samples, 2) or (n_samples, 3)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    kwargs = {
        "num_sample": num_sample,
        "perplexity": perplexity,
        "n_neighbors": n_neighbors,
        "min_distance": min_distance,
        "metric": metric,
        "device": device,
        "q": q,
        "knn": knn,
        "seed": seed,
    }
    if method == "tsne_2d":
        embed, rgb = rgb_from_tsne_2d(eigen_vector, **kwargs)
    elif method == "tsne_3d":
        embed, rgb = rgb_from_tsne_3d(eigen_vector, **kwargs)
    elif method == "umap_sphere":
        embed, rgb = rgb_from_umap_sphere(eigen_vector, **kwargs)
    elif method == "umap_2d":
        embed, rgb = rgb_from_umap_2d(eigen_vector, **kwargs)
    elif method == "umap_3d":
        embed, rgb = rgb_from_umap_3d(eigen_vector, **kwargs)
    else:
        raise ValueError("method should be 'tsne_2d', 'tsne_3d' or 'umap_sphere'")

    return embed, rgb


def nystrom_ncut(
    features : torch.Tensor,
    num_eig : int = 100,
    num_sample : int = 10000,
    knn : int = 10,
    sample_method : Literal["farthest", "random"] = "farthest",
    precomputed_sampled_indices : torch.Tensor = None,
    distance : Literal["cosine", "euclidean", "rbf"] = "cosine",
    affinity_focal_gamma : float = 1.0,
    indirect_connection : bool = True,
    indirect_pca_dim : int = 100,
    device : str = None,
    eig_solver : Literal["svd_lowrank", "lobpcg", "svd", "eigh"] = "svd_lowrank",
    normalize_features : bool = None,
    matmul_chunk_size : int = 8096,
    make_orthogonal : bool = True,
    verbose : bool = False,
    no_propagation : bool = False,
):
    """PyTorch implementation of Faster Nystrom Normalized cut.

    Args:
        features (torch.Tensor): feature matrix, shape (n_samples, n_features)
        num_eig (int): default 100, number of top eigenvectors to return
        num_sample (int): default 10000, number of samples for Nystrom-like approximation
        knn (int): default 10, number of KNN for propagating eigenvectors from subgraph to full graph,
            smaller knn will result in more sharp eigenvectors,
        sample_method (str): sample method, 'farthest' (default) or 'random'
            'farthest' is recommended for better approximation
        precomputed_sampled_indices (torch.Tensor): precomputed sampled indices, shape (num_sample,)
            override the sample_method, if not None
        distance (str): distance metric, 'cosine' (default) or 'euclidean', 'rbf'
        affinity_focal_gamma (float): affinity matrix parameter, lower t reduce the weak edge weights,
            resulting in more sharp eigenvectors, default 1.0
        indirect_connection (bool): include indirect connection in the subgraph, default True
        indirect_pca_dim (int): default 100, PCA dimension to reduce the node dimension, only applied to
            the not sampled nodes, not applied to the sampled nodes
        device (str): device to use for computation, if None, will not change device
            a good practice is to pass features by CPU since it's usually large,
            and move subgraph affinity to GPU to speed up eigenvector computation
        eig_solver (str): eigen decompose solver, 'svd_lowrank' (default), 'lobpcg', 'svd', 'eigh'
            'svd_lowrank' is recommended for large scale graph, it's the fastest
            they correspond to torch.svd_lowrank, torch.lobpcg, torch.svd, torch.linalg.eigh
        normalize_features (bool): normalize input features before computing affinity matrix,
            default 'None' is True for cosine distance, False for euclidean distance and rbf
        matmul_chunk_size (int): chunk size for matrix multiplication
            large matrix multiplication is chunked to reduce memory usage,
            smaller chunk size will reduce memory usage but slower computation, default 8096
        make_orthogonal (bool): make eigenvectors orthogonal after propagation, default True
        verbose (bool): show progress bar when propagating eigenvectors from subgraph to full graph
        no_propagation (bool): if True, skip the eigenvector propagation step, only return the subgraph eigenvectors

    Returns:
        (torch.Tensor): eigenvectors, shape (n_samples, num_eig)
        (torch.Tensor): eigenvalues, sorted in descending order, shape (num_eig,)
        (torch.Tensor): sampled_indices used by Nystrom-like approximation subgraph, shape (num_sample,)
    """

    # check if features dimension greater than num_eig
    if eig_solver in ["svd_lowrank", "lobpcg"]:
        assert features.shape[0] > (
            num_eig * 2
        ), "number of nodes should be greater than 2*num_eig"
    if eig_solver in ["svd", "eigh"]:
        assert (
            features.shape[0] > num_eig
        ), "number of nodes should be greater than num_eig"

    assert distance in ["cosine", "euclidean", "rbf"], "distance should be 'cosine', 'euclidean', 'rbf'"
    if normalize_features is None:
        if distance in ["cosine"]:
            normalize_features = True
        if distance in ["euclidean", "rbf"]:
            normalize_features = False
    
    if normalize_features:
        # features need to be normalized for affinity matrix computation (cosine distance)
        features = torch.nn.functional.normalize(features, dim=-1)

    if precomputed_sampled_indices is not None:
        sampled_indices = precomputed_sampled_indices
    else:
        sampled_indices = run_subgraph_sampling(
            features,
            num_sample=num_sample,
            sample_method=sample_method,
        )

    sampled_features = features[sampled_indices]
    # move subgraph gpu to speed up
    original_device = sampled_features.device
    device = original_device if device is None else device
    sampled_features = sampled_features.to(device)

    # compute affinity matrix on subgraph
    A = affinity_from_features(
        sampled_features, affinity_focal_gamma=affinity_focal_gamma, 
        distance=distance,
    )

    # check if all nodes are sampled, if so, no need for Nystrom approximation
    all_indices = torch.zeros(features.shape[0], dtype=torch.bool)
    all_indices[sampled_indices] = True
    not_sampled = ~all_indices
    _n_not_sampled = not_sampled.sum()
    

    if _n_not_sampled == 0:
        # if sampled all nodes, no need for nyström approximation
        eigen_vector, eigen_value = ncut(A, num_eig, eig_solver=eig_solver)
        return eigen_vector, eigen_value, sampled_indices

    # 1) PCA to reduce the node dimension for the not sampled nodes
    # 2) compute indirect connection on the PC nodes
    if _n_not_sampled > 0 and indirect_connection:
        indirect_pca_dim = min(indirect_pca_dim, min(*features.shape))
        U, S, V = torch.pca_lowrank(features[not_sampled].T, q=indirect_pca_dim)
        S = S / math.sqrt(_n_not_sampled)
        feature_B_T = U @ torch.diag(S)
        feature_B = feature_B_T.T
        feature_B = feature_B.to(device)
        
        B = affinity_from_features(
            sampled_features,
            feature_B,
            affinity_focal_gamma=affinity_focal_gamma,
            distance=distance,
            fill_diagonal=False,
        )
        # P is 1-hop random walk matrix
        B_row = B / B.sum(axis=1, keepdim=True)
        B_col = B / B.sum(axis=0, keepdim=True)
        P = B_row @ B_col.T
        P = (P + P.T) / 2
        # fill diagonal with 0
        P[torch.arange(P.shape[0]), torch.arange(P.shape[0])] = 0
        A = A + P

    # compute normalized cut on the subgraph
    eigen_vector, eigen_value = ncut(A, num_eig, eig_solver=eig_solver)
    eigen_vector = eigen_vector.to(dtype=features.dtype, device=original_device)
    eigen_value = eigen_value.to(dtype=features.dtype, device=original_device)

    if no_propagation:
        return eigen_vector, eigen_value, sampled_indices

    # propagate eigenvectors from subgraph to full graph
    eigen_vector = propagate_knn(
        eigen_vector,
        features,
        sampled_features,
        knn,
        distance=distance,
        chunk_size=matmul_chunk_size,
        device=device,
        use_tqdm=verbose,
    )

    # post-hoc orthogonalization
    if make_orthogonal:
        eigen_vector = gram_schmidt(eigen_vector)

    return eigen_vector, eigen_value, sampled_indices


def affinity_from_features(
    features : torch.Tensor,
    features_B : torch.Tensor = None,
    affinity_focal_gamma : float = 1.0,
    distance : Literal["cosine", "euclidean", "rbf"] = "cosine",
    fill_diagonal : bool = True,
):
    """Compute affinity matrix from input features.

    Args:
        features (torch.Tensor): input features, shape (n_samples, n_features)
        feature_B (torch.Tensor, optional): optional, if not None, compute affinity between two features
        affinity_focal_gamma (float): affinity matrix parameter, lower t reduce the edge weights
            on weak connections, default 1.0
        distance (str): distance metric, 'cosine' (default) or 'euclidean', 'rbf'.
        normalize_features (bool): normalize input features before computing affinity matrix

    Returns:
        (torch.Tensor): affinity matrix, shape (n_samples, n_samples)
    """
    # compute affinity matrix from input features

    # if feature_B is not provided, compute affinity matrix on features x features
    # if feature_B is provided, compute affinity matrix on features x feature_B
    if features_B is not None:
        assert not fill_diagonal, "fill_diagonal should be False when feature_B is None"
    features_B = features if features_B is None else features_B

    if distance == "cosine":
        if not check_if_normalized(features):
            features = F.normalize(features, dim=-1)
        if not check_if_normalized(features_B):
            features_B = F.normalize(features_B, dim=-1)
        A = 1 - features @ features_B.T
    elif distance == "euclidean":
        A = torch.cdist(features, features_B, p=2)
    elif distance == "rbf":
        d = torch.cdist(features, features_B, p=2)
        A = torch.pow(d, 2)
    else:
        raise ValueError("distance should be 'cosine' or 'euclidean', 'rbf'")

    if fill_diagonal:
        A[torch.arange(A.shape[0]), torch.arange(A.shape[0])] = 0

    # torch.exp make affinity matrix positive definite,
    # lower affinity_focal_gamma reduce the weak edge weights
    if distance != "rbf":
        A = torch.exp(-A / affinity_focal_gamma)
    if distance == "rbf":
        sigma = 2 * affinity_focal_gamma * features.var(dim=0).sum()
        A = torch.exp(-A / sigma)
    return A


def ncut(
    A : torch.Tensor,
    num_eig : int = 100,
    eig_solver : Literal["svd_lowrank", "lobpcg", "svd", "eigh"] = "svd_lowrank",
):
    """PyTorch implementation of Normalized cut without Nystrom-like approximation.

    Args:
        A (torch.Tensor): affinity matrix, shape (n_samples, n_samples)
        num_eig (int): number of eigenvectors to return
        eig_solver (str): eigen decompose solver, ['svd_lowrank', 'lobpcg', 'svd', 'eigh']

    Returns:
        (torch.Tensor): eigenvectors corresponding to the eigenvalues, shape (n_samples, num_eig)
        (torch.Tensor): eigenvalues of the eigenvectors, sorted in descending order
    """

    # make sure A is symmetric
    A = (A + A.T) / 2

    # symmetrical normalization; A = D^(-1/2) A D^(-1/2)
    D = A.sum(dim=-1).detach().clone()
    A /= torch.sqrt(D)[:, None]
    A /= torch.sqrt(D)[None, :]

    # compute eigenvectors
    if eig_solver == "svd_lowrank":  # default
        # only top q eigenvectors, fastest
        eigen_vector, eigen_value, _ = torch.svd_lowrank(A, q=num_eig)
    elif eig_solver == "lobpcg":
        # only top k eigenvectors, fast
        eigen_value, eigen_vector = torch.lobpcg(A, k=num_eig)
    elif eig_solver == "svd":
        # all eigenvectors, slow
        eigen_vector, eigen_value, _ = torch.svd(A)
    elif eig_solver == "eigh":
        # all eigenvectors, slow
        eigen_value, eigen_vector = torch.linalg.eigh(A)
    else:
        raise ValueError(
            "eigen_solver should be 'lobpcg', 'svd_lowrank', 'svd' or 'eigh'"
        )

    # sort eigenvectors by eigenvalues, take top (descending order)
    eigen_value = eigen_value.real
    eigen_vector = eigen_vector.real
    sort_order = torch.argsort(eigen_value, descending=True)[:num_eig]
    eigen_value = eigen_value[sort_order]
    eigen_vector = eigen_vector[:, sort_order]

    # correct the random rotation (flipping sign) of eigenvectors
    eigen_vector = correct_rotation(eigen_vector)

    if eigen_value.min() < 0:
        logging.warning(
            "negative eigenvalues detected, please make sure the affinity matrix is positive definite"
        )

    return eigen_vector, eigen_value


def rgb_from_tsne_3d(
    features : torch.Tensor,
    num_sample : int = 300,
    perplexity : int = 150,
    metric : Literal["cosine", "euclidean"] = "cosine",
    device : str = None,
    seed : int = 0,
    q : float = 0.95,
    knn : int = 10,
    **kwargs,
):
    """

    Returns:
        (torch.Tensor): Embedding in 3D, shape (n_samples, 3)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError(
            "sklearn import failed, please install `pip install scikit-learn`"
        )

    subgraph_indices = run_subgraph_sampling(
        features,
        num_sample=num_sample,
        sample_method="farthest",
    )
    if perplexity > len(subgraph_indices) // 2:
        logging.warning(
            f"perplexity is larger than num_sample, set perplexity to {len(subgraph_indices)//2}"
        )
        perplexity = len(subgraph_indices) // 2
    _inp = features[subgraph_indices].cpu().numpy()
    _subgraph_embed = TSNE(
        n_components=3,
        perplexity=perplexity,
        metric=metric,
        random_state=seed,
    ).fit_transform(_inp)

    _subgraph_embed = torch.tensor(_subgraph_embed, dtype=torch.float32)
    embedding = propagate_knn(
        _subgraph_embed,
        features,
        features[subgraph_indices],
        distance=metric,
        knn=knn,
        device=device,
        move_output_to_cpu=True,
    )

    X_3d = embedding.cpu().numpy()
    rgb = rgb_from_3d_rgb_cube(torch.tensor(X_3d), q=q)

    return X_3d, rgb


def rgb_from_tsne_2d(
    features,
    num_sample=300,
    perplexity=150,
    metric="cosine",
    device=None,
    seed=0,
    q=0.95,
    knn=10,
    **kwargs,
):
    """

    Returns:
        (torch.Tensor): Embedding in 2D, shape (n_samples, 2)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError(
            "sklearn import failed, please install `pip install scikit-learn`"
        )

    subgraph_indices = run_subgraph_sampling(
        features,
        num_sample=num_sample,
        sample_method="farthest",
    )
    if perplexity > len(subgraph_indices) // 2:
        logging.warning(
            f"perplexity is larger than num_sample, set perplexity to {len(subgraph_indices)//2}"
        )
        perplexity = len(subgraph_indices) // 2
    _inp = features[subgraph_indices].cpu().numpy()
    _subgraph_embed = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric=metric,
        random_state=seed,
    ).fit_transform(_inp)

    _subgraph_embed = torch.tensor(_subgraph_embed, dtype=torch.float32)
    embedding = propagate_knn(
        _subgraph_embed,
        features,
        features[subgraph_indices],
        distance=metric,
        knn=knn,
        device=device,
        move_output_to_cpu=True,
    )

    X_2d = embedding.cpu().numpy()
    rgb = rgb_from_2d_colormap(torch.tensor(X_2d), q=q)

    return X_2d, rgb


def rgb_from_umap_2d(
    features,
    num_sample=300,
    n_neighbors=150,
    min_dist=0.1,
    metric="cosine",
    device=None,
    seed=0,
    q=0.95,
    knn=10,
    **kwargs,
):
    """

    Returns:
        (torch.Tensor): Embedding in 2D, shape (n_samples, 2)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap import failed, please install `pip install umap-learn`")

    subgraph_indices = run_subgraph_sampling(
        features,
        num_sample=num_sample,
        sample_method="farthest",
    )
    _inp = features[subgraph_indices].cpu().numpy()
    _subgraph_embed = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    ).fit_transform(_inp)

    _subgraph_embed = torch.tensor(_subgraph_embed, dtype=torch.float32)
    embedding = propagate_knn(
        _subgraph_embed,
        features,
        features[subgraph_indices],
        distance=metric,
        knn=knn,
        device=device,
        move_output_to_cpu=True,
    )

    X_2d = embedding.cpu().numpy()
    rgb = rgb_from_2d_colormap(torch.tensor(X_2d), q=q)

    return X_2d, rgb


def rgb_from_umap_sphere(
    features,
    num_sample=300,
    n_neighbors=150,
    min_dist=0.1,
    metric="cosine",
    device=None,
    seed=0,
    q=0.95,
    knn=10,
    **kwargs,
):
    """

    Returns:
        (torch.Tensor): Embedding in 3D, shape (n_samples, 3)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap import failed, please install `pip install umap-learn`")

    subgraph_indices = run_subgraph_sampling(
        features,
        num_sample=num_sample,
        sample_method="farthest",
    )
    _inp = features[subgraph_indices].cpu().numpy()
    _subgraph_embed = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        output_metric="haversine",
        random_state=seed,
    ).fit_transform(_inp)

    _subgraph_embed = torch.tensor(_subgraph_embed, dtype=torch.float32)
    embedding = propagate_knn(
        _subgraph_embed,
        features,
        features[subgraph_indices],
        distance=metric,
        knn=knn,
        device=device,
        move_output_to_cpu=True,
    )

    x = np.sin(embedding[:, 0]) * np.cos(embedding[:, 1])
    y = np.sin(embedding[:, 0]) * np.sin(embedding[:, 1])
    z = np.cos(embedding[:, 0])

    X_3d = np.stack([x, y, z], axis=1)
    rgb = rgb_from_3d_rgb_cube(torch.tensor(X_3d), q=q)

    return X_3d, rgb


def rgb_from_umap_3d(
    features,
    num_sample=300,
    n_neighbors=150,
    min_dist=0.1,
    metric="cosine",
    device=None,
    seed=0,
    q=0.95,
    knn=10,
    **kwargs,
):
    """

    Returns:
        (torch.Tensor): Embedding in 3D, shape (n_samples, 3)
        (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap import failed, please install `pip install umap-learn`")

    subgraph_indices = run_subgraph_sampling(
        features,
        num_sample=num_sample,
        sample_method="farthest",
    )
    _inp = features[subgraph_indices].cpu().numpy()
    _subgraph_embed = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    ).fit_transform(_inp)

    _subgraph_embed = torch.tensor(_subgraph_embed, dtype=torch.float32)
    X_3d = propagate_knn(
        _subgraph_embed,
        features,
        features[subgraph_indices],
        distance=metric,
        knn=knn,
        device=device,
        move_output_to_cpu=True,
    )

    rgb = rgb_from_3d_rgb_cube(torch.tensor(X_3d), q=q)

    return X_3d, rgb


def flatten_sphere(X_3d):
    x = np.arctan2(X_3d[:, 0], X_3d[:, 1])
    y = -np.arccos(X_3d[:, 2])
    X_2d = np.stack([x, y], axis=1)
    return X_2d


def rotate_rgb_cube(rgb, position=1):
    """rotate RGB cube to different position

    Args:
        rgb (torch.Tensor): RGB color space [0, 1], shape (*, 3)
        position (int): position to rotate, 0, 1, 2, 3, 4, 5, 6

    Returns:
        torch.Tensor: RGB color space, shape (n_samples, 3)
    """
    assert position in range(0, 7), "position should be 0, 1, 2, 3, 4, 5, 6"
    rotation_matrix = torch.tensor(
        [
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ]
    ).float()
    n_mul = position % 3
    rotation_matrix = torch.matrix_power(rotation_matrix, n_mul)
    rgb = rgb @ rotation_matrix
    if position > 3:
        rgb = 1 - rgb
    return rgb


def farthest_point_sampling(
    features,
    num_sample=300,
    h=9,
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

@torch.no_grad()
def run_subgraph_sampling(
    features,
    num_sample=300,
    max_draw=1000000,
    sample_method="farthest",
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


def gram_schmidt(matrix):
    """Orthogonalize a matrix column-wise using the Gram-Schmidt process.

    Args:
        matrix (torch.Tensor): A matrix to be orthogonalized (m x n).
            the second dimension is orthogonalized
    Returns:
        torch.Tensor: Orthogonalized matrix (m x n).
    """
    
    # Get the number of rows (m) and columns (n) of the input matrix
    m, n = matrix.shape

    # Create an empty matrix to store the orthogonalized columns
    orthogonal_matrix = torch.zeros((m, n), dtype=matrix.dtype)

    for i in range(n):
        # Start with the i-th column of the input matrix
        vec = matrix[:, i]

        for j in range(i):
            # Subtract the projection of vec onto the j-th orthogonal column
            proj = torch.dot(orthogonal_matrix[:, j], matrix[:, i]) / torch.dot(
                orthogonal_matrix[:, j], orthogonal_matrix[:, j]
            )
            vec = vec - proj * orthogonal_matrix[:, j]

        # Store the orthogonalized vector
        orthogonal_matrix[:, i] = vec / torch.norm(vec)

    return orthogonal_matrix


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


def rgb_from_3d_rgb_cube(X_3d, q=0.95):
    """convert 3D t-SNE to RGB color space

    Args:
        X_3d (torch.Tensor): 3D t-SNE embedding, shape (n_samples, 3)
        q (float): quantile, default 0.95

    Returns:
        torch.Tensor: RGB color space, shape (n_samples, 3)
    """
    assert X_3d.shape[1] == 3, "input should be (n_samples, 3)"
    assert len(X_3d.shape) == 2, "input should be (n_samples, 3)"
    rgb = []
    for i in range(3):
        rgb.append(quantile_normalize(X_3d[:, i], q=q))
    rgb = torch.stack(rgb, dim=-1)
    return rgb


def rgb_from_2d_colormap(X_2d, q=0.95):
    xy = X_2d.clone()
    for i in range(2):
        xy[:, i] = quantile_normalize(xy[:, i], q=q)

    try:
        from pycolormap_2d import (
            ColorMap2DBremm,
            ColorMap2DZiegler,
            ColorMap2DCubeDiagonal,
            ColorMap2DSchumann,
        )
    except ImportError:
        raise ImportError(
            "pycolormap_2d import failed, please install `pip install pycolormap-2d`"
        )

    cmap = ColorMap2DCubeDiagonal()
    xy = xy.cpu().numpy()
    len_x, len_y = cmap._cmap_data.shape[:2]
    x = (xy[:, 0] * (len_x - 1)).astype(int)
    y = (xy[:, 1] * (len_y - 1)).astype(int)
    rgb = cmap._cmap_data[x, y]
    rgb = torch.tensor(rgb, dtype=torch.float32) / 255
    return rgb


def correct_rotation(eigen_vector):
    # correct the random rotation (flipping sign) of eigenvectors
    rand_w = torch.ones(
        eigen_vector.shape[0], device=eigen_vector.device, dtype=eigen_vector.dtype
    )
    s = rand_w[None, :] @ eigen_vector
    s = s.sign()
    return eigen_vector * s


def propagate_knn(
    subgraph_output,
    inp_features,
    subgraph_features,
    knn=10,
    distance="cosine",
    chunk_size=8096,
    device=None,
    use_tqdm=False,
    move_output_to_cpu=False,
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
    eigenvectors,
    features,
    new_features,
    knn=10,
    num_sample=3000,
    sample_method="farthest",
    chunk_size=8096,
    device=None,
    use_tqdm=False,
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


def propagate_rgb_color(
    rgb,
    eigenvectors,
    new_eigenvectors,
    knn=10,
    num_sample=300,
    sample_method="farthest",
    chunk_size=8096,
    device=None,
    use_tqdm=False,
):
    """Propagate RGB color to new nodes using KNN.

    Args:
        rgb (torch.Tensor): RGB color for each data sample, shape (n_samples, 3)
        features (torch.Tensor): features from existing nodes, shape (n_samples, n_features)
        new_features (torch.Tensor): features from new nodes, shape (n_new_samples, n_features)
        knn (int): number of KNN to propagate RGB color, default 1
        num_sample (int): number of samples for subgraph sampling, default 50000
        sample_method (str): sample method, 'farthest' (default) or 'random'
        chunk_size (int): chunk size for matrix multiplication, default 8096
        device (str): device to use for computation, if None, will not change device
        use_tqdm (bool): show progress bar when propagating RGB color from subgraph to full graph

    Returns:
        torch.Tensor: propagated RGB color for each data sample, shape (n_new_samples, 3)

    Examples:
        >>> old_rgb = torch.randn(3000, 3)
        >>> old_eigenvectors = torch.randn(3000, 20)
        >>> new_eigenvectors = torch.randn(200, 20)
        >>> new_rgb = propagate_rgb_color(old_rgb, new_eigenvectors, old_eigenvectors)
        >>> # new_eigenvectors.shape = (200, 3)
    """
    device = rgb.device if device is None else device

    # sample subgraph
    subgraph_indices = run_subgraph_sampling(
        eigenvectors,
        num_sample=num_sample,
        sample_method=sample_method,
    )

    subgraph_rgb = rgb[subgraph_indices].to(device)
    subgraph_eigenvectors = eigenvectors[subgraph_indices].to(device)
    new_eigenvectors = new_eigenvectors.to(device)

    # propagate RGB color from subgraph to new nodes
    new_rgb = propagate_knn(
        subgraph_rgb,
        new_eigenvectors,
        subgraph_eigenvectors,
        knn=knn,
        chunk_size=chunk_size,
        device=device,
        use_tqdm=use_tqdm,
    )

    return new_rgb


# Multiclass Spectral Clustering, SX Yu, J Shi, 2003

def _discretisation_eigenvector(eigen_vector):
    # Function that discretizes rotated eigenvectors
    n, k = eigen_vector.shape

    # Find the maximum index along each row
    _, J = torch.max(eigen_vector, dim=1)
    Y = torch.zeros(n, k, device=eigen_vector.device).scatter_(1, J.unsqueeze(1), 1)

    return Y

def kway_ncut(eigen_vectors:torch.Tensor, max_iter=300, return_rotation=False):
    """Multiclass Spectral Clustering, SX Yu, J Shi, 2003

    Args:
        eigen_vectors (torch.Tensor): continuous eigenvectors from NCUT, shape (n, k)
        max_iter (int, optional): Maximum number of iterations.

    Returns:
        torch.Tensor: Discretized eigenvectors, shape (n, k), each row is a one-hot vector.
    """
    # Normalize eigenvectors
    n, k = eigen_vectors.shape
    vm = torch.sqrt(torch.sum(eigen_vectors ** 2, dim=1))
    eigen_vectors = eigen_vectors / vm.unsqueeze(1)

    # Initialize R matrix with the first column from a random row of EigenVectors
    R = torch.zeros(k, k, device=eigen_vectors.device)
    R[:, 0] = eigen_vectors[torch.randint(0, n, (1,))].squeeze()

    # Loop to populate R with k orthogonal directions
    c = torch.zeros(n, device=eigen_vectors.device)
    for j in range(1, k):
        c += torch.abs(eigen_vectors @ R[:, j - 1])
        _, i = torch.min(c, dim=0)
        R[:, j] = eigen_vectors[i]

    # Iterative optimization loop
    last_objective_value = 0
    exit_loop = False
    nb_iterations_discretisation = 0

    while not exit_loop:
        nb_iterations_discretisation += 1

        # Discretize the projected eigenvectors
        eigenvectors_discrete = _discretisation_eigenvector(eigen_vectors @ R)

        # SVD decomposition
        U, S, Vh = torch.linalg.svd(eigenvectors_discrete.T @ eigen_vectors, full_matrices=False)
        V = Vh.T

        # Compute the Ncut value
        ncut_value = 2 * (n - torch.sum(S))

        # Check for convergence
        if torch.abs(ncut_value - last_objective_value) < torch.finfo(torch.float32).eps or nb_iterations_discretisation > max_iter:
            exit_loop = True
        else:
            last_objective_value = ncut_value
            R = V @ U.T

    if return_rotation:
        return eigenvectors_discrete, R

    return eigenvectors_discrete


def axis_align(eigen_vectors, max_iter=300):
    return kway_ncut(eigen_vectors, max_iter=max_iter, return_rotation=True)


# application: get segmentation mask fron a reference eigenvector (point prompt)

def _transform_heatmap(heatmap, gamma=1.0):
    """Transform the heatmap using gamma, normalize and min-max normalization.

    Args:
        heatmap (torch.Tensor): distance heatmap, shape (B, H, W)
        gamma (float, optional): scaling factor, higher means smaller mask. Defaults to 1.0.

    Returns:
        torch.Tensor: transformed heatmap, shape (B, H, W)
    """
    # normalize the heatmap
    heatmap = (heatmap - heatmap.mean()) / heatmap.std()
    heatmap = torch.exp(heatmap)
    # transform the heatmap using gamma
    # large gamma means more focus on the high values, hence smaller mask
    heatmap = 1 / heatmap ** gamma
    # min-max normalization [0, 1]
    vmin, vmax = quantile_min_max(heatmap.flatten())
    heatmap = (heatmap - vmin) / (vmax - vmin)
    return heatmap


def _clean_mask(mask, min_area=500):
    """clean the binary mask by removing small connected components.
    
    Args:
    - mask: A numpy image of a binary mask with 255 for the object and 0 for the background.
    - min_area: Minimum area for a connected component to be considered valid (default 500).
    
    Returns:
    - bounding_boxes: List of bounding boxes for valid objects (x, y, width, height).
    - cleaned_pil_mask: A Pillow image of the cleaned mask, with small components removed.
    """
    
    import cv2
    # Find connected components in the cleaned mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Initialize an empty mask to store the final cleaned mask
    final_cleaned_mask = np.zeros_like(mask)

    # Collect bounding boxes for components that are larger than the threshold and update the cleaned mask
    bounding_boxes = []
    for i in range(1, num_labels):  # Skip label 0 (background)
        x, y, w, h, area = stats[i]
        if area >= min_area:
            # Add the bounding box of the valid component
            bounding_boxes.append((x, y, w, h))
            # Keep the valid components in the final cleaned mask
            final_cleaned_mask[labels == i] = 255

    return final_cleaned_mask, bounding_boxes


def get_mask(
        all_eigvecs : torch.Tensor, prompt_eigvec: torch.Tensor, 
        threshold : float=0.5, gamma : float=1.0, 
        denoise : bool=True, denoise_area_th : int=3):
    """Segmentation mask from one prompt eigenvector (at a clicked latent pixel).
        </br> The mask is computed by measuring the cosine similarity between the clicked eigenvector and all the eigenvectors in the latent space.
        </br> 1. Compute the cosine similarity between the clicked eigenvector and all the eigenvectors in the latent space.
        </br> 2. Transform the heatmap, normalize and apply scaling (gamma).
        </br> 3. Threshold the heatmap to get the mask.
        </br> 4. Optionally denoise the mask by removing small connected components
        
    Args:
        all_eigvecs (torch.Tensor): (B, H, W, num_eig)
        prompt_eigvec (torch.Tensor): (num_eig,)
        threshold (float, optional): mask threshold, higher means smaller mask. Defaults to 0.5.
        gamma (float, optional): mask scaling factor, higher means smaller mask. Defaults to 1.0.
        denoise (bool, optional): mask denoising flag. Defaults to True.
        denoise_area_th (int, optional): mask denoising area threshold. higher means more aggressive denoising. Defaults to 3.
    
    Returns:
        np.ndarray: masks (B, H, W), 1 for object, 0 for background
        
    Examples:
        >>> all_eigvecs = torch.randn(10, 64, 64, 20)
        >>> prompt_eigvec = all_eigvecs[0, 32, 32]  # center pixel
        >>> masks = get_mask(all_eigvecs, prompt_eigvec, threshold=0.5, gamma=1.0, denoise=True, denoise_area_th=3)
        >>> # masks.shape = (10, 64, 64)
    """
    
    # normalize the eigenvectors to unit norm, to compute cosine similarity
    if not check_if_normalized(all_eigvecs.reshape(-1, all_eigvecs.shape[-1])):
        all_eigvecs = F.normalize(all_eigvecs, p=2, dim=-1)
        
    prompt_eigvec = F.normalize(prompt_eigvec, p=2, dim=-1)
    
    # compute the cosine similarity
    cos_sim = all_eigvecs @ prompt_eigvec.unsqueeze(-1)  # (B, H, W, 1)
    cos_sim = cos_sim.squeeze(-1)  # (B, H, W)
    
    heatmap = 1 - cos_sim
    
    # transform the heatmap, normalize and apply scaling (gamma)
    heatmap = _transform_heatmap(heatmap, gamma=gamma)
    
    masks = heatmap > threshold
    masks = masks.cpu().numpy().astype(np.uint8)
    
    if denoise:
        cleaned_masks = []
        for mask in masks:
            cleaned_mask, _ = _clean_mask(mask, min_area=denoise_area_th)
            cleaned_masks.append(cleaned_mask)
        cleaned_masks = np.stack(cleaned_masks)
        return cleaned_masks
    
    return masks

