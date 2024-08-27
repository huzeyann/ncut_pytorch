# %%
import logging
import torch
import torch.nn.functional as F
import numpy as np


class NCUT:
    """Nystrom Normalized Cut for large scale graph."""

    def __init__(
        self,
        num_eig=100,
        knn=10,
        affinity_focal_gamma=1.0,
        num_sample=10000,
        sample_method="farthest",
        distance="cosine",
        indirect_connection=True,
        indirect_pca_dim=100,
        device=None,
        eig_solver="svd_lowrank",
        normalize_features=True,
        matmul_chunk_size=8096,
        make_orthogonal=False,
        verbose=False,
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
            distance (str): distance metric for affinity matrix, ['cosine', 'euclidean'].
            indirect_connection (bool): include indirect connection in the Nystrom-like approximation
            indirect_pca_dim (int): when compute indirect connection, PCA to reduce the node dimension,
            device (str): device to use for eigen computation,
                move to GPU to speeds up a bit (~5x faster)
            eig_solver (str): eigen decompose solver, ['svd_lowrank', 'lobpcg', 'svd', 'eigh'].
            normalize_features (bool): normalize input features before computing affinity matrix
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
        self.eig_solver = eig_solver
        self.normalize_features = normalize_features
        self.matmul_chunk_size = matmul_chunk_size
        self.make_orthogonal = make_orthogonal
        self.verbose = verbose

    @torch.no_grad()
    def fit(self, features):
        """Fit Nystrom Normalized Cut on the input features.

        Args:
            features (torch.Tensor): input features, shape (n_samples, n_features)

        Returns:
            (NCUT): self
        """
        self.subgraph_eigen_vector, self.eigen_value, self.subgraph_indices = (
            nystrom_ncut(
                features,
                num_eig=self.num_eig,
                num_sample=self.num_sample,
                sample_method=self.sample_method,
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

    @torch.no_grad()
    def transform(self, features):
        """Transform new features using the fitted Nystrom Normalized Cut.

        Args:
            features (torch.Tensor): new features, shape (n_samples, n_features)

        Returns:
            (torch.Tensor): eigen_vectors, shape (n_samples, num_eig)
            (torch.Tensor): eigen_values, sorted in descending order, shape (num_eig,)
        """
        eigen_vector = propagate_knn(
            self.subgraph_eigen_vector,
            features,
            self.subgraph_features,
            self.knn,
            chunk_size=self.matmul_chunk_size,
            device=self.device,
            use_tqdm=self.verbose,
        )
        if self.make_orthogonal:
            eigen_vector = gram_schmidt(eigen_vector)
        return eigen_vector, self.eigen_value

    def fit_transform(self, features):
        """

        Args:
            features (torch.Tensor): input features, shape (n_samples, n_features)

        Returns:
            (torch.Tensor): eigen_vectors, shape (n_samples, num_eig)
            (torch.Tensor): eigen_values, sorted in descending order, shape (num_eig,)
        """
        return self.fit(features).transform(features)


def eigenvector_to_rgb(
    eigen_vector,
    method="tsne_3d",
    num_sample=300,
    perplexity=150,
    n_neighbors=150,
    min_distance=0.1,
    metric="euclidean",
    device=None,
    q=0.95,
    knn=10,
    seed=0,
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
        metric (str): distance metric, default 'euclidean'
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
    if method == "tsne_2d":
        embed, rgb = rgb_from_tsne_2d(
            eigen_vector,
            q=q,
            num_sample=num_sample,
            perplexity=perplexity,
            seed=seed,
            device=device,
            metric=metric,
            knn=knn,
        )
    elif method == "tsne_3d":
        embed, rgb = rgb_from_tsne_3d(
            eigen_vector,
            q=q,
            num_sample=num_sample,
            perplexity=perplexity,
            seed=seed,
            device=device,
            metric=metric,
            knn=knn,
        )
    elif method == "umap_sphere":
        embed, rgb = rgb_from_umap_sphere(
            eigen_vector,
            q=q,
            num_sample=num_sample,
            n_neighbors=n_neighbors,
            min_dist=min_distance,
            seed=seed,
            device=device,
            metric=metric,
            knn=knn,
        )
    elif method == "umap_2d":
        embed, rgb = rgb_from_umap_2d(
            eigen_vector,
            q=q,
            num_sample=num_sample,
            n_neighbors=n_neighbors,
            min_dist=min_distance,
            seed=seed,
            device=device,
            metric=metric,
            knn=knn,
        )
    elif method == "umap_3d":
        embed, rgb = rgb_from_umap_3d(
            eigen_vector,
            q=q,
            num_sample=num_sample,
            n_neighbors=n_neighbors,
            min_dist=min_distance,
            seed=seed,
            device=device,
            metric=metric,
            knn=knn,
        )
    else:
        raise ValueError("method should be 'tsne_2d', 'tsne_3d' or 'umap_sphere'")

    return embed, rgb


def nystrom_ncut(
    features,
    num_eig=100,
    num_sample=10000,
    knn=10,
    sample_method="farthest",
    distance="cosine",
    affinity_focal_gamma=1.0,
    indirect_connection=True,
    indirect_pca_dim=100,
    device=None,
    eig_solver="svd_lowrank",
    normalize_features=True,
    matmul_chunk_size=8096,
    make_orthogonal=False,
    verbose=False,
    no_propagation=False,
):
    """PyTorch implementation of Faster Nystrom Normalized cut.

    Args:
        features (torch.Tensor): feature matrix, shape (n_samples, n_features)
        num_eig (int): default 20, number of top eigenvectors to return
        num_sample (int): default 30000, number of samples for Nystrom-like approximation
        knn (int): default 3, number of KNN for propagating eigenvectors from subgraph to full graph,
            smaller knn will result in more sharp eigenvectors,
        sample_method (str): sample method, 'farthest' (default) or 'random'
            'farthest' is recommended for better approximation
        distance (str): distance metric, 'cosine' (default) or 'euclidean'
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
            default True
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

    features = features.clone()
    if normalize_features:
        # features need to be normalized for affinity matrix computation (cosine distance)
        features = torch.nn.functional.normalize(features, dim=-1)

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
        sampled_features, affinity_focal_gamma=affinity_focal_gamma, distance=distance
    )

    not_sampled = torch.tensor(
        list(set(range(features.shape[0])) - set(sampled_indices))
    )

    if len(not_sampled) == 0:
        # if sampled all nodes, no need for nyström approximation
        eigen_vector, eigen_value = ncut(A, num_eig, eig_solver=eig_solver)
        return eigen_vector, eigen_value, sampled_indices

    # 1) PCA to reduce the node dimension for the not sampled nodes
    # 2) compute indirect connection on the PC nodes
    if len(not_sampled) > 0 and indirect_connection:
        indirect_pca_dim = min(indirect_pca_dim, min(*features.shape))
        U, S, V = torch.pca_lowrank(features[not_sampled].T, q=indirect_pca_dim)
        feature_B = (features[not_sampled].T @ V).T  # project to PCA space
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
        sampled_indices,
        knn,
        chunk_size=matmul_chunk_size,
        device=device,
        use_tqdm=verbose,
    )

    # post-hoc orthogonalization
    if make_orthogonal:
        eigen_vector = gram_schmidt(eigen_vector)

    return eigen_vector, eigen_value, sampled_indices


def affinity_from_features(
    features,
    features_B=None,
    affinity_focal_gamma=1.0,
    distance="cosine",
    normalize_features=True,
    fill_diagonal=True,
):
    """Compute affinity matrix from input features.

    Args:
        features (torch.Tensor): input features, shape (n_samples, n_features)
        feature_B (torch.Tensor, optional): optional, if not None, compute affinity between two features
        affinity_focal_gamma (float): affinity matrix parameter, lower t reduce the edge weights
            on weak connections, default 1.0
        distance (str): distance metric, 'cosine' (default) or 'euclidean'.
        apply_normalize (bool): normalize input features before computing affinity matrix,
            default True

    Returns:
        (torch.Tensor): affinity matrix, shape (n_samples, n_samples)
    """
    # compute affinity matrix from input features
    features = features.clone()
    if features_B is not None:
        features_B = features_B.clone()

    # if feature_B is not provided, compute affinity matrix on features x features
    # if feature_B is provided, compute affinity matrix on features x feature_B
    if features_B is not None:
        assert not fill_diagonal, "fill_diagonal should be False when feature_B is None"
    features_B = features if features_B is None else features_B

    if normalize_features:
        features = F.normalize(features, dim=-1)
        features_B = F.normalize(features_B, dim=-1)

    if distance == "cosine":
        if not check_if_normalized(features):
            features = F.normalize(features, dim=-1)
        if not check_if_normalized(features_B):
            features_B = F.normalize(features_B, dim=-1)
        A = 1 - features @ features_B.T
    elif distance == "euclidean":
        A = torch.cdist(features, features_B, p=2)
    else:
        raise ValueError("distance should be 'cosine' or 'euclidean'")

    if fill_diagonal:
        A[torch.arange(A.shape[0]), torch.arange(A.shape[0])] = 0

    # torch.exp make affinity matrix positive definite,
    # lower affinity_focal_gamma reduce the weak edge weights
    A = torch.exp(-((A / affinity_focal_gamma)))
    return A


def ncut(
    A,
    num_eig=20,
    eig_solver="svd_lowrank",
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
    features,
    num_sample=300,
    perplexity=150,
    metric="euclidean",
    device=None,
    seed=0,
    q=0.95,
    knn=10,
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
        knn=knn,
        device=device,
    )

    X_3d = embedding.cpu().numpy()
    rgb = rgb_from_3d_rgb_cube(torch.tensor(X_3d), q=q)

    return X_3d, rgb


def rgb_from_tsne_2d(
    features,
    num_sample=300,
    perplexity=150,
    metric="euclidean",
    device=None,
    seed=0,
    q=0.95,
    knn=10,
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
        knn=knn,
        device=device,
    )

    X_2d = embedding.cpu().numpy()
    rgb = rgb_from_2d_colormap(torch.tensor(X_2d), q=q)

    return X_2d, rgb


def rgb_from_umap_2d(
    features,
    num_sample=300,
    n_neighbors=150,
    min_dist=0.1,
    metric="euclidean",
    device=None,
    seed=0,
    q=0.95,
    knn=10,
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
        knn=knn,
        device=device,
    )

    X_2d = embedding.cpu().numpy()
    rgb = rgb_from_2d_colormap(torch.tensor(X_2d), q=q)

    return X_2d, rgb


def rgb_from_umap_sphere(
    features,
    num_sample=300,
    n_neighbors=150,
    min_dist=0.1,
    metric="euclidean",
    device=None,
    seed=0,
    q=0.95,
    knn=10,
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
        knn=knn,
        device=device,
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
    metric="euclidean",
    device=None,
    seed=0,
    q=0.95,
    knn=10,
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
        knn=knn,
        device=device,
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
        features = features @ v

    h = min(h, int(np.log2(features.shape[0])))

    kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(
        features.cpu().numpy(), num_sample, h
    ).astype(np.int64)
    return kdline_fps_samples_idx


def run_subgraph_sampling(
    features,
    num_sample=300,
    max_draw=1000000,
    sample_method="farthest",
):
    if num_sample > features.shape[0]:
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
                    features[draw_indices],
                    num_sample=num_sample,
                )
                sampled_indices = draw_indices[sampled_indices]
            else:
                sampled_indices = farthest_point_sampling(
                    features,
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
    if len(x) > 1e6:
        _x = x[torch.randperm(len(x))[: int(1e6)]]
    else:
        _x = x
    vmax, vmin = _x.quantile(q), _x.quantile(1 - q)
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
    chunk_size=8096,
    device=None,
    use_tqdm=False,
):
    """A generic function to propagate new nodes using KNN.
    
    Args:
        subgraph_output (torch.Tensor): output from subgraph, shape (num_sample, D)
        inp_features (torch.Tensor): features from existing nodes, shape (new_num_samples, n_features)
        subgraph_features (torch.Tensor): features from subgraph, shape (num_sample, n_features)
        knn (int): number of KNN to propagate eigenvectors
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
        _A = _v @ subgraph_features.T

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

        V_list.append(_V.cpu())

    subgraph_output = torch.cat(V_list, dim=0)

    return subgraph_output


def propagate_nearest(
    subgraph_output,
    inp_features,
    subgraph_features,
    chunk_size=8096,
    device=None,
):
    device = subgraph_output.device if device is None else device

    # used in nystrom_tsne, equivalent to propagate_by_knn with knn=1
    # propagate tSNE from subgraph to full graph
    V_list = []
    subgraph_features = subgraph_features.to(device)
    for i in range(0, inp_features.shape[0], chunk_size):
        end = min(i + chunk_size, inp_features.shape[0])
        _v = inp_features[i:end].to(device)
        _A = _v @ subgraph_features.T
        # keep top1 for each row
        top_idx = _A.argmax(dim=-1).cpu()
        _V = subgraph_output[top_idx]
        V_list.append(_V)

    subgraph_output = torch.cat(V_list, dim=0)

    return subgraph_output


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
