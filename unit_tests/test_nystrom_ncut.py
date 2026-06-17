import statistics
import time
from types import SimpleNamespace

import pytest
import torch
from ncut_pytorch import ncut_fn
from ncut_pytorch.ncuts import ncut_nystrom as nystrom_utils
from ncut_pytorch.ncuts.ncut_nystrom import _plain_ncut, nystrom_propagate
from ncut_pytorch.utils.math import gram_schmidt, keep_topk_per_row, rbf_affinity
from ncut_pytorch.utils.sigma import _find_sigma_by_degree


def _reference_gram_schmidt(matrix: torch.Tensor) -> torch.Tensor:
    """Reference implementation matching the pre-optimization behavior."""
    m, n = matrix.shape
    orthogonal_matrix = torch.zeros((m, n), dtype=matrix.dtype, device=matrix.device)

    for i in range(n):
        vec = matrix[:, i]
        for j in range(i):
            proj = torch.dot(orthogonal_matrix[:, j], matrix[:, i]) / torch.dot(
                orthogonal_matrix[:, j], orthogonal_matrix[:, j]
            )
            vec = vec - proj * orthogonal_matrix[:, j]
        orthogonal_matrix[:, i] = vec / torch.norm(vec)
    return orthogonal_matrix


def _median_runtime(fn, matrix: torch.Tensor, repeats: int = 3) -> float:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(matrix)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def _median_call_runtime(fn, repeats: int = 3) -> float:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def _candidate_find_sigma_by_degree_rbf(
    X: torch.Tensor,
    quantile_sigma: float = 0.25,
    init_sigma: float = 0.5,
    r_tol: float = 1e-2,
    max_iter: int = 100,
) -> float:
    """Candidate fast path that reuses one pairwise distance matrix across sigma search."""
    x_sq = X.pow(2).sum(dim=1, keepdim=True)
    dist2 = x_sq + x_sq.T
    dist2.addmm_(X, X.T, beta=1.0, alpha=-2.0)
    dist2.clamp_min_(0)

    scale_inv_sigma = X.std(0).sum()
    target_degree = torch.exp(dist2 * (-0.5 / (scale_inv_sigma * scale_inv_sigma))).mean(1)
    target_degree = target_degree.float().quantile(quantile_sigma).item()

    sigma = init_sigma
    current_degree = torch.exp(dist2 * (-0.5 / (sigma * sigma))).mean().item()
    low, high = 0.0, float("inf")
    tol = r_tol * target_degree
    i_iter = 0
    while abs(current_degree - target_degree) > tol and i_iter < max_iter:
        if current_degree > target_degree:
            high = sigma
            sigma = (low + sigma) / 2
        else:
            low = sigma
            sigma = sigma * 2 if high == float("inf") else (sigma + high) / 2
        current_degree = torch.exp(dist2 * (-0.5 / (sigma * sigma))).mean().item()
        i_iter += 1
    return sigma


def _reference_weighted_neighbor_sum(
    weights: torch.Tensor,
    indices: torch.Tensor,
    nystrom_out: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("nk,nkd->nd", weights, nystrom_out[indices])


class TestNystromNcut:
    """Test the nystrom_ncut module."""

    def test_ncut_fn(self, small_feature_matrix):
        """Test the ncut_fn function."""
        n_eig = 5
        eigvec, eigval = ncut_fn(small_feature_matrix, n_eig=n_eig, make_orthogonal=True)
        
        # Check shapes
        assert eigvec.shape == (small_feature_matrix.shape[0], n_eig)
        assert eigval.shape == (n_eig,)
        
        # Check that eigenvalues are sorted in descending order
        assert torch.all(eigval[:-1] >= eigval[1:])
        
        # Check that eigenvectors have unit norm
        norms = torch.norm(eigvec, dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_ncut_fn_with_no_propagation(self, small_feature_matrix):
        """Test the ncut_fn function with no_propagation=True."""
        n_eig = 5
        eigvec, eigval, indices, sigma = ncut_fn(
            small_feature_matrix, 
            n_eig=n_eig, 
            no_propagation=True
        )
        
        # Check shapes
        assert eigvec.shape[1] == n_eig
        assert eigval.shape == (n_eig,)
        assert indices.shape[0] <= small_feature_matrix.shape[0]  # Number of sampled points
        assert isinstance(sigma, float)
        
        # Check that eigenvalues are sorted in descending order
        assert torch.all(eigval[:-1] >= eigval[1:])

    def test_ncut_fn_with_affinity_diag_eps(self, small_feature_matrix):
        """Test the ncut_fn function with a small diagonal shift."""
        n_eig = 5
        eigvec, eigval = ncut_fn(
            small_feature_matrix,
            n_eig=n_eig,
            affinity_diag_eps=1e-6,
            make_orthogonal=True,
        )

        assert eigvec.shape == (small_feature_matrix.shape[0], n_eig)
        assert eigval.shape == (n_eig,)
        assert not torch.isnan(eigvec).any()
        assert not torch.isnan(eigval).any()

    def test_plain_ncut(self):
        """Test the _plain_ncut function."""
        # Create a simple affinity matrix
        n = 20
        n_eig = 5
        A = torch.rand(n, n)
        A = (A + A.T) / 2  # Make it symmetric
        
        eigvec, eigval = _plain_ncut(A, n_eig=n_eig)
        
        # Check shapes
        assert eigvec.shape == (n, n_eig)
        assert eigval.shape == (n_eig,)
        
        # Check that eigenvalues are sorted in descending order
        assert torch.all(eigval[:-1] >= eigval[1:])
        
        # Check that eigenvectors have unit norm
        norms = torch.norm(eigvec, dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_plain_ncut_affinity_diag_eps_preserves_exact_eigvecs(self):
        """A diagonal shift should preserve exact eigenvectors and shift eigenvalues uniformly."""
        torch.manual_seed(0)
        X = torch.randn(32, 6)
        A = rbf_affinity(X, sigma=0.7)
        A.requires_grad_()
        n_eig = 6
        affinity_diag_eps = 1e-4

        eigvec_base, eigval_base = _plain_ncut(
            A,
            n_eig=n_eig,
            exact_gradient=True,
        )
        eigvec_shifted, eigval_shifted = _plain_ncut(
            A,
            n_eig=n_eig,
            exact_gradient=True,
            affinity_diag_eps=affinity_diag_eps,
        )

        cosine = torch.abs(torch.sum(eigvec_base * eigvec_shifted, dim=0))
        assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-4, rtol=1e-4)
        expected_shift = torch.full_like(eigval_base, affinity_diag_eps)
        assert torch.allclose(eigval_shifted - eigval_base, expected_shift, atol=1e-4, rtol=1e-4)

    def test_nystrom_propagate(self, small_feature_matrix):
        """Test the _nystrom_propagate function."""
        # First get some eigenvectors using ncut_fn with no_propagation
        n_eig = 5
        nystrom_eigvec, _, indices, gamma = ncut_fn(
            small_feature_matrix, 
            n_eig=n_eig, 
            no_propagation=True
        )
        
        nystrom_X = small_feature_matrix[indices]
        
        # Now propagate the eigenvectors
        eigvec = nystrom_propagate(
            nystrom_eigvec,
            small_feature_matrix,
            nystrom_X,
        )
        
        # Check shape
        assert eigvec.shape == (small_feature_matrix.shape[0], n_eig)
        
        # Check that the propagated eigenvectors have reasonable values
        assert not torch.isnan(eigvec).any()
        assert not torch.isinf(eigvec).any()

    def test_ncut_fn_forwards_fixed_chunk_size_to_propagate(self, monkeypatch, small_feature_matrix):
        """ncut_fn should forward the fixed chunk-size config to nystrom_propagate."""
        recorded_kwargs = {}

        def fake_nystrom_propagate(nystrom_out, X, nystrom_X, **kwargs):
            recorded_kwargs.update(kwargs)
            return torch.zeros((X.shape[0], nystrom_out.shape[1]), dtype=nystrom_out.dtype, device=X.device)

        monkeypatch.setattr(nystrom_utils, "nystrom_propagate", fake_nystrom_propagate)

        eigvec, eigval = ncut_fn(
            small_feature_matrix,
            n_eig=3,
            n_sample2=64,
            chunk_size=2048,
        )

        assert eigvec.shape == (small_feature_matrix.shape[0], 3)
        assert eigval.shape == (3,)
        assert recorded_kwargs["n_sample2"] == 64
        assert recorded_kwargs["chunk_size"] == 2048

    def test_nystrom_propagate_uses_fixed_chunk_size(self, monkeypatch):
        """nystrom_propagate should split the input by the configured fixed chunk size."""
        chunk_sizes = []
        X = torch.randn(17, 4)
        nystrom_X = torch.randn(8, 4)
        nystrom_out = torch.randn(8, 3)

        monkeypatch.setattr(nystrom_utils, "find_sigma_by_degree", lambda *args, **kwargs: 1.0)

        def fake_rbf_topk_from_squared_distance(Xi, sampled_X, sampled_x_sq, sigma, n_neighbors):
            chunk_sizes.append(Xi.shape[0])
            weights = torch.ones((Xi.shape[0], n_neighbors), device=Xi.device, dtype=Xi.dtype)
            indices = torch.zeros((Xi.shape[0], n_neighbors), device=Xi.device, dtype=torch.long)
            return weights, indices

        def fake_weighted_neighbor_sum(weights, indices, sampled_out, offsets_cache):
            return torch.zeros((weights.shape[0], sampled_out.shape[-1]), device=weights.device, dtype=sampled_out.dtype)

        monkeypatch.setattr(nystrom_utils, "_rbf_topk_from_squared_distance", fake_rbf_topk_from_squared_distance)
        monkeypatch.setattr(nystrom_utils, "_weighted_neighbor_sum", fake_weighted_neighbor_sum)

        out = nystrom_propagate(
            nystrom_out,
            X,
            nystrom_X,
            device="cpu",
            n_sample2=nystrom_out.shape[0],
            n_neighbors=4,
            chunk_size=5,
        )

        assert out.shape == (X.shape[0], nystrom_out.shape[1])
        assert chunk_sizes == [5, 5, 5, 2]

    def test_nystrom_propagate_reuses_precomputed_cache(self, monkeypatch):
        """nystrom_propagate should reuse sigma and degree precomputation from the cache."""
        X = torch.randn(13, 4)
        nystrom_X = torch.randn(8, 4)
        nystrom_out = torch.randn(8, 3)
        cache = SimpleNamespace(
            _propagation_indices=None,
            _propagation_sampled_x=None,
            _propagation_sigma=None,
            _propagation_D=None,
            _propagation_nystrom_x_sq=None,
        )
        counters = {"fps": 0, "sigma": 0}

        def fake_farthest_point_sampling(values, n_sample, device=None):
            counters["fps"] += 1
            return torch.tensor([0, 2, 4, 6], dtype=torch.long)

        def fake_find_sigma_by_degree(*args, **kwargs):
            counters["sigma"] += 1
            return 1.0

        monkeypatch.setattr(nystrom_utils, "farthest_point_sampling", fake_farthest_point_sampling)
        monkeypatch.setattr(nystrom_utils, "find_sigma_by_degree", fake_find_sigma_by_degree)

        first = nystrom_propagate(
            nystrom_out,
            X,
            nystrom_X,
            device="cpu",
            n_sample2=4,
            n_neighbors=4,
            chunk_size=32,
            cache=cache,
        )
        second = nystrom_propagate(
            nystrom_out,
            X,
            nystrom_X,
            device="cpu",
            n_sample2=4,
            n_neighbors=4,
            chunk_size=32,
            cache=cache,
        )

        assert first.shape == second.shape == (X.shape[0], nystrom_out.shape[1])
        assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)
        assert counters == {"fps": 1, "sigma": 1}
        assert cache._propagation_sigma == pytest.approx(1.0)
        assert torch.equal(cache._propagation_indices, torch.tensor([0, 2, 4, 6], dtype=torch.long))

    def test_weighted_neighbor_sum_matches_reference(self):
        """Test that embedding_bag matches the original gather + einsum formula."""
        weights = torch.tensor([[0.2, 0.3, 0.5], [1.0, 0.1, 0.4]], dtype=torch.float32)
        indices = torch.tensor([[1, 4, 2], [0, 5, 3]], dtype=torch.long)
        nystrom_out = torch.arange(1, 13, dtype=torch.float32).reshape(6, 2)

        expected = _reference_weighted_neighbor_sum(weights, indices, nystrom_out)
        actual = nystrom_utils._weighted_neighbor_sum(weights, indices, nystrom_out, {})

        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

    def test_dist_topk_embedding_bag_matches_reference_on_distinct_distances(self):
        """Test that the fast top-k path matches the original formula when distances are unique."""
        X = torch.tensor(
            [[0.2, 0.1], [1.4, 1.1], [4.3, 2.2]],
            dtype=torch.float32,
        )
        nystrom_X = torch.tensor(
            [[0.0, 0.0], [1.1, 2.3], [2.7, 0.8], [3.2, 4.4], [5.5, 1.7]],
            dtype=torch.float32,
        )
        nystrom_out = torch.tensor(
            [[1.0, 0.0], [0.5, 1.0], [1.5, -0.5], [0.0, 2.0], [1.2, 0.8]],
            dtype=torch.float32,
        )
        sigma = 1.7
        n_neighbors = 3
        D = rbf_affinity(nystrom_X, sigma=sigma).mean(1)

        expected_weights, expected_indices = keep_topk_per_row(
            rbf_affinity(X, nystrom_X, sigma=sigma),
            n_neighbors,
        )
        expected_norm = expected_weights / D[expected_indices].sum(1, keepdim=True)
        expected = _reference_weighted_neighbor_sum(expected_norm, expected_indices, nystrom_out)

        nystrom_x_sq = nystrom_X.pow(2).sum(dim=1).unsqueeze(0)
        actual_weights, actual_indices = nystrom_utils._rbf_topk_from_squared_distance(
            X,
            nystrom_X,
            nystrom_x_sq,
            sigma,
            n_neighbors,
        )
        actual_norm = actual_weights / D[actual_indices].sum(1, keepdim=True)
        actual = nystrom_utils._weighted_neighbor_sum(actual_norm, actual_indices, nystrom_out, {})

        assert torch.equal(actual_indices, expected_indices)
        assert torch.allclose(actual_weights, expected_weights, atol=1e-6, rtol=1e-6)
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

    def test_ncut_fn_with_different_parameters(self, small_feature_matrix):
        """Test ncut_fn with different parameters."""
        # Test with different n_eig
        n_eig = 3
        eigvec, eigval = ncut_fn(small_feature_matrix, n_eig=n_eig)
        assert eigvec.shape == (small_feature_matrix.shape[0], n_eig)
        assert eigval.shape == (n_eig,)
        
        # Test with different d_sigma
        d_sigma = 0.5
        eigvec, eigval = ncut_fn(small_feature_matrix, n_eig=n_eig, quantile_sigma=d_sigma)
        assert eigvec.shape == (small_feature_matrix.shape[0], n_eig)
        assert eigval.shape == (n_eig,)
        
        # Test with explicit sigma
        sigma = 0.1
        eigvec, eigval = ncut_fn(small_feature_matrix, n_eig=n_eig, sigma=sigma)
        assert eigvec.shape == (small_feature_matrix.shape[0], n_eig)
        assert eigval.shape == (n_eig,)
        
        # Test with make_orthogonal=True
        eigvec, eigval = ncut_fn(small_feature_matrix, n_eig=n_eig, make_orthogonal=True)
        assert eigvec.shape == (small_feature_matrix.shape[0], n_eig)
        assert eigval.shape == (n_eig,)

    def test_gram_schmidt_matches_reference_output(self, random_seed):
        """Test that gram_schmidt matches the original implementation."""
        torch.manual_seed(random_seed)
        matrix = torch.randn(512, 16)

        expected = _reference_gram_schmidt(matrix)
        actual = gram_schmidt(matrix)

        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

    def test_gram_schmidt_preserves_dtype(self):
        """Test that gram_schmidt preserves dtype after the QR fast path."""
        matrix = torch.randn(128, 8, dtype=torch.float16)

        actual = gram_schmidt(matrix)

        assert actual.dtype == torch.float16
        assert actual.shape == matrix.shape

    def test_gram_schmidt_reference_speedup(self, random_seed):
        """Test that the QR fast path significantly outperforms the reference loop."""
        torch.manual_seed(random_seed)
        matrix = torch.randn(4096, 64)

        expected = _reference_gram_schmidt(matrix)
        actual = gram_schmidt(matrix)
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

        reference_time = _median_runtime(_reference_gram_schmidt, matrix)
        optimized_time = _median_runtime(gram_schmidt, matrix)

        assert reference_time / optimized_time >= 3.0

    @pytest.mark.slow
    def test_sigma_search_precomputed_distance_candidate_speedup(self):
        """Simulation benchmark for reusing one distance matrix across sigma search."""
        X = torch.randn(1000, 256)

        expected = _find_sigma_by_degree(X, quantile_sigma=0.25, affinity_fn=rbf_affinity)
        actual = _candidate_find_sigma_by_degree_rbf(X, quantile_sigma=0.25)

        assert actual == pytest.approx(expected, abs=1e-6)

        reference_time = _median_call_runtime(
            lambda: _find_sigma_by_degree(X, quantile_sigma=0.25, affinity_fn=rbf_affinity)
        )
        candidate_time = _median_call_runtime(
            lambda: _candidate_find_sigma_by_degree_rbf(X, quantile_sigma=0.25)
        )

        assert reference_time / candidate_time >= 1.5
