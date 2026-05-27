import statistics
import time

import torch
from ncut_pytorch import ncut_fn
from ncut_pytorch.ncuts.ncut_nystrom import _plain_ncut, nystrom_propagate
from ncut_pytorch.utils.math import gram_schmidt


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

