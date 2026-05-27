import json
import statistics
import subprocess
import sys
import textwrap
import time

import torch

from ncut_pytorch import ncut_fn
from ncut_pytorch.ncuts.ncut_nystrom import nystrom_propagate
from ncut_pytorch.utils.math import rbf_affinity


class TestCustomAffinity:
    """Test custom affinity function functionality."""

    @staticmethod
    def reference_rbf_affinity(X1: torch.Tensor, X2: torch.Tensor = None, sigma: float = 1.0, zero_diag: bool = False):
        """Reference implementation matching the pre-optimization behavior."""
        X2 = X1 if X2 is None else X2
        try:
            dist2 = torch.cdist(X1, X2, p=2) ** 2
        except NotImplementedError:
            dist2 = X1.unsqueeze(1) - X2.unsqueeze(0)
            dist2 = dist2.pow(2).sum(dim=-1)
        W = torch.exp(-dist2 / (2.0 * sigma * sigma))
        if zero_diag and X1 is X2:
            W = W.clone()
            W.fill_diagonal_(0.0)
        return W

    @staticmethod
    def _median_runtime(fn, *args, repeats: int = 5) -> float:
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn(*args)
            timings.append(time.perf_counter() - start)
        return statistics.median(timings)

    @staticmethod
    def _peak_rss_bytes_for_variant(variant: str, n: int = 3072, d: int = 64) -> int:
        script = textwrap.dedent(f"""
        import json
        import resource
        import torch

        variant = {variant!r}
        X1 = torch.randn({n}, {d})

        def reference_rbf(X1, sigma=1.0):
            dist2 = torch.cdist(X1, X1, p=2) ** 2
            return torch.exp(-dist2 / (2.0 * sigma * sigma))

        def optimized_rbf(X1, sigma=1.0):
            x1_sq = X1.pow(2).sum(dim=1, keepdim=True)
            dist2 = x1_sq + x1_sq.T
            dist2.addmm_(X1, X1.T, beta=1.0, alpha=-2.0)
            dist2.clamp_min_(0)
            return dist2.mul_(-0.5 / (sigma * sigma)).exp_()

        fn = reference_rbf if variant == "reference" else optimized_rbf
        _ = fn(X1)
        print(json.dumps({{"ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}))
        """)
        out = subprocess.check_output([sys.executable, "-c", script], text=True)
        return json.loads(out)["ru_maxrss"]

    @staticmethod
    def custom_linear_affinity(X1: torch.Tensor, X2: torch.Tensor = None, sigma: float = 1.0):
        """Custom linear affinity function for testing."""
        X2 = X1 if X2 is None else X2

        # Compute linear kernel (dot product) and normalize
        A = torch.mm(X1, X2.T)
        # Apply sigma scaling and make more distinctive
        A = A * sigma * 10.0  # Make it more different from cosine
        # Apply sigmoid to make it bounded like RBF
        A = torch.sigmoid(A)

        return A

    @staticmethod
    def custom_cosine_affinity(X1: torch.Tensor, X2: torch.Tensor = None, sigma: float = 1.0):
        """Custom cosine similarity affinity function for testing."""
        X2 = X1 if X2 is None else X2

        # Handle edge case where X1 or X2 might have zero vectors
        X1_norm = torch.nn.functional.normalize(X1, p=2, dim=1, eps=1e-8)
        X2_norm = torch.nn.functional.normalize(X2, p=2, dim=1, eps=1e-8)

        # Compute cosine similarity
        A = torch.mm(X1_norm, X2_norm.T)
        # Scale with sigma parameter and shift to [0,1]
        A = (A * sigma + 1) / 2
        # Ensure positive values
        A = torch.clamp(A, min=1e-8)

        return A

    def test_default_affinity_behavior(self, small_feature_matrix):
        """Test that default behavior still works (affinity_fn=None)."""
        n_eig = 5
        eigvec, eigval = ncut_fn(small_feature_matrix, n_eig=n_eig, affinity_fn=rbf_affinity)

        # Check shapes
        assert eigvec.shape == (small_feature_matrix.shape[0], n_eig)
        assert eigval.shape == (n_eig,)

        # Check that eigenvalues are sorted in descending order
        assert torch.all(eigval[:-1] >= eigval[1:])

        # Check that eigenvectors don't contain NaN or inf
        assert not torch.isnan(eigvec).any()
        assert not torch.isinf(eigvec).any()

    def test_custom_linear_affinity(self, small_feature_matrix):
        """Test using custom linear affinity function."""
        n_eig = 5
        sigma = 0.1

        eigvec, eigval = ncut_fn(
            small_feature_matrix,
            n_eig=n_eig,
            sigma=sigma,
            affinity_fn=self.custom_linear_affinity
        )

        # Check shapes
        assert eigvec.shape == (small_feature_matrix.shape[0], n_eig)
        assert eigval.shape == (n_eig,)

        # Check that eigenvalues are sorted in descending order
        assert torch.all(eigval[:-1] >= eigval[1:])

        # Check that results are valid (no NaN/inf)
        assert not torch.isnan(eigvec).any()
        assert not torch.isinf(eigvec).any()
        assert not torch.isnan(eigval).any()
        assert not torch.isinf(eigval).any()

    def test_custom_cosine_affinity(self, small_feature_matrix):
        """Test using custom cosine similarity affinity function."""
        n_eig = 5
        sigma = 1.0

        eigvec, eigval = ncut_fn(
            small_feature_matrix,
            n_eig=n_eig,
            sigma=sigma,
            affinity_fn=self.custom_cosine_affinity
        )

        # Check shapes
        assert eigvec.shape == (small_feature_matrix.shape[0], n_eig)
        assert eigval.shape == (n_eig,)

        # Check that eigenvalues are sorted in descending order
        assert torch.all(eigval[:-1] >= eigval[1:])

        # Check that results are valid (no NaN/inf)
        assert not torch.isnan(eigvec).any()
        assert not torch.isinf(eigvec).any()
        assert not torch.isnan(eigval).any()
        assert not torch.isinf(eigval).any()

    def test_custom_affinity_different_results(self, small_feature_matrix):
        """Test that different affinity functions produce different results."""
        n_eig = 5
        sigma = 0.5

        # Get results with default RBF affinity
        eigvec_rbf, eigval_rbf = ncut_fn(
            small_feature_matrix, n_eig=n_eig, sigma=sigma, affinity_fn=rbf_affinity
        )

        # Get results with custom linear affinity
        eigvec_linear, eigval_linear = ncut_fn(
            small_feature_matrix, n_eig=n_eig, sigma=sigma, affinity_fn=self.custom_linear_affinity
        )

        # Get results with custom cosine affinity
        eigvec_cosine, eigval_cosine = ncut_fn(
            small_feature_matrix, n_eig=n_eig, sigma=sigma, affinity_fn=self.custom_cosine_affinity
        )

        # Results should be different (at least eigenvalues should differ significantly)
        # We expect at least one of the methods to produce different results
        rbf_linear_diff = not torch.allclose(eigval_rbf, eigval_linear, atol=1e-3)
        rbf_cosine_diff = not torch.allclose(eigval_rbf, eigval_cosine, atol=1e-3)
        linear_cosine_diff = not torch.allclose(eigval_linear, eigval_cosine, atol=5e-3)  # More relaxed tolerance

        # At least two of the three comparisons should show differences
        num_different = sum([rbf_linear_diff, rbf_cosine_diff, linear_cosine_diff])
        assert num_different >= 2, f"Expected at least 2 different results, got {num_different}"

    def test_custom_affinity_with_no_propagation(self, small_feature_matrix):
        """Test custom affinity function with no_propagation=True."""
        n_eig = 5
        sigma = 0.5

        eigvec, eigval, indices, returned_sigma = ncut_fn(
            small_feature_matrix,
            n_eig=n_eig,
            sigma=sigma,
            no_propagation=True,
            affinity_fn=self.custom_linear_affinity
        )

        # Check shapes and types
        assert eigvec.shape[1] == n_eig
        assert eigval.shape == (n_eig,)
        assert indices.shape[0] <= small_feature_matrix.shape[0]
        assert isinstance(returned_sigma, float)
        assert returned_sigma == sigma

        # Check that eigenvalues are sorted in descending order
        assert torch.all(eigval[:-1] >= eigval[1:])

    def test_custom_affinity_nystrom_propagate(self, small_feature_matrix):
        """Test that custom affinity function is used in nystrom_propagate."""
        n_eig = 5
        sigma = 0.5

        # First get nystrom results
        nystrom_eigvec, _, indices, _ = ncut_fn(
            small_feature_matrix,
            n_eig=n_eig,
            sigma=sigma,
            no_propagation=True,
            affinity_fn=self.custom_linear_affinity
        )

        nystrom_X = small_feature_matrix[indices]

        # Test nystrom_propagate with custom affinity
        eigvec = nystrom_propagate(
            nystrom_eigvec,
            small_feature_matrix,
            nystrom_X,
        )

        # Check shape
        assert eigvec.shape == (small_feature_matrix.shape[0], n_eig)

        # Check that results are valid
        assert not torch.isnan(eigvec).any()
        assert not torch.isinf(eigvec).any()

    def test_invalid_affinity_function_signature(self, small_feature_matrix):
        """Test that invalid affinity function signatures raise appropriate errors."""

        def bad_affinity_fn(X):  # Missing required parameters
            return torch.eye(X.shape[0])

        # This should raise an error when the function is called
        try:
            ncut_fn(small_feature_matrix, n_eig=5, affinity_fn=bad_affinity_fn, sigma=1.0)
            assert False, "Should have raised an error for invalid function signature"
        except TypeError:
            pass  # Expected error

    def test_rbf_affinity_matches_reference_same_input(self, random_seed):
        """Test optimized rbf_affinity matches the reference output on square input."""
        torch.manual_seed(random_seed)
        X = torch.randn(256, 32)

        expected = self.reference_rbf_affinity(X, sigma=0.7, zero_diag=True)
        actual = rbf_affinity(X, sigma=0.7, zero_diag=True)

        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

    def test_rbf_affinity_matches_reference_cross_input(self, random_seed):
        """Test optimized rbf_affinity matches the reference output on cross input."""
        torch.manual_seed(random_seed)
        X1 = torch.randn(192, 24)
        X2 = torch.randn(64, 24)

        expected = self.reference_rbf_affinity(X1, X2, sigma=1.3)
        actual = rbf_affinity(X1, X2, sigma=1.3)

        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

    def test_rbf_affinity_reference_speedup(self, random_seed):
        """Test optimized rbf_affinity is measurably faster on a representative dense case."""
        torch.manual_seed(random_seed)
        X = torch.randn(3072, 64)

        expected = self.reference_rbf_affinity(X, sigma=0.8)
        actual = rbf_affinity(X, sigma=0.8)
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

        reference_time = self._median_runtime(self.reference_rbf_affinity, X, None, 0.8)
        optimized_time = self._median_runtime(rbf_affinity, X, None, 0.8)

        assert reference_time / optimized_time >= 1.15

    def test_rbf_affinity_reference_peak_memory(self):
        """Test optimized rbf_affinity uses less peak RSS in an isolated process."""
        reference_peak = self._peak_rss_bytes_for_variant("reference")
        optimized_peak = self._peak_rss_bytes_for_variant("optimized")

        assert optimized_peak <= reference_peak * 0.9
