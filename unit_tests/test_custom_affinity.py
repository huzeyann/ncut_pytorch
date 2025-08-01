import torch

from ncut_pytorch import ncut_fn
from ncut_pytorch.ncuts import nystrom_propagate
from ncut_pytorch.utils.math import rbf_affinity


class TestCustomAffinity:
    """Test custom affinity function functionality."""

    @staticmethod
    def custom_linear_affinity(X1: torch.Tensor, X2: torch.Tensor = None, gamma: float = 1.0):
        """Custom linear affinity function for testing."""
        X2 = X1 if X2 is None else X2

        # Compute linear kernel (dot product) and normalize
        A = torch.mm(X1, X2.T)
        # Apply gamma scaling and make more distinctive
        A = A * gamma * 10.0  # Make it more different from cosine
        # Apply sigmoid to make it bounded like RBF
        A = torch.sigmoid(A)

        return A

    @staticmethod
    def custom_cosine_affinity(X1: torch.Tensor, X2: torch.Tensor = None, gamma: float = 1.0):
        """Custom cosine similarity affinity function for testing."""
        X2 = X1 if X2 is None else X2

        # Handle edge case where X1 or X2 might have zero vectors
        X1_norm = torch.nn.functional.normalize(X1, p=2, dim=1, eps=1e-8)
        X2_norm = torch.nn.functional.normalize(X2, p=2, dim=1, eps=1e-8)

        # Compute cosine similarity
        A = torch.mm(X1_norm, X2_norm.T)
        # Scale with gamma parameter and shift to [0,1]
        A = (A * gamma + 1) / 2
        # Ensure positive values
        A = torch.clamp(A, min=1e-8)

        return A

    def test_default_affinity_behavior(self, small_feature_matrix):
        """Test that default behavior still works (affinity_fn=None)."""
        n_eig = 5
        eigvec, eigval = ncut_fn(small_feature_matrix, n_eig=n_eig, affinity_fn=rbf_affinity, d_gamma='auto')

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
        gamma = 0.1

        eigvec, eigval = ncut_fn(
            small_feature_matrix,
            n_eig=n_eig,
            gamma=gamma,
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
        gamma = 1.0

        eigvec, eigval = ncut_fn(
            small_feature_matrix,
            n_eig=n_eig,
            gamma=gamma,
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
        gamma = 0.5

        # Get results with default RBF affinity
        eigvec_rbf, eigval_rbf = ncut_fn(
            small_feature_matrix, n_eig=n_eig, gamma=gamma, affinity_fn=rbf_affinity
        )

        # Get results with custom linear affinity
        eigvec_linear, eigval_linear = ncut_fn(
            small_feature_matrix, n_eig=n_eig, gamma=gamma, affinity_fn=self.custom_linear_affinity
        )

        # Get results with custom cosine affinity
        eigvec_cosine, eigval_cosine = ncut_fn(
            small_feature_matrix, n_eig=n_eig, gamma=gamma, affinity_fn=self.custom_cosine_affinity
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
        gamma = 0.5

        eigvec, eigval, indices, returned_gamma = ncut_fn(
            small_feature_matrix,
            n_eig=n_eig,
            gamma=gamma,
            no_propagation=True,
            affinity_fn=self.custom_linear_affinity
        )

        # Check shapes and types
        assert eigvec.shape[1] == n_eig
        assert eigval.shape == (n_eig,)
        assert indices.shape[0] <= small_feature_matrix.shape[0]
        assert isinstance(returned_gamma, float)
        assert returned_gamma == gamma

        # Check that eigenvalues are sorted in descending order
        assert torch.all(eigval[:-1] >= eigval[1:])

    def test_custom_affinity_nystrom_propagate(self, small_feature_matrix):
        """Test that custom affinity function is used in nystrom_propagate."""
        n_eig = 5
        gamma = 0.5

        # First get nystrom results
        nystrom_eigvec, _, indices, _ = ncut_fn(
            small_feature_matrix,
            n_eig=n_eig,
            gamma=gamma,
            no_propagation=True,
            affinity_fn=self.custom_linear_affinity
        )

        nystrom_X = small_feature_matrix[indices]

        # Test nystrom_propagate with custom affinity
        eigvec = nystrom_propagate(
            nystrom_eigvec,
            small_feature_matrix,
            nystrom_X,
            gamma=gamma,
            affinity_fn=self.custom_linear_affinity
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
            ncut_fn(small_feature_matrix, n_eig=5, affinity_fn=bad_affinity_fn)
            assert False, "Should have raised an error for invalid function signature"
        except TypeError:
            pass  # Expected error
