import pytest
import torch
import torch.nn.functional as F
from ncut_pytorch import kway_ncut, axis_align, ncut_fn


class TestKwayNcut:
    """Test the kway_ncut module."""

    def test_axis_align(self, small_feature_matrix):
        """Test the axis_align function."""
        # First get some eigenvectors using ncut_fn
        n_eig = 5
        eigvec, _ = ncut_fn(small_feature_matrix, n_eig=n_eig)
        
        # Apply axis_align
        R = axis_align(eigvec, max_iter=100, n_sample=50)
        
        # Check shape
        assert R.shape == (n_eig, n_eig)
        
        # Check that R is orthogonal (R^T R = I)
        I = torch.eye(n_eig, device=R.device)
        assert torch.allclose(R.T @ R, I, atol=1e-5)
        
        # Check that the rotation preserves the norm
        eigvec_normalized = F.normalize(eigvec, dim=1)
        rotated_eigvec = eigvec_normalized @ R
        assert torch.allclose(
            torch.norm(eigvec_normalized, dim=1),
            torch.norm(rotated_eigvec, dim=1),
            atol=1e-5
        )

    def test_kway_ncut(self, small_feature_matrix):
        """Test the kway_ncut function."""
        # First get some eigenvectors using ncut_fn
        n_eig = 5
        eigvec, _ = ncut_fn(small_feature_matrix, n_eig=n_eig)
        
        # Apply kway_ncut
        rotated_eigvec = kway_ncut(eigvec, max_iter=100, n_sample=50)
        
        # Check shape
        assert rotated_eigvec.shape == eigvec.shape
        
        # Check that the output is normalized
        norms = torch.norm(rotated_eigvec, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        
        # Check that we can get cluster assignments
        cluster_assignments = rotated_eigvec.argmax(dim=1)
        assert cluster_assignments.shape == (small_feature_matrix.shape[0],)
        assert cluster_assignments.min() >= 0
        assert cluster_assignments.max() < n_eig

    def test_kway_ncut_with_different_parameters(self, small_feature_matrix):
        """Test kway_ncut with different parameters."""
        # First get some eigenvectors using ncut_fn
        n_eig = 3
        eigvec, _ = ncut_fn(small_feature_matrix, n_eig=n_eig)
        
        # Test with different max_iter
        rotated_eigvec = kway_ncut(eigvec, max_iter=50)
        assert rotated_eigvec.shape == eigvec.shape
        
        # Test with different n_sample
        rotated_eigvec = kway_ncut(eigvec, n_sample=30)
        assert rotated_eigvec.shape == eigvec.shape

    def test_onehot_discretize(self, small_feature_matrix):
        """Test the _onehot_discretize function (indirectly through axis_align)."""
        # First get some eigenvectors using ncut_fn
        n_eig = 5
        eigvec, _ = ncut_fn(small_feature_matrix, n_eig=n_eig)
        
        # Apply axis_align with a small number of iterations to test _onehot_discretize
        R = axis_align(eigvec, max_iter=1, n_sample=50)
        
        # Check shape
        assert R.shape == (n_eig, n_eig)