import pytest
import torch
import numpy as np
from ncut_pytorch import ncut_fn


class TestNcutDeterministic:
    """Test the deterministic behavior of ncut_fn with fixed random seeds."""

    def test_deterministic_with_large_input(self, large_feature_matrix, random_seed):
        """Test that ncut_fn produces deterministic results with a fixed random seed on large input."""
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # First run
        eigvec1, eigval1 = ncut_fn(large_feature_matrix, n_eig=100, n_sample=50)
        
        # Reset random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Second run
        eigvec2, eigval2 = ncut_fn(large_feature_matrix, n_eig=100, n_sample=50)
        
        # Check that results are identical
        assert torch.allclose(eigvec1, eigvec2, atol=1e-6)
        assert torch.allclose(eigval1, eigval2, atol=1e-6)
        
    def test_different_seeds_produce_different_results(self, large_feature_matrix):
        """Test that different random seeds produce different results."""
        # First run with seed 42
        torch.manual_seed(42)
        np.random.seed(42)
        eigvec1, eigval1 = ncut_fn(large_feature_matrix, n_eig=10, n_sample=50)
        
        # Second run with different seed
        torch.manual_seed(43)
        np.random.seed(43)
        eigvec2, eigval2 = ncut_fn(large_feature_matrix, n_eig=10, n_sample=50)
        
        # Check that results are different
        # We use not torch.allclose with a small tolerance to ensure they're meaningfully different
        assert not torch.allclose(eigvec1, eigvec2, atol=1e-3)
        
    def test_deterministic_with_very_large_input(self):
        """Test deterministic behavior with a very large input matrix."""
        # Create a very large feature matrix
        torch.manual_seed(42)  # Set seed for reproducible matrix creation
        very_large_matrix = torch.rand(10000, 50)
        
        # Set random seeds for ncut_fn
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # First run
        eigvec1, eigval1 = ncut_fn(very_large_matrix, n_eig=10, n_sample=50)
        
        # Reset random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Second run
        eigvec2, eigval2 = ncut_fn(very_large_matrix, n_eig=10, n_sample=50)
        
        # Check that results are identical
        assert torch.allclose(eigvec1, eigvec2, atol=1e-6)
        assert torch.allclose(eigval1, eigval2, atol=1e-6)
        
    def test_deterministic_with_different_parameters(self, large_feature_matrix):
        """Test deterministic behavior with different parameters."""
        seed = 42
        
        # Test with different n_eig
        torch.manual_seed(seed)
        np.random.seed(seed)
        eigvec1, eigval1 = ncut_fn(large_feature_matrix, n_eig=5, d_gamma=0.2, n_sample=50)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        eigvec2, eigval2 = ncut_fn(large_feature_matrix, n_eig=5, d_gamma=0.2, n_sample=50)
        
        assert torch.allclose(eigvec1, eigvec2, atol=1e-6)
        assert torch.allclose(eigval1, eigval2, atol=1e-6)
        
        # Test with make_orthogonal=True
        torch.manual_seed(seed)
        np.random.seed(seed)
        eigvec1, eigval1 = ncut_fn(large_feature_matrix, n_eig=5, make_orthogonal=True, n_sample=50)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        eigvec2, eigval2 = ncut_fn(large_feature_matrix, n_eig=5, make_orthogonal=True, n_sample=50)
        
        assert torch.allclose(eigvec1, eigvec2, atol=1e-6)
        assert torch.allclose(eigval1, eigval2, atol=1e-6)