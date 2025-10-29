import torch
from ncut_pytorch import ncut_fn
from ncut_pytorch.ncuts.ncut_nystrom import _plain_ncut, nystrom_propagate


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
        eigvec, eigval, indices, gamma = ncut_fn(
            small_feature_matrix, 
            n_eig=n_eig, 
            no_propagation=True
        )
        
        # Check shapes
        assert eigvec.shape[1] == n_eig
        assert eigval.shape == (n_eig,)
        assert indices.shape[0] <= small_feature_matrix.shape[0]  # Number of sampled points
        assert isinstance(gamma, float)
        
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
        
        # Test with different d_gamma
        d_gamma = 0.5
        eigvec, eigval = ncut_fn(small_feature_matrix, n_eig=n_eig, d_gamma=d_gamma)
        assert eigvec.shape == (small_feature_matrix.shape[0], n_eig)
        assert eigval.shape == (n_eig,)
        
        # Test with explicit gamma
        gamma = 0.1
        eigvec, eigval = ncut_fn(small_feature_matrix, n_eig=n_eig, gamma=gamma)
        assert eigvec.shape == (small_feature_matrix.shape[0], n_eig)
        assert eigval.shape == (n_eig,)
        
        # Test with make_orthogonal=True
        eigvec, eigval = ncut_fn(small_feature_matrix, n_eig=n_eig, make_orthogonal=True)
        assert eigvec.shape == (small_feature_matrix.shape[0], n_eig)
        assert eigval.shape == (n_eig,)


