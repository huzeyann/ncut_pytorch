import pytest
import torch
from ncut_pytorch import Ncut


class TestNcut:
    """Test the Ncut class."""

    def test_init(self, ncut_params):
        """Test that the Ncut class initializes correctly."""
        ncut = Ncut(**ncut_params)
        assert ncut.n_eig == ncut_params['n_eig']
        assert ncut.d_gamma == ncut_params['d_gamma']
        assert ncut.device == ncut_params['device']
        assert ncut.track_grad == ncut_params['track_grad']
        assert ncut._nystrom_x is None
        assert ncut._nystrom_eigvec is None
        assert ncut._eigval is None

    def test_fit(self, small_feature_matrix, ncut_params):
        """Test the fit method."""
        ncut = Ncut(**ncut_params)
        result = ncut.fit(small_feature_matrix)
        
        # Check that fit returns self
        assert result is ncut
        
        # Check that the internal state is updated
        assert ncut._nystrom_x is not None
        assert ncut._nystrom_eigvec is not None
        assert ncut._eigval is not None
        assert ncut.gamma is not None
        
        # Check shapes
        assert ncut._eigval.shape == (ncut_params['n_eig'],)

    def test_transform(self, small_feature_matrix, ncut_params):
        """Test the transform method."""
        ncut = Ncut(**ncut_params)
        ncut.fit(small_feature_matrix)
        
        # Transform the same data
        eigvec = ncut.transform(small_feature_matrix)
        
        # Check shape
        assert eigvec.shape == (small_feature_matrix.shape[0], ncut_params['n_eig'])
        
        # Transform a subset of the data
        subset = small_feature_matrix[:10]
        eigvec_subset = ncut.transform(subset)
        
        # Check shape
        assert eigvec_subset.shape == (subset.shape[0], ncut_params['n_eig'])

    def test_fit_transform(self, small_feature_matrix, ncut_params):
        """Test the fit_transform method."""
        ncut = Ncut(**ncut_params)
        eigvec = ncut.fit_transform(small_feature_matrix)
        
        # Check shape
        assert eigvec.shape == (small_feature_matrix.shape[0], ncut_params['n_eig'])
        
        # Check that the internal state is updated
        assert ncut._nystrom_x is not None
        assert ncut._nystrom_eigvec is not None
        assert ncut._eigval is not None
        assert ncut.gamma is not None

    def test_call(self, small_feature_matrix, ncut_params):
        """Test the __call__ method."""
        ncut = Ncut(**ncut_params)
        eigvec = ncut(small_feature_matrix)
        
        # Check shape
        assert eigvec.shape == (small_feature_matrix.shape[0], ncut_params['n_eig'])
        
        # Check that the internal state is updated
        assert ncut._nystrom_x is not None
        assert ncut._nystrom_eigvec is not None
        assert ncut._eigval is not None
        assert ncut.gamma is not None

    def test_function_like_behavior(self, small_feature_matrix, ncut_params):
        """Test the function-like behavior of Ncut."""
        # When called with data as the first argument, Ncut should behave like a function
        eigvec = Ncut(small_feature_matrix, **ncut_params)
        
        # Check shape
        assert eigvec.shape == (small_feature_matrix.shape[0], ncut_params['n_eig'])

    def test_eigval_property(self, small_feature_matrix, ncut_params):
        """Test the eigval property."""
        ncut = Ncut(**ncut_params)
        ncut.fit(small_feature_matrix)
        
        # Check that eigval returns the eigenvalues
        assert ncut.eigval is ncut._eigval
        assert ncut.eigval.shape == (ncut_params['n_eig'],)