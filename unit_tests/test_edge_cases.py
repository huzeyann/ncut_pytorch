import pytest
import torch
import numpy as np
from ncut_pytorch import Ncut, ncut_fn, kway_ncut
from ncut_pytorch.ncuts.ncut_click import ncut_click_prompt


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_gamma(self, small_feature_matrix):
        """Test with invalid gamma values."""
        # Check that Ncut raises an error with negative gamma
        with pytest.raises(Exception):
            ncut = Ncut(n_eig=5, d_gamma=-0.1)
            ncut.fit(small_feature_matrix)
        
        # Check that ncut_fn raises an error with negative gamma
        with pytest.raises(Exception):
            eigvec, eigval = ncut_fn(small_feature_matrix, n_eig=5, d_gamma=-0.1)
        
        # Check that ncut_fn raises an error with zero gamma
        with pytest.raises(Exception):
            eigvec, eigval = ncut_fn(small_feature_matrix, n_eig=5, d_gamma=0.0)

    def test_invalid_device(self, small_feature_matrix):
        """Test with invalid device."""
        # Check that Ncut raises an error with invalid device
        with pytest.raises(Exception):
            ncut = Ncut(n_eig=5, device="invalid_device")
            ncut.fit(small_feature_matrix)
        
        # Check that ncut_fn raises an error with invalid device
        with pytest.raises(Exception):
            eigvec, eigval = ncut_fn(small_feature_matrix, n_eig=5, device="invalid_device")

    def test_transform_without_fit(self, small_feature_matrix):
        """Test transform without fit."""
        # Check that transform raises an error if called before fit
        with pytest.raises(Exception):
            ncut = Ncut(n_eig=5)
            ncut.transform(small_feature_matrix)

    def test_bias_ncut_invalid_clicks(self, small_feature_matrix):
        """Test bias_ncut_soft with invalid clicks."""
        # Check that bias_ncut_soft raises an error with out-of-bounds foreground clicks
        with pytest.raises(Exception):
            fg_idx = torch.tensor([small_feature_matrix.shape[0] + 1])
            eigvecs, eigvals = ncut_click_prompt(small_feature_matrix, fg_idx)
        
        # Check that bias_ncut_soft raises an error with out-of-bounds background clicks
        with pytest.raises(Exception):
            fg_idx = torch.tensor([0])
            bg_idx = torch.tensor([small_feature_matrix.shape[0] + 1])
            eigvecs, eigvals = ncut_click_prompt(small_feature_matrix, fg_idx, bg_idx)

    def test_kway_ncut_invalid_eigvec(self):
        """Test kway_ncut with invalid eigenvectors."""
        # Check that kway_ncut raises an error with empty eigenvectors
        with pytest.raises(Exception):
            eigvec = torch.zeros((0, 5))
            rotated_eigvec = kway_ncut(eigvec)
        
        # Check that kway_ncut raises an error with eigenvectors containing NaN
        with pytest.raises(Exception):
            eigvec = torch.ones((10, 5))
            eigvec[0, 0] = float('nan')
            rotated_eigvec = kway_ncut(eigvec)
        
        # Check that kway_ncut raises an error with eigenvectors containing Inf
        with pytest.raises(Exception):
            eigvec = torch.ones((10, 5))
            eigvec[0, 0] = float('inf')
            rotated_eigvec = kway_ncut(eigvec)

    def test_edge_case_parameters(self, small_feature_matrix):
        """Test with edge case parameters."""
        # Test with very small d_gamma
        ncut = Ncut(n_eig=5, d_gamma=1e-10)
        ncut.fit(small_feature_matrix)
        
        # Test with very large d_gamma
        ncut = Ncut(n_eig=5, d_gamma=1e10)
        ncut.fit(small_feature_matrix)
        
        # Test with n_eig=1
        ncut = Ncut(n_eig=1)
        ncut.fit(small_feature_matrix)
        
        # Test with n_eig equal to the number of samples
        n_eig = min(small_feature_matrix.shape[0] - 1, 10)  # Use a smaller value for testing
        ncut = Ncut(n_eig=n_eig)
        ncut.fit(small_feature_matrix)