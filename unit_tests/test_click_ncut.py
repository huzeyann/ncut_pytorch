from tkinter import N
import pytest
import torch
from ncut_pytorch.ncuts.ncut_click import ncut_click_prompt


class TestBiasedNcut:
    """Test the biased_ncut module."""

    def test_bias_ncut_soft_with_fg_only(self, small_feature_matrix):
        """Test bias_ncut_soft with foreground clicks only."""
        # Create foreground clicks
        fg_idx = torch.tensor([0, 10, 20])
        
        # Run bias_ncut_soft
        eigvecs, eigvals = ncut_click_prompt(
            small_feature_matrix,
            fg_idx,
            n_eig=5,
        )
        
        # Check shapes
        assert eigvecs.shape == (small_feature_matrix.shape[0], 5)
        assert eigvals.shape == (5,)
        
        # Check that eigenvalues are sorted in descending order
        assert torch.all(eigvals[:-1] >= eigvals[1:])
        
        # Check that eigenvectors have reasonable values
        assert not torch.isnan(eigvecs).any()
        assert not torch.isinf(eigvecs).any()

    def test_bias_ncut_soft_with_fg_and_bg(self, small_feature_matrix):
        """Test bias_ncut_soft with both foreground and background clicks."""
        # Create foreground and background clicks
        fg_idx = torch.tensor([0, 10, 20])
        bg_idx = torch.tensor([50, 60, 70])
        
        # Run bias_ncut_soft
        eigvecs, eigvals = ncut_click_prompt(
            small_feature_matrix,
            fg_idx,
            bg_idx,
            n_eig=5,
        )
        
        # Check shapes
        assert eigvecs.shape == (small_feature_matrix.shape[0], 5)
        assert eigvals.shape == (5,)
        
        # Check that eigenvalues are sorted in descending order
        assert torch.all(eigvals[:-1] >= eigvals[1:])
        
        # Check that eigenvectors have reasonable values
        assert not torch.isnan(eigvecs).any()
        assert not torch.isinf(eigvecs).any()

    def test_bias_ncut_soft_with_different_parameters(self, small_feature_matrix):
        """Test bias_ncut_soft with different parameters."""
        # Create foreground clicks
        fg_idx = torch.tensor([0, 10, 20])
        
        # Test with different num_eig
        eigvecs, eigvals = ncut_click_prompt(
            small_feature_matrix,
            fg_idx,
            n_eig=3,
            click_weight=0.5,
            d_gamma=0.5
        )
        assert eigvecs.shape == (small_feature_matrix.shape[0], 3)
        assert eigvals.shape == (3,)
        
        # Test with different click_weight
        eigvecs, eigvals = ncut_click_prompt(
            small_feature_matrix,
            fg_idx,
            n_eig=5,
            click_weight=0.8,
            d_gamma=0.5
        )
        assert eigvecs.shape == (small_feature_matrix.shape[0], 5)
        
        # Test with different degree
        eigvecs, eigvals = ncut_click_prompt(
            small_feature_matrix,
            fg_idx,
            n_eig=5,
            click_weight=0.5,
            d_gamma=0.2
        )
        assert eigvecs.shape == (small_feature_matrix.shape[0], 5)