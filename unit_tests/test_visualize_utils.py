import pytest
import torch
import numpy as np
from ncut_pytorch import (
    mspace_color,
    tsne_color,
    umap_color,
    umap_sphere_color,
    convert_to_lab_color,
    rotate_rgb_cube,
)


class TestVisualizeUtils:
    """Test the visualize_utils module."""

    def test_mspace_color(self, small_feature_matrix):
        """Test the mspace_color function."""
        # Run with minimal training steps for faster testing
        rgb = mspace_color(
            small_feature_matrix,
            q=0.95,
            n_eig=10,
            n_dim=3,
            training_steps=5,
            progress_bar=False
        )
        
        # Check shape
        assert rgb.shape == (small_feature_matrix.shape[0], 3)
        
        # Check that RGB values are in [0, 1]
        assert torch.all(rgb >= 0)
        assert torch.all(rgb <= 1)
        
        # Check that the output has reasonable values
        assert not torch.isnan(rgb).any()
        assert not torch.isinf(rgb).any()

    def test_tsne_color(self, small_feature_matrix):
        """Test the tsne_color function."""
        # Use a small number of samples for faster testing
        rgb = tsne_color(
            small_feature_matrix,
            num_sample=50,
            perplexity=10,
            n_dim=3,
            metric="cosine",
            q=0.95,
            knn=5
        )
        
        # Check shape
        assert rgb.shape == (small_feature_matrix.shape[0], 3)
        
        # Check that RGB values are in [0, 1]
        assert torch.all(rgb >= 0)
        assert torch.all(rgb <= 1)
        
        # Check that the output has reasonable values
        assert not torch.isnan(rgb).any()
        assert not torch.isinf(rgb).any()

    @pytest.mark.slow
    def test_umap_color(self, small_feature_matrix):
        """Test the umap_color function."""
        # Use a small number of samples for faster testing
        rgb = umap_color(
            small_feature_matrix,
            num_sample=50,
            n_neighbors=10,
            min_dist=0.1,
            n_dim=3,
            metric="cosine",
            q=0.95,
            knn=5
        )
        
        # Check shape
        assert rgb.shape == (small_feature_matrix.shape[0], 3)
        
        # Check that RGB values are in [0, 1]
        assert torch.all(rgb >= 0)
        assert torch.all(rgb <= 1)
        
        # Check that the output has reasonable values
        assert not torch.isnan(rgb).any()
        assert not torch.isinf(rgb).any()

    @pytest.mark.slow
    def test_umap_sphere_color(self, small_feature_matrix):
        """Test the umap_sphere_color function."""
        # Use a small number of samples for faster testing
        rgb = umap_sphere_color(
            small_feature_matrix,
            num_sample=50,
            n_neighbors=10,
            min_dist=0.1,
            metric="cosine",
            q=0.95,
            knn=5
        )
        
        # Check shape
        assert rgb.shape == (small_feature_matrix.shape[0], 3)
        
        # Check that RGB values are in [0, 1]
        assert torch.all(rgb >= 0)
        assert torch.all(rgb <= 1)
        
        # Check that the output has reasonable values
        assert not torch.isnan(rgb).any()
        assert not torch.isinf(rgb).any()

    def test_convert_to_lab_color(self):
        """Test the convert_to_lab_color function."""
        # Create a sample RGB tensor
        rgb = torch.tensor([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 1.0],  # White
            [0.0, 0.0, 0.0],  # Black
        ])
        
        # Convert to LAB color space
        lab = convert_to_lab_color(rgb)
        lab = torch.from_numpy(np.array(lab))
        
        # Check shape
        assert lab.shape == rgb.shape
        
        # Check that the output has reasonable values
        assert not torch.isnan(lab).any()
        assert not torch.isinf(lab).any()
        
        # Test with full_range=False
        lab_normalized = convert_to_lab_color(rgb, full_range=False)
        lab_normalized = torch.from_numpy(np.array(lab_normalized))

        # Check shape
        assert lab_normalized.shape == rgb.shape
        
        # Check that the output has reasonable values
        assert not torch.isnan(lab_normalized).any()
        assert not torch.isinf(lab_normalized).any()
        
        # Check that values are in [0, 1]
        assert torch.all(lab_normalized >= 0)
        assert torch.all(lab_normalized <= 1)

    def test_rotate_rgb_cube(self):
        """Test the rotate_rgb_cube function."""
        # Create a sample RGB tensor
        rgb = torch.tensor([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 1.0],  # White
            [0.0, 0.0, 0.0],  # Black
        ])
        
        # Rotate the RGB cube
        for position in range(1, 7):
            rotated_rgb = rotate_rgb_cube(rgb, position=position)
            
            # Check shape
            assert rotated_rgb.shape == rgb.shape
            
            # Check that RGB values are in [0, 1]
            assert torch.all(rotated_rgb >= 0)
            assert torch.all(rotated_rgb <= 1)
            
            # Check that the output has reasonable values
            assert not torch.isnan(rotated_rgb).any()
            assert not torch.isinf(rotated_rgb).any()