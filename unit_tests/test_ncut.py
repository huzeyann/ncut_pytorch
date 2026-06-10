import pytest
import torch
import torch.nn.functional as F

import ncut_pytorch.ncut as ncut_module
from ncut_pytorch import Ncut


class TestNcut:
    """Test the Ncut class."""

    def test_init(self, ncut_params):
        """Test that the Ncut class initializes correctly."""
        ncut = Ncut(**ncut_params)
        assert ncut.n_eig == ncut_params['n_eig']
        assert ncut.quantile_sigma == ncut_params['quantile_sigma']
        assert ncut.device == ncut_params['device']
        assert ncut.exact_gradient == ncut_params['exact_gradient']
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
        assert ncut.sigma is not None
        
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
        assert torch.allclose(ncut._nystrom_x, small_feature_matrix, atol=1e-6)
        
        # Check shape
        assert eigvec.shape == (small_feature_matrix.shape[0], ncut_params['n_eig'])
        
        # Check that the internal state is updated
        assert ncut._nystrom_x is not None
        assert ncut._nystrom_eigvec is not None
        assert ncut._eigval is not None
        assert ncut.sigma is not None

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
        assert ncut.sigma is not None

    def test_call_fp16(self, small_feature_matrix):
        """Test the __call__ method with fp16."""
        ncut = Ncut(n_eig=5)
        eigvec = ncut(small_feature_matrix.half())
        assert eigvec.shape == (small_feature_matrix.shape[0], 5)
        assert eigvec.dtype == torch.float16

    def test_cuda_fp16(self, small_feature_matrix):
        """Test the __call__ method with fp16 on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        ncut = Ncut(n_eig=5)
        eigvec = ncut(small_feature_matrix.half().cuda())
        assert eigvec.shape == (small_feature_matrix.shape[0], 5)
        assert eigvec.dtype == torch.float16


    def test_eigval_property(self, small_feature_matrix, ncut_params):
        """Test the eigval property."""
        ncut = Ncut(**ncut_params)
        ncut.fit(small_feature_matrix)
        
        # Check that eigval returns the eigenvalues
        assert ncut.eigval is ncut._eigval
        assert ncut.eigval.shape == (ncut_params['n_eig'],)

    def test_kway_fit_requires_fit_first(self):
        """Test that kway_fit requires a fitted Ncut model."""
        ncut = Ncut(n_eig=5)

        with pytest.raises(ValueError, match="Call fit\\(\\) first"):
            ncut.kway_fit(n_clusters=3, n_eig=3)

    def test_kway_fit_and_transform(self, small_feature_matrix, ncut_params):
        """Test the k-way fit/transform workflow with fewer eigenvectors than the base model."""
        ncut = Ncut(**ncut_params).fit(small_feature_matrix)

        result = ncut.kway_fit(n_clusters=3, n_eig=4, kmeans_iter=2)
        kway_eigvec = ncut.kway_transform(small_feature_matrix[:8], n_clusters=3, n_eig=4)

        assert result is ncut
        assert (3, 4) in ncut._kway_R
        assert kway_eigvec.shape == (8, 3)

    def test_kway_transform_requires_cached_rotation(self, small_feature_matrix, ncut_params):
        """Test that kway_transform requires a matching cached rotation."""
        ncut = Ncut(**ncut_params).fit(small_feature_matrix)

        with pytest.raises(ValueError, match="Call kway_fit\\(\\)"):
            ncut.kway_transform(small_feature_matrix, n_clusters=3, n_eig=3)

    def test_kway_fit_rejects_too_many_eigenvectors(self, small_feature_matrix, ncut_params):
        """Test that kway_fit validates the requested eigenvector count."""
        ncut = Ncut(**ncut_params).fit(small_feature_matrix)

        with pytest.raises(ValueError, match="exceeds fitted eigenvector count"):
            ncut.kway_fit(n_clusters=3, n_eig=ncut_params['n_eig'] + 1)

    def test_kway_fit_forwards_custom_fit_eigvec(self, small_feature_matrix, ncut_params, monkeypatch):
        """Test that kway_fit uses the provided eigenvectors and forwards its config."""
        ncut = Ncut(**ncut_params).fit(small_feature_matrix)
        fit_eigvec = torch.randn_like(ncut._nystrom_eigvec)
        recorded = {}
        expected_R = torch.randn(4, 3)

        def fake_quick_kway(eigvec, **kwargs):
            recorded["eigvec"] = eigvec.clone()
            recorded.update(kwargs)
            return expected_R

        monkeypatch.setattr(ncut_module, "quick_kway", fake_quick_kway)

        result = ncut.kway_fit(n_clusters=3, n_eig=4, fit_eigvec=fit_eigvec)

        assert result is ncut
        assert torch.equal(recorded["eigvec"], fit_eigvec[:, :4])
        assert recorded["n_clusters"] == 3
        assert recorded["n_eig"] == 4
        assert recorded["n_sample"] == ncut._nystrom_eigvec.shape[0]
        assert recorded["device"] == ncut.device
        assert recorded["kmeans_iter"] == 300
        assert recorded["ret_R"] is True
        assert torch.equal(ncut._kway_R[(3, 4)], expected_R)

    def test_kway_transform_normalize_matches_manual_projection(self, monkeypatch):
        """Test that kway_transform optionally normalizes eigenvectors before projection."""
        ncut = Ncut(n_eig=4, device="cpu")
        ncut._nystrom_x = torch.zeros((1, 2))
        ncut._nystrom_eigvec = torch.zeros((1, 4))

        base_eigvec = torch.tensor(
            [
                [3.0, 0.0, 4.0, 2.0],
                [0.0, 5.0, 12.0, 1.0],
            ]
        )
        R = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, -1.0],
            ]
        )
        ncut._kway_R[(2, 3)] = R

        monkeypatch.setattr(ncut, "transform", lambda X: base_eigvec.clone())

        raw = ncut.kway_transform(torch.randn(2, 5), n_clusters=2, n_eig=3, normalize=False)
        normalized = ncut.kway_transform(torch.randn(2, 5), n_clusters=2, n_eig=3, normalize=True)

        expected_raw = base_eigvec[:, :3] @ R
        expected_normalized = F.normalize(base_eigvec[:, :3], dim=-1) @ R

        assert torch.allclose(raw, expected_raw, atol=1e-6, rtol=1e-6)
        assert torch.allclose(normalized, expected_normalized, atol=1e-6, rtol=1e-6)
        assert not torch.allclose(raw, normalized)

if __name__ == "__main__":
    pytest.main([__file__])
