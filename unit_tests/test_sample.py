import numpy as np
import pytest
import torch

from ncut_pytorch.utils import sample as sample_utils


def test_farthest_point_sampling_remaps_presampled_indices(monkeypatch):
    X = torch.arange(30, dtype=torch.float32).reshape(10, 3)

    monkeypatch.setattr(
        sample_utils,
        "_stratified_presample_indices",
        lambda num_data, num_draw: torch.tensor([7, 1, 9, 4], dtype=torch.long),
    )
    monkeypatch.setattr(
        sample_utils,
        "_farthest_point_sampling",
        lambda X_subset, n_sample, max_dim=8, device=None: torch.tensor([2, 0], dtype=torch.long),
    )

    sampled = sample_utils.farthest_point_sampling(X, n_sample=2, max_draw_ratio=2.0)

    assert torch.equal(sampled, torch.tensor([9, 7], dtype=torch.long))


def test_sample_idx_with_fpsample_uses_supported_kdtree_api(monkeypatch):
    calls = {}

    class FakeFPSample:
        def fps_npdu_kdtree_sampling(self, X_np, n_sample):
            calls["shape"] = X_np.shape
            calls["n_sample"] = n_sample
            return np.array([3, 1], dtype=np.int32)

    monkeypatch.setattr(sample_utils, "_HAS_FPSAMPLE", True)
    monkeypatch.setattr(sample_utils, "_HAS_FPSAMPLE_BUCKET_FPS", False)
    monkeypatch.setattr(sample_utils, "_HAS_FPSAMPLE_KDTREE_FPS", True)
    monkeypatch.setattr(sample_utils, "_fpsample", FakeFPSample())

    sampled = sample_utils._sample_idx_with_fpsample(
        torch.randn(6, 10),
        2,
        max_dim=4,
        device="cpu",
    )

    assert torch.equal(sampled, torch.tensor([3, 1], dtype=torch.long))
    assert calls == {"shape": (6, 4), "n_sample": 2}


def test_internal_fps_uses_fpsample_fallback_without_torch_quickfps(monkeypatch):
    calls = {}

    def fake_fpsample_backend(X, n_sample, *, max_dim, device):
        calls["shape"] = tuple(X.shape)
        calls["n_sample"] = n_sample
        calls["max_dim"] = max_dim
        calls["device"] = device
        return torch.tensor([5, 1, 3], dtype=torch.long)

    monkeypatch.setattr(sample_utils, "_HAS_TORCH_QUICKFPS", False)
    monkeypatch.setattr(sample_utils, "_HAS_FPSAMPLE", True)
    monkeypatch.setattr(sample_utils, "_sample_idx_with_fpsample", fake_fpsample_backend)

    sampled = sample_utils._farthest_point_sampling(
        torch.randn(8, 6),
        3,
        max_dim=5,
        device="cpu",
    )

    assert torch.equal(sampled, torch.tensor([5, 1, 3], dtype=torch.long))
    assert calls == {
        "shape": (8, 6),
        "n_sample": 3,
        "max_dim": 5,
        "device": "cpu",
    }


def test_internal_fps_requires_fpsample_when_no_backend_available(monkeypatch):
    monkeypatch.setattr(sample_utils, "_HAS_TORCH_QUICKFPS", False)
    monkeypatch.setattr(sample_utils, "_HAS_FPSAMPLE", False)
    monkeypatch.setattr(sample_utils, "_FPSAMPLE_IMPORT_ERROR", ImportError("fpsample missing"))

    with pytest.raises(ImportError, match=r"fpsample>=0\.2\.0"):
        sample_utils._farthest_point_sampling(torch.randn(8, 4), 3)
