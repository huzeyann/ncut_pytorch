import statistics
import time

import pytest
import torch
import torch.nn.functional as F
from ncut_pytorch import kway_ncut, axis_align, quick_kway, ncut_fn
from ncut_pytorch.utils.device import auto_device
from ncut_pytorch.utils.sample import farthest_point_sampling
from ncut_pytorch.utils import sample as sample_utils


def _reference_kmeans_kway(
    eigvec: torch.Tensor,
    n_clusters: int = 10,
    n_eig: int = 10,
    n_sample: int = 10240,
    device: str | None = None,
    kmeans_iter: int = 10,
) -> torch.Tensor:
    original_device = eigvec.device
    original_dtype = eigvec.dtype
    device = auto_device(original_device, device)
    if n_sample is not None and n_sample < eigvec.shape[0]:
        random_indices = torch.randperm(eigvec.shape[0])[:n_sample]
        eigvec = eigvec[random_indices]
    eigvec = eigvec[:, :n_eig]
    eigvec = eigvec.to(device)
    eigvec = F.normalize(eigvec, dim=1)
    indices = farthest_point_sampling(eigvec, n_clusters)
    centroids = eigvec[indices].clone()
    for _ in range(kmeans_iter):
        similarities = torch.mm(eigvec, centroids.t())
        assignments = similarities.argmax(dim=1)
        for k in range(n_clusters):
            mask = assignments == k
            if mask.any():
                centroids[k] = eigvec[mask].mean(dim=0)
                centroids[k] = F.normalize(centroids[k], dim=0)
    R = centroids.t()
    R = R[:, torch.argsort(R[1])]
    return R.to(device=original_device, dtype=original_dtype)


def _reference_farthest_point_sampling(
    X: torch.Tensor,
    n_sample: int,
    max_draw_ratio: float = 4.0,
    max_dim: int = 8,
    device: str | None = None,
) -> torch.Tensor:
    """Reference implementation matching the pre-optimization behavior."""
    num_data = X.shape[0]
    num_draw = int(n_sample * max_draw_ratio)

    if num_draw > num_data:
        return sample_utils._farthest_point_sampling(
            X,
            n_sample=n_sample,
            max_dim=max_dim,
            device=device,
        )

    draw_indices = torch.randperm(num_data)[:num_draw]
    sampled_indices = sample_utils._farthest_point_sampling(
        X[draw_indices],
        n_sample=n_sample,
        max_dim=max_dim,
        device=device,
    )
    return draw_indices[sampled_indices]


def _median_runtime(fn, repeats: int = 3) -> float:
    timings = []
    for seed in range(repeats):
        torch.manual_seed(seed)
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


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
        
        # Check that we can get cluster assignments
        cluster_assignments = rotated_eigvec.argmax(dim=1)
        assert cluster_assignments.shape == (small_feature_matrix.shape[0],)
        assert cluster_assignments.min() >= 0
        assert cluster_assignments.max() < n_eig

    def test_kway_ncut_fp16(self, small_feature_matrix):
        """Test the kway_ncut function with fp16."""
        # First get some eigenvectors using ncut_fn
        n_eig = 5
        eigvec, _ = ncut_fn(small_feature_matrix.half(), n_eig=n_eig)
        
        # Apply kway_ncut
        rotated_eigvec = kway_ncut(eigvec, max_iter=100, n_sample=50)
        assert rotated_eigvec.shape == eigvec.shape
        assert rotated_eigvec.dtype == torch.float16

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

    def test_quick_kway_basic(self, small_feature_matrix):
        """Test the quick_kway function basic functionality."""
        # First get some eigenvectors using ncut_fn
        n_eig = 10
        n_clusters = 5
        eigvec, _ = ncut_fn(small_feature_matrix, n_eig=n_eig)
        
        # Apply quick_kway
        rotated_eigvec = quick_kway(eigvec, n_clusters=n_clusters, n_eig=10, n_sample=50, kmeans_iter=5)
        
        # Check shape
        assert rotated_eigvec.shape == (eigvec.shape[0], n_clusters)
        
        # Check that we can get cluster assignments
        cluster_assignments = rotated_eigvec.argmax(dim=1)
        assert cluster_assignments.shape == (small_feature_matrix.shape[0],)
        assert cluster_assignments.min() >= 0
        assert cluster_assignments.max() < n_clusters

    def test_quick_kway_ret_R(self, small_feature_matrix):
        """Test quick_kway with ret_R=True."""
        n_eig = 10
        n_clusters = 5
        eigvec, _ = ncut_fn(small_feature_matrix, n_eig=n_eig)
        
        # Get rotation matrix
        R = quick_kway(eigvec, n_clusters=n_clusters, n_eig=10, n_sample=50, kmeans_iter=5, ret_R=True)
        
        # Check shape
        assert R.shape == (10, 5)

    def test_quick_kway_different_parameters(self, small_feature_matrix):
        """Test quick_kway with different parameters."""
        n_eig = 8
        eigvec, _ = ncut_fn(small_feature_matrix, n_eig=n_eig)
        
        # Test with different n_clusters
        rotated_eigvec = quick_kway(eigvec, n_clusters=3, n_eig=8, n_sample=50)
        assert rotated_eigvec.shape == (eigvec.shape[0], 3)
        
        # Test with different n_eig
        rotated_eigvec = quick_kway(eigvec, n_clusters=4, n_eig=6, n_sample=50)
        assert rotated_eigvec.shape == (eigvec.shape[0], 4)
        
        # Test with different kmeans_iter
        rotated_eigvec = quick_kway(eigvec, n_clusters=5, n_eig=8, n_sample=50, kmeans_iter=20)
        assert rotated_eigvec.shape == (eigvec.shape[0], 5)

    def test_quick_kway_ret_R_matches_reference_update(self, random_seed):
        """Test that quick_kway keeps the original K-means update semantics."""
        eigvec = torch.randn(256, 12)
        kwargs = {
            "n_clusters": 6,
            "n_eig": 10,
            "n_sample": 128,
            "device": "cpu",
            "kmeans_iter": 6,
        }

        torch.manual_seed(random_seed)
        expected = _reference_kmeans_kway(eigvec.clone(), **kwargs)
        torch.manual_seed(random_seed)
        actual = quick_kway(eigvec.clone(), ret_R=True, **kwargs)

        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

    def test_farthest_point_sampling_matches_full_path_when_presample_skipped(self):
        """Test that the fast path still matches direct FPS when no pre-sampling is needed."""
        X = torch.randn(64, 16)

        torch.manual_seed(0)
        expected = sample_utils._farthest_point_sampling(X, n_sample=32, device="cpu")
        torch.manual_seed(0)
        actual = farthest_point_sampling(X, n_sample=32, device="cpu")

        assert torch.equal(actual, expected)

    def test_farthest_point_sampling_presample_indices_are_sorted(self, random_seed):
        """Test that stratified pre-sampling yields monotonic in-range candidate indices."""
        torch.manual_seed(random_seed)
        draw_indices = sample_utils._stratified_presample_indices(4096, 1024)

        assert draw_indices.shape == (1024,)
        assert torch.all(draw_indices[1:] > draw_indices[:-1])
        assert draw_indices[0] >= 0
        assert draw_indices[-1] < 4096

    def test_farthest_point_sampling_non_cuda_forces_cpu_kernel(self, monkeypatch):
        """Test that non-CUDA environments always execute FPS on CPU tensors."""
        seen = {}

        def fake_sample_idx(x: torch.Tensor, n_sample: int) -> torch.Tensor:
            seen["device"] = x.device.type
            return torch.arange(n_sample, device=x.device)

        monkeypatch.setattr(sample_utils, "_HAS_CUDA_KERNEL", False)
        monkeypatch.setattr(sample_utils, "sample_idx", fake_sample_idx)

        result = sample_utils._farthest_point_sampling(
            torch.randn(32, 4),
            n_sample=8,
            device="mps",
        )

        assert seen["device"] == "cpu"
        assert result.device.type == "cpu"
        assert torch.equal(result, torch.arange(8))

    def test_farthest_point_sampling_returns_cpu_indices_from_accelerator_path(self, monkeypatch):
        """Test that FPS indices are normalized back to CPU for downstream indexing."""
        accelerator = None
        if torch.cuda.is_available():
            accelerator = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator = "mps"

        if accelerator is None:
            pytest.skip("Accelerator device is required to validate CPU index normalization.")

        def fake_sample_idx(x: torch.Tensor, n_sample: int) -> torch.Tensor:
            return torch.arange(n_sample, device=x.device)

        monkeypatch.setattr(sample_utils, "_HAS_CUDA_KERNEL", True)
        monkeypatch.setattr(sample_utils, "sample_idx", fake_sample_idx)

        result = sample_utils._farthest_point_sampling(
            torch.randn(32, 4),
            n_sample=8,
            device=accelerator,
        )

        assert result.device.type == "cpu"
        assert torch.equal(result, torch.arange(8))

    def test_farthest_point_sampling_reference_speedup(self):
        """Test that reduced pre-sampling significantly speeds up FPS on representative data."""
        X = torch.randn(8192, 128)

        torch.manual_seed(0)
        actual = farthest_point_sampling(X, n_sample=2048, device="cpu")

        assert actual.shape == (2048,)
        assert torch.unique(actual).numel() == 2048
        assert actual.min() >= 0
        assert actual.max() < X.shape[0]

        reference_time = _median_runtime(
            lambda: _reference_farthest_point_sampling(X, n_sample=2048, device="cpu")
        )
        optimized_time = _median_runtime(
            lambda: farthest_point_sampling(X, n_sample=2048, device="cpu")
        )

        assert reference_time / optimized_time >= 1.6
