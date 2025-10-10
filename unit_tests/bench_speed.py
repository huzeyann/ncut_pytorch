import pytest
import torch
from sklearn.manifold import SpectralEmbedding
from ncut_pytorch import Ncut

TEST_SEED = 42
N_DIM = 768
TEST_BENCHMARK_SETTINGS = {
    "ncut_vs_sklearn": {"group": "ncut-pytorch (CPU) vs sklearn", "warmup": True},
    "n_data": {"group": "ncut-pytorch (GPU) n_data", "warmup": True},
    "n_eig": {"group": "ncut-pytorch (GPU) n_eig", "warmup": True},
    "gpu_ncut": {"group": "ncut-pytorch (CPU) vs ncut-pytorch (GPU)", "warmup": True},
}


def create_sample_data(n_points: int, n_dim: int = N_DIM, seed: int = TEST_SEED):
    torch.manual_seed(seed)
    return torch.rand(n_points, n_dim)


####### ncut-pytorch (CPU) vs sklearn #######

@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["ncut_vs_sklearn"])
def test_ncut_cpu_100_data_10_eig(benchmark):
    data = create_sample_data(100, N_DIM)
    ncut = Ncut(n_eig=10, device='cpu')
    benchmark(ncut.fit_transform, data)

@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["ncut_vs_sklearn"])
def test_sklearn_100_data_10_eig(benchmark):
    data = create_sample_data(100, N_DIM)
    spectral_embedding = SpectralEmbedding(n_components=10, affinity='rbf')
    benchmark(spectral_embedding.fit_transform, data)
    
@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["ncut_vs_sklearn"])
def test_ncut_cpu_300_data_10_eig(benchmark):
    data = create_sample_data(300, N_DIM)
    ncut = Ncut(n_eig=10, device='cpu')
    benchmark(ncut.fit_transform, data)

@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["ncut_vs_sklearn"])
def test_sklearn_300_data_10_eig(benchmark):
    data = create_sample_data(300, N_DIM)
    spectral_embedding = SpectralEmbedding(n_components=10, affinity='rbf')
    benchmark(spectral_embedding.fit_transform, data)

@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["ncut_vs_sklearn"])
def test_ncut_cpu_1000_data_10_eig(benchmark):
    data = create_sample_data(1000, N_DIM)
    ncut = Ncut(n_eig=10, device='cpu')
    benchmark(ncut.fit_transform, data)

@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["ncut_vs_sklearn"])
def test_sklearn_1000_data_10_eig(benchmark):
    data = create_sample_data(1000, N_DIM)
    spectral_embedding = SpectralEmbedding(n_components=10, affinity='rbf')
    benchmark(spectral_embedding.fit_transform, data)
    
@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["ncut_vs_sklearn"])
def test_ncut_cpu_3000_data_10_eig(benchmark):
    data = create_sample_data(3000, N_DIM)
    ncut = Ncut(n_eig=10, device='cpu')
    benchmark(ncut.fit_transform, data)

@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["ncut_vs_sklearn"])
def test_sklearn_3000_data_10_eig(benchmark):
    data = create_sample_data(3000, N_DIM)
    spectral_embedding = SpectralEmbedding(n_components=10, affinity='rbf')
    benchmark(spectral_embedding.fit_transform, data)


####### ncut-pytorch (GPU) n_data #######

@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["n_data"])
def test_ncut_gpu_100_data_10_eig(benchmark):
    data = create_sample_data(100, N_DIM)
    ncut = Ncut(n_eig=10, device='cuda')
    benchmark(ncut.fit_transform, data)

@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["n_data"])
def test_ncut_gpu_1000_data_10_eig(benchmark):
    data = create_sample_data(300, N_DIM)
    ncut = Ncut(n_eig=10, device='cuda')
    benchmark(ncut.fit_transform, data)

@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["n_data"])
def test_ncut_gpu_10000_data_10_eig(benchmark):
    data = create_sample_data(10000, N_DIM)
    ncut = Ncut(n_eig=10, device='cuda')
    benchmark(ncut.fit_transform, data)

@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["n_data"])
def test_ncut_gpu_100000_data_10_eig(benchmark):
    data = create_sample_data(100000, N_DIM)
    ncut = Ncut(n_eig=10, device='cuda')
    benchmark(ncut.fit_transform, data)

@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["n_data"])
def test_ncut_gpu_1000000_data_10_eig(benchmark):
    data = create_sample_data(1000000, N_DIM)
    ncut = Ncut(n_eig=10, device='cuda')
    benchmark(ncut.fit_transform, data)
    
####### ncut-pytorch (GPU) n_eig #######

@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["n_eig"])
def test_ncut_gpu_10000_data_10_eig(benchmark):
    data = create_sample_data(10000, N_DIM)
    ncut = Ncut(n_eig=10, device='cuda')
    benchmark(ncut.fit_transform, data)

@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["n_eig"])
def test_ncut_gpu_10000_data_100_eig(benchmark):
    data = create_sample_data(10000, N_DIM)
    ncut = Ncut(n_eig=100, device='cuda')
    benchmark(ncut.fit_transform, data)

@pytest.mark.benchmark(**TEST_BENCHMARK_SETTINGS["n_eig"])
def test_ncut_gpu_10000_data_1000_eig(benchmark):
    data = create_sample_data(10000, N_DIM)
    ncut = Ncut(n_eig=1000, device='cuda')
    benchmark(ncut.fit_transform, data)
