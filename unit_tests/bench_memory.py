import torch
from ncut_pytorch import Ncut
from tabulate import tabulate

TEST_SEED = 42
N_DIM = 768
N_EIG = 10  # Fixed number of eigenvectors

def create_sample_data(n_points: int, n_dim: int = N_DIM, seed: int = TEST_SEED):
    torch.manual_seed(seed)
    return torch.rand(n_points, n_dim)

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def test_gpu_memory(n_points):
    """Test GPU memory usage for a given number of data points"""
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Record initial GPU memory
    initial_mem = get_gpu_memory_usage()
    
    # Run model
    data = create_sample_data(n_points, N_DIM)
    model = Ncut(n_eig=N_EIG, device='cuda')
    result = model.fit_transform(data)
    
    # Record peak memory
    peak_mem = get_gpu_memory_usage()
    
    # Clear GPU cache again
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return peak_mem - initial_mem

def test_memory_scaling():
    """Test memory scaling with different data sizes"""
    if not torch.cuda.is_available():
        print("GPU not available!")
        return
        
    # Test configurations
    data_sizes = [1000, 10000, 100000, 1000000]
    
    # Run tests and collect results
    results = []
    for n_points in data_sizes:
        gpu_mem = test_gpu_memory(n_points)
        results.append([f"{n_points:,}", f"{gpu_mem:.2f}"])
    
    # Print results as a table
    headers = ["Data Points", "Peak GPU Memory (MB)"]
    print("\nGPU Memory Usage Summary")
    print("=======================")
    print(tabulate(results, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    test_memory_scaling()
