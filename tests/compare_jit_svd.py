import torch
import time
from ncut_pytorch.utils.torch_mod import svd_lowrank as my_torch_svd_lowrank


def svd_lowrank(mat: torch.Tensor, q: int):
    """
    SVD lowrank
    mat: (n, m), n data, m features
    q: int
    return: (n, q), (q,), (q, m)
    """
    dtype = mat.dtype
    if dtype == torch.float16 or dtype == torch.bfloat16:
        mat = mat.float()  # svd_lowrank does not support float16

    u, s, v = my_torch_svd_lowrank(mat, q=q + 10)

    u = u[:, :q]
    s = s[:q]
    v = v[:, :q]

    u = u.to(dtype)
    s = s.to(dtype)
    v = v.to(dtype)
    return u, s, v


@torch.jit.script
def svd_lowrank_jit(mat: torch.Tensor, q: int):
    """
    SVD lowrank
    mat: (n, m), n data, m features
    q: int
    return: (n, q), (q,), (q, m)
    """
    dtype = mat.dtype
    if dtype == torch.float16 or dtype == torch.bfloat16:
        mat = mat.float()  # svd_lowrank does not support float16

    u, s, v = my_torch_svd_lowrank(mat, q=q + 10)

    u = u[:, :q]
    s = s[:q]
    v = v[:, :q]

    u = u.to(dtype)
    s = s.to(dtype)
    v = v.to(dtype)
    return u, s, v


def benchmark_svd(mat: torch.Tensor, q: int, n_runs: int = 5):

    times_normal = []
    times_jit = []

    # Warmup
    svd_lowrank(mat, q)
    svd_lowrank_jit(mat, q)

    for _ in range(n_runs):
        start = time.perf_counter()
        svd_lowrank(mat, q)
        times_normal.append(time.perf_counter() - start)

        start = time.perf_counter()
        svd_lowrank_jit(mat, q)
        times_jit.append(time.perf_counter() - start)

    avg_normal = sum(times_normal) / len(times_normal)
    avg_jit = sum(times_jit) / len(times_jit)

    print(f"Normal SVD average time: {avg_normal:.4f}s")
    print(f"JIT SVD average time: {avg_jit:.4f}s")
    print(f"Speedup: {avg_normal / avg_jit:.2f}x")

if __name__ == "__main__":
    device = torch.device("mps")
    mat = torch.randn(10000, 10000, device=device)
    q = 100
    benchmark_svd(mat, q)