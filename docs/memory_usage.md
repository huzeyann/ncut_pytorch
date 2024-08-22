# How to Optimize Memory Usage

In the case of GPU memory runs out, please consider the following:


### Lower the Sampling Size

The sub-sampled affinity graph $A \in \mathbb{R}^{n \times n}$ is stored and computed on GPU (see [How NCUT Works](how_ncut_works.md)), $n$ is the `num_samples` parameter. $A$ could get large and consume a lot of GPU memory. It's recommended to use as large sampling size as it fits into memory. Please consider lowering the `num_samples` parameter if out of memory.

```py
eigvectors, eigvalues = NCUT(num_eig=20, num_samples=10000).fit_transform(data)
```


### Use CPU to Store, GPU to Compute

For a simple and easy usage of NCUT, the input can be on neither cpu and gpu. To save more GPU memory, please save the input at CPU memory, and use `NCUT(device='cuda:0')`, NCUT will move the critical computation from CPU to GPU.

```py
data = torch.rand(1000, 256, device='cpu')
eigvectors, eigvalues = NCUT(num_eig=20, device='cuda:0').fit_transform(data)
```

> Computation that moved to GPU (see [How NCUT Works](how_ncut_works.md)): 
>
> 1. cosine similarity to compute the sub-sampled graph affinity
> 
> 2. eigen-decomposition on the sub-sampled graph
>
> 3. KNN propagation, each chunk is moved to GPU sequentially
>
> Computation that stays in CPU:
>
> 1. normalization of input features (optional, fast)
> 
> 2. Farthest Point Sampling (QuickFPS, fast)
>
> 3. Post-hoc orthogonalization of eigenvectors (optional, slow)
>

