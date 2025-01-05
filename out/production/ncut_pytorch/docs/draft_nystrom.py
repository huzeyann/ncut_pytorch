#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate 1000 2D points in 4 clusters
X, y = make_blobs(n_samples=30, centers=3, n_features=2, random_state=42)
arg_sort = np.argsort(y)
X = X[arg_sort]
y = y[arg_sort]

# Identify centroids of clusters
centroids = np.array([X[y == i].mean(axis=0) for i in range(4)])

# Create a line of dots connecting the 2nd and 3rd cluster centroids
num_line_points = 10
line_points = np.linspace(centroids[0], centroids[1], num_line_points)

# Plot the points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10_r', s=50, alpha=0.6)
plt.scatter(line_points[:, 0], line_points[:, 1], c='grey', s=50, alpha=0.6)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Randomly Placed 2D Dots in 3 Clusters with Connecting Line')
plt.show()
# %%
all_dots = np.concatenate([X, line_points], axis=0)
# %%
def ncut(A):
    eigen_value, eigen_vector = torch.linalg.eigh(A.float())
    # sort eigenvectors by eigenvalues, take top num_eig
    args = torch.argsort(eigen_value, descending=True)
    eigen_value = eigen_value[args]
    eigen_vector = eigen_vector[:, args]

    eigen_value = eigen_value.real
    eigen_vector = eigen_vector.real
    return eigen_value, eigen_vector

def propagate_V_knn(
    subgraph_eigen_vector,
    A,
    rand_indices,
    knn,
    chunk_size=8096,
    device="cpu",
    use_tqdm=True,
):
    # used in nystrom_ncut
    # propagate eigen_vector from subgraph to full graph
    subgraph_eigen_vector = subgraph_eigen_vector.to(device)
    V_list = []
    if use_tqdm:
        try:
            from tqdm import tqdm
        except ImportError:
            use_tqdm = False
    if use_tqdm:
        iterator = tqdm(range(0, A.shape[0], chunk_size), "propagate_V")
    else:
        iterator = range(0, A.shape[0], chunk_size)
    for i in iterator:
        end = min(i + chunk_size, A.shape[0])
        
        _A = A[i:end][:, rand_indices]

        # keep topk KNN for each row
        topk_sim, topk_idx = _A.topk(knn, dim=-1, largest=True)
        row_id = torch.arange(topk_idx.shape[0], device=_A.device)[:, None].expand(
            -1, topk_idx.shape[1]
        )
        _A = torch.sparse_coo_tensor(
            torch.stack([row_id, topk_idx], dim=-1).reshape(-1, 2).T,
            topk_sim.reshape(-1),
            size=(_A.shape[0], _A.shape[1]),
            device=_A.device,
        )
        _A = _A.to_dense().to(dtype=subgraph_eigen_vector.dtype)
        # _A is KNN graph

        _D = _A.sum(-1)
        _A /= _D[:, None]

        _V = _A @ subgraph_eigen_vector

        V_list.append(_V.cpu())

    subgraph_eigen_vector = torch.cat(V_list, dim=0)

    return subgraph_eigen_vector


def nystrom_nut(A, num_sample=10, knn=3, sample_indices=None):
    if sample_indices is None:
        rand_indices = np.random.choice(A.shape[0], num_sample, replace=False)
    else:
        rand_indices = sample_indices
    _A = A[rand_indices][:, rand_indices]
    eigen_value, eigen_vector = ncut(_A)
    eigen_vector = propagate_V_knn(eigen_vector, A, rand_indices, knn=knn)
    return eigen_value, eigen_vector

def fps_sampling(A, num_sample=10, seed=42):
    np.random.seed(seed)
    rand_indices = [np.random.choice(A.shape[0])]
    for i in range(num_sample - 1):
        _A = A[rand_indices]
        min_dist = _A.min(dim=0).values
        # sorted_idx = torch.argsort(min_dist, descending=True)
        # next_idx = sorted_idx[len(rand_indices)].item()
        next_idx = torch.argmax(min_dist).item()
        print(next_idx)
        rand_indices.append(next_idx)
    rand_indices = torch.tensor(rand_indices)
    return rand_indices

# %%

def top_and_bottom_percent_clamp(d, p=0.01):
    new_vmax = d.quantile(1 - p)
    new_vmin = d.quantile(p)
    d = torch.clamp(d, min=new_vmin, max=new_vmax)
    return d


def eig_inv_truck(A, n_inv=-1):

    # if n_inv > 0:
    #     ei, ev = torch.lobpcg(A, k=n_inv)
    #     c = ei.sum() / torch.trace(A)
    #     # ei *= c
    #     print("eig_inv, c", c)
    # else:
    ei, ev = torch.linalg.eigh(A)
    print("eig_inv, ei", ei.min(), ei.max())
    A_inv = ev @ torch.diag(1 / (ei.abs() + 1e-8)) @ ev.T
    return A_inv


def symmetric_normalize(A, B, D):
    M = A.shape[0]
    A /= torch.sqrt(D)[:M, None]
    A /= torch.sqrt(D)[None, :M]
    B /= torch.sqrt(D)[:M, None]
    B /= torch.sqrt(D)[None, M:]
    return A, B

def chunked_matmul(
    A,
    B,
    chunk_size=8096,
    device="cuda:0",
    large_device="cpu",
    transform=lambda x: x,
):
    A = A.to(large_device)
    B = B.to(large_device)
    C = torch.zeros(A.shape[0], B.shape[1], device=large_device)
    for i in range(0, A.shape[0], chunk_size):
        end_i = min(i + chunk_size, A.shape[0])
        for j in range(0, B.shape[1], chunk_size):
            end_j = min(j + chunk_size, B.shape[1])
            _A = A[i:end_i].to(device)
            _B = B[:, j:end_j].to(device)
            _C = _A @ _B
            _C = transform(_C)
            _C = _C.to(large_device)
            C[i:end_i, j:end_j] = _C
    return C


def chunked_matmul(
    A,
    B,
    chunk_size=8096,
    device="cuda:0",
    large_device="cpu",
    transform=lambda x: x,
    use_tqdm=True,
):
    A = A.to(large_device)
    B = B.to(large_device)
    C = torch.zeros(A.shape[0], B.shape[1], device=large_device)
    iterator = range(0, A.shape[0], chunk_size)
    for i in iterator:
        end_i = min(i + chunk_size, A.shape[0])
        for j in range(0, B.shape[1], chunk_size):
            end_j = min(j + chunk_size, B.shape[1])
            _A = A[i:end_i]
            _B = B[:, j:end_j]
            _C_ij = None
            for k in range(0, A.shape[1], chunk_size):
                end_k = min(k + chunk_size, A.shape[1])
                __A = _A[:, k:end_k].to(device)
                __B = _B[k:end_k].to(device)
                _C = __A @ __B
                _C_ij = _C if _C_ij is None else _C_ij + _C
            _C_ij = transform(_C_ij)

            _C_ij = _C_ij.to(large_device)
            C[i:end_i, j:end_j] = _C_ij
    return C


def get_d(A, B, n_inv=-1, device="cuda:0", chunk_size=8096):
    # A : (m, m)
    # B : (m, n)
    d_A, d_B = A.sum(1), B.sum(1)
    d1 = d_A + d_B

    # d2 = B.sum(0) + B.T @ eig_inv(A, n_inv=n_inv) @ B.sum(1)
    # code below is memory efficient to the line above
    # A_inv = torch.linalg.pinv(A.to(device))
    A_inv = eig_inv_truck(A.to(device).float(), n_inv=n_inv)
    A_inv = A_inv.to(dtype=B.dtype)
    B_sum = d_B.to(device)
    _right_mat = A_inv @ B_sum.reshape(-1, 1)
    # _right_mat = _right_mat.abs() # FIXME: this is a hack
    d2 = chunked_matmul(B.T, _right_mat, chunk_size=chunk_size, device=device)
    d2 = d2.reshape(-1)
    d2 = d2 + B.sum(0)

    d = torch.cat([d1, d2])
    if d.min() <= 0 and d.min() > -1e-1:
        d = d.abs()
    d += 1e-5
    assert torch.all(d > 0), f"negative degree found, {d.min()}"
    print("degree", d.min(), d.max())
    return d


def get_topk_V(A, B, num_eig=20, n_inv=100, device="cuda:0", chunk_size=8096):
    # ei, ev = torch.linalg.eigh(A)
    A = A.to(device)
    ei, ev = torch.lobpcg(A.float(), k=n_inv)
    ei, ev = ei.to(B.dtype), ev.to(B.dtype)
    c = ei.sum() / torch.trace(A)
    print("get_topk_V, c", c)
    print("get_topk_V, ei", ei.min(), ei.max())
    A_sqrt_inv = ev @ torch.diag(1 / torch.sqrt(ei.abs() + 1e-8)) @ ev.T

    BB_T = chunked_matmul(B, B.T, chunk_size=chunk_size, device=device)
    BB_T = BB_T.to(device)
    print("BB_T", BB_T.min(), BB_T.max())
    S = A + A_sqrt_inv @ BB_T @ A_sqrt_inv
    
    indirect_A = S - A

    # ei_s, ev_s = torch.lobpcg(S.float(), k=num_eig)
    ei_s, ev_s = torch.linalg.eigh(S)
    arg_sort = torch.argsort(ei_s, descending=True)
    ei_s, ev_s = ei_s[arg_sort], ev_s[:, arg_sort]
    # ei_s, ev_s = ei_s[:num_eig], ev_s[:, :num_eig]
    ei_s, ev_s = ei_s.to(B.dtype), ev_s.to(B.dtype)

    V_left = torch.cat([A.cpu(), B.T.cpu()], dim=0)
    # V = V_left @ A_sqrt_inv @ ev_s @ torch.diag(1 / torch.sqrt(ei_s.abs() + 1e-8))
    # the following code is memory efficient to the line above
    __r = chunked_matmul(A_sqrt_inv, ev_s, chunk_size=chunk_size, device=device)
    _ei_s = torch.diag(1 / torch.sqrt(ei_s.abs() + 1e-8))
    _right_mat = chunked_matmul(__r, _ei_s, chunk_size=chunk_size, device=device)
    V = chunked_matmul(V_left, _right_mat, chunk_size=chunk_size, device=device)
    return V, ei_s, indirect_A


def knn_each_row(A, k=100):
    topk_sim, topk_idx = A.topk(k, dim=-1, largest=True)
    row_id = torch.arange(topk_idx.shape[0], device=A.device)[:, None].expand(
        -1, topk_idx.shape[1]
    )
    A = torch.sparse_coo_tensor(
        torch.stack([row_id, topk_idx], dim=-1).reshape(-1, 2).T,
        topk_sim.reshape(-1),
        size=(A.shape[0], A.shape[1]),
        device=A.device,
    )
    return A

from torch.nn import functional as F
@torch.no_grad()
def original_nystrom(
    W,
    sample_indices=None,
    num_sample=2048,
    t=1.0,
    num_eig=50,
    n_inv=5,
    device="cuda:0",
    chunk_size=8096,
    normalize=True,
    knn_B=-1,
):
    num_sample = len(sample_indices)
    not_sampled = np.setdiff1d(np.arange(W.shape[0]), sample_indices)
    indices = np.concatenate([sample_indices, not_sampled])
    reverse_indices = np.argsort(indices)

    A = W[sample_indices][:, sample_indices]
    B = W[sample_indices][:, not_sampled]

    if knn_B > 0:
        # A = knn_each_row(A.T, k=knn_B).to_dense().T
        # A = 0.5 * (A + A.T)
        B = knn_each_row(B.T, k=knn_B).to_dense().T

    D = get_d(A, B, n_inv=n_inv, device=device, chunk_size=chunk_size)
    A, B = symmetric_normalize(A, B, D)
    # D = A.sum(1) + B.sum(1)
    # A /= D[:, None]
    # B /= D[:, None]

    V, L, indirect_A = get_topk_V(
        A, B, num_eig=num_eig, n_inv=n_inv, device=device, chunk_size=chunk_size
    )
    V = V[reverse_indices]

    return L, V, indirect_A
# %%
euclidean_dists = np.linalg.norm(all_dots[:, None] - all_dots, axis=-1)
th = 3
mask = euclidean_dists > th
# euclidean_dists[mask] = 0
w = np.exp(-euclidean_dists**2/line_points.std()**2)
w[mask] = 0

import torch
A = torch.tensor(w, dtype=torch.float32)
D = A.sum(dim=-1).detach().clone()
A /= torch.sqrt(D)[:, None]
A /= torch.sqrt(D)[None, :]

# sample_indices = np.concatenate([np.arange(0, 30, 2), np.array([33, 36])])
# sample_indices = np.arange(40)
euclidean_dists = torch.tensor(euclidean_dists, dtype=torch.float32)
sample_indices = fps_sampling(euclidean_dists, num_sample=17, seed=42)
eigen_value, eigen_vector = nystrom_nut(A, num_sample=20, knn=3, sample_indices=sample_indices)
eigen_value, eigen_vector, indirect_A = original_nystrom(A, sample_indices=sample_indices)
for i in range(len(sample_indices)):
    for j in range(i + 1, len(sample_indices)):
        A[sample_indices[i], sample_indices[j]] = indirect_A[i, j]
eigen_value, eigen_vector = ncut(A)

fig, axs = plt.subplots(1, 7, figsize=(15, 2))
# Connectivity plot
X = all_dots
ax = axs[0]
ax.scatter(X[:, 0], X[:, 1], c='grey', s=50, alpha=0.2)
for i in range(A.shape[0]):
    for j in range(i + 1, A.shape[1]):
        if i not in sample_indices or j not in sample_indices:
            continue
        if A[i, j] > 0.01:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'k-', alpha=0.5)
ax.set_title('Connectivity')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
for i, ax in enumerate(axs):
    if i == 0:
        continue
    vminmax = max(abs(eigen_vector[:, i].min()), abs(eigen_vector[:, i].max()))
    ax.scatter(all_dots[:, 0], all_dots[:, 1], c=eigen_vector[:, i], cmap='coolwarm', s=50, alpha=0.9, vmin=-vminmax, vmax=vminmax)
    ax.set_title(f'Eigenvalue={eigen_value[i]:.2f}')
    if i == 0:
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
    ax.set_xticks([])
    ax.set_yticks([])

# plt.suptitle('Full graph Ncut\n col1 is connectivity (after proximity threshold), col2-7 are eigenvectors', y=-0.1)
plt.suptitle('My approximation w/o indirect connection\n col1 is connectivity on sampled nodes (after proximity threshold), col2-7 are eigenvectors', y=-0.1)
# plt.suptitle('Original Nystrom cut w/ indirect connection \n col1 is connectivity on sampled nodes (after proximity threshold), col2-7 are eigenvectors', y=-0.1)
# %%
# Calculate global vmin and vmax for the first and second eigenvectors
vmin1 = min(eigen_vector[:, 1].min(), eigen_vector[sample_indices, 1].min())
vmax1 = max(eigen_vector[:, 1].max(), eigen_vector[sample_indices, 1].max())
vmin2 = min(eigen_vector[:, 2].min(), eigen_vector[sample_indices, 2].min())
vmax2 = max(eigen_vector[:, 2].max(), eigen_vector[sample_indices, 2].max())

# Create a 2x2 plot
fig, axs = plt.subplots(1, 2, figsize=(8, 3))

# Plot the eigenvector for sampled dots
ax = axs[0]
ax.scatter(all_dots[:, 0], all_dots[:, 1], c='grey', s=50, alpha=0.2)
sc = ax.scatter(all_dots[sample_indices, 0], all_dots[sample_indices, 1], c=eigen_vector[sample_indices, 1], cmap='coolwarm', s=50, alpha=0.9, vmin=vmin1, vmax=vmax1)
ax.set_title('Solved Eigenvector')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.scatter(all_dots[sample_indices, 0], all_dots[sample_indices, 1], facecolors='none', edgecolors='k', s=60)
ax.set_xticks([])
ax.set_yticks([])

# Plot the eigenvector for KNN propagated
ax = axs[1]
sc = ax.scatter(all_dots[:, 0], all_dots[:, 1], c=eigen_vector[:, 1], cmap='coolwarm', s=50, alpha=0.9, vmin=vmin1, vmax=vmax1)
ax.set_title('Propagated Eigenvector')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.scatter(all_dots[sample_indices, 0], all_dots[sample_indices, 1], facecolors='none', edgecolors='k', s=60)
ax.set_xticks([])
ax.set_yticks([])

# # Plot the second eigenvector for sampled dots
# ax = axs[1, 0]
# ax.scatter(all_dots[:, 0], all_dots[:, 1], c='grey', s=50, alpha=0.2)
# sc = ax.scatter(all_dots[sample_indices, 0], all_dots[sample_indices, 1], c=eigen_vector[sample_indices, 2], cmap='coolwarm', s=50, alpha=0.9, vmin=vmin2, vmax=vmax2)
# ax.set_title('Second Eigenvector (Sampled Dots)')
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.scatter(all_dots[sample_indices, 0], all_dots[sample_indices, 1], facecolors='none', edgecolors='k', s=60)
# ax.set_xticks([])
# ax.set_yticks([])

# # Plot the second eigenvector for KNN propagated
# ax = axs[1, 1]
# sc = ax.scatter(all_dots[:, 0], all_dots[:, 1], c=eigen_vector[:, 2], cmap='coolwarm', s=50, alpha=0.9, vmin=vmin2, vmax=vmax2)
# ax.set_title('Second Eigenvector (KNN propagated)')
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.scatter(all_dots[sample_indices, 0], all_dots[sample_indices, 1], facecolors='none', edgecolors='k', s=60)
# ax.set_xticks([])
# ax.set_yticks([])

# plt.tight_layout()
plt.show()


# %%
import seaborn as sns
sns.heatmap(indirect_A.cpu().numpy())
plt.title('Indirect Connection Matrix (on sampled nodes)')
# %%
plt.scatter(all_dots[:, 0], all_dots[:, 1], c='grey', s=50, alpha=0.2)
for i in range(len(sample_indices)):
    plt.scatter(all_dots[sample_indices[i], 0], all_dots[sample_indices[i], 1], c='r', s=50, alpha=0.9)
plt.title('Sampled Dots')
# %%
X = all_dots
A = torch.tensor(w, dtype=torch.float32)
D = A.sum(dim=-1).detach().clone()
A /= torch.sqrt(D)[:, None]
A /= torch.sqrt(D)[None, :]

fig, axs = plt.subplots(1, 2, figsize=(8, 3))
for ax in axs.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    

ax = axs[0]
ax.scatter(X[:, 0], X[:, 1], c='grey', s=50, alpha=0.2)
for i in range(A.shape[0]):
    for j in range(i + 1, A.shape[1]):
        if A[i, j] > 0.01:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'k-', alpha=0.5)
ax.set_title('Full Graph')

sample_indices = fps_sampling(euclidean_dists, num_sample=17, seed=42)


# not_sampled = np.setdiff1d(np.arange(A.shape[0]), sample_indices)
# B = A[sample_indices][:, not_sampled]
# B_row_sum = B.sum(1)
# B_col_sum = B.sum(0)
# indirect_A = (B / B_row_sum[:, None]) @ (B / B_col_sum[None, :]).T

# for ii, i in enumerate(sample_indices):
#     for jj, j in enumerate(sample_indices):
#         A[i, j] += indirect_A[ii, jj]
        
ax = axs[1]
ax.scatter(X[:, 0], X[:, 1], c='grey', s=50, alpha=0.2)
# highlight sampled points
for i in range(len(sample_indices)):
    ax.scatter(X[sample_indices[i], 0], X[sample_indices[i], 1], c='r', s=25, alpha=0.9)
for i in range(A.shape[0]):
    for j in range(i + 1, A.shape[1]):
        if i not in sample_indices or j not in sample_indices:
            continue
        if A[i, j] > 0.01:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'k-', alpha=0.5)
ax.set_title('Sub-sampled Graph')
# %%

A = torch.tensor(w, dtype=torch.float32)
D = A.sum(dim=-1).detach().clone()
A /= torch.sqrt(D)[:, None]
A /= torch.sqrt(D)[None, :]

sample_indices = fps_sampling(euclidean_dists, num_sample=17, seed=42)

fig, axs = plt.subplots(1, 2, figsize=(8, 3))
for ax in axs.flatten():
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    
ax = axs[0]
ax.scatter(X[:, 0], X[:, 1], c='grey', s=50, alpha=0.2)
# highlight sampled points
for i in range(len(sample_indices)):
    ax.scatter(X[sample_indices[i], 0], X[sample_indices[i], 1], c='r', s=25, alpha=0.9)
for i in range(A.shape[0]):
    for j in range(i + 1, A.shape[1]):
        if i not in sample_indices or j not in sample_indices:
            continue
        if A[i, j] > 0.01:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'k-', alpha=0.5)
            
            
ax = axs[1]

not_sampled = np.setdiff1d(np.arange(A.shape[0]), sample_indices)
B = A[sample_indices][:, not_sampled]
B_row_sum = B.sum(1)
B_col_sum = B.sum(0)
indirect_A = (B / B_row_sum[:, None]) @ (B / B_col_sum[None, :]).T

for ii, i in enumerate(sample_indices):
    for jj, j in enumerate(sample_indices):
        A[i, j] += indirect_A[ii, jj]
        
ax.scatter(X[:, 0], X[:, 1], c='grey', s=50, alpha=0.2)
# highlight sampled points
for i in range(len(sample_indices)):
    ax.scatter(X[sample_indices[i], 0], X[sample_indices[i], 1], c='r', s=25, alpha=0.9)
for i in range(A.shape[0]):
    for j in range(i + 1, A.shape[1]):
        if i not in sample_indices or j not in sample_indices:
            continue
        if A[i, j] > 0.01:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'k-', alpha=0.5)
            
axs[0].set_title('no indirect connection')
axs[1].set_title('add indirect connection')
plt.suptitle('Farthest Point Sampling, 17 points', y=-0.03)
# plt.tight_layout()
# %%
