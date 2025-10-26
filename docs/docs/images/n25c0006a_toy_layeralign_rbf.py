#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs


def make_dots():
    X, y = make_blobs(n_samples=30, centers=3, n_features=2, random_state=42)
    arg_sort = np.argsort(y)
    X = X[arg_sort]
    y = y[arg_sort]
    X = torch.tensor(X, dtype=torch.float32)
    return X

def rbf_affinity_from_features(
    features: torch.Tensor,
    features_B: torch.Tensor = None,
    gamma: float = 1.0,
):
    features_B = features if features_B is None else features_B

    d = torch.cdist(features, features_B, p=2)
    A = torch.pow(d, 2)

    sigma = 2 * gamma * features.var(dim=0).sum()
    # sigma = 2 * gamma 
    A = torch.exp(-A / sigma)
    
    return A

def _plot_two_set_dots_with_affinity(dots1, dots2, A):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].scatter(dots1[:, 0], dots1[:, 1], s=100, alpha=1.0, marker='o', label='Set 1')
    for i, (x, y) in enumerate(dots1):
        axes[0].text(x, y, str(i), color=[0.35, 0.35, 0.35], fontsize=8, ha='center', va='center')
    if len(dots2) > 0:  # dots2 is empty tensor, plot only dots1
        axes[0].scatter(dots2[:, 0], dots2[:, 1], s=100, alpha=1.0, marker='s', label='Set 2')
        for i, (x, y) in enumerate(dots2):
            axes[0].text(x, y, str(i + len(dots1)), color=[0.35, 0.35, 0.35], fontsize=8, ha='center', va='center')
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    axes[0].legend()
    axes[0].set_title('Data Points')
    
        

    
    sns.heatmap(A.numpy(), cmap='magma', square=True, ax=axes[1], cbar=True, cbar_kws={'shrink': 0.5}, vmin=0, vmax=1)
    axes[1].set_title('Affinity Matrix')
    ticks = np.arange(0, A.shape[0], 10)
    axes[1].set_xticks(ticks)
    axes[1].set_yticks(ticks)
    axes[1].set_xticklabels(ticks)
    axes[1].set_yticklabels(ticks)
    return fig, axes

def plot_dots_with_affinity_align(dots1, dots2, gamma1=1.0, gamma2=1.0, gammaA2=1.0, less_dot=False, make_outlier=False):
    if make_outlier:
        outlier_idx = [4, 12]
        dots2[outlier_idx, 0] = 50
    if less_dot:
        _dots1 = dots1[-5:]
        _dots2 = dots2[-5:]
    else:
        _dots1 = dots1
        _dots2 = dots2
    A1_1 = rbf_affinity_from_features(dots1, _dots1, gamma=gamma1)
    A1_2 = rbf_affinity_from_features(dots2, _dots2, gamma=gamma2)
    print(A1_1.shape, A1_2.shape)
    A1 = torch.cat([A1_1, A1_2], dim=0)
    A2 = rbf_affinity_from_features(A1, gamma=gammaA2)
    fig, axes = _plot_two_set_dots_with_affinity(dots1, dots2, A2)
    plt.suptitle(f"after alignment")
    return fig, axes
    
def plot_dots_with_affinity_not_align(dots1, dots2, gammaA2=1.0, make_outlier=False):
    if make_outlier:
        # outlier_idx = [4, 5, 12, 13, 14]
        outlier_idx = [4, 12]
        dots2[outlier_idx, 0] = 50
    X = torch.cat([dots1, dots2], dim=0)
    A2 = rbf_affinity_from_features(X, gamma=gammaA2)
    fig, axes = _plot_two_set_dots_with_affinity(dots1, dots2, A2)
    plt.suptitle(f"before alignment")
    return fig, axes
    
# %%
dots = make_dots()
plot_dots_with_affinity_not_align(dots, torch.tensor([]))
plot_dots_with_affinity_not_align(dots, dots)
# %%
theta = np.radians(45)
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
plot_dots_with_affinity_not_align(dots, 0.25*dots @ rotation_matrix + 10)
plot_dots_with_affinity_align(dots, 0.25*dots @ rotation_matrix + 10)
# %%
plot_dots_with_affinity_not_align(dots, dots @ rotation_matrix * torch.tensor([0.25, 4]) + 10)
plot_dots_with_affinity_align(dots, dots @ rotation_matrix * torch.tensor([0.25, 4]) + 10)
# %%
rotation_matrix
# %%
theta = np.radians(45)
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
plot_dots_with_affinity_not_align(dots, 0.25*dots @ rotation_matrix + 10)
plot_dots_with_affinity_align(dots, 0.25*dots @ rotation_matrix + 10, less_dot=True)
# %%
theta = np.radians(45)
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
plot_dots_with_affinity_not_align(dots, 0.25*dots @ rotation_matrix + 10, make_outlier=True)
plot_dots_with_affinity_align(dots, 0.25*dots @ rotation_matrix + 10, make_outlier=True)
# %%
plot_dots_with_affinity_align(dots, 0.25*dots @ rotation_matrix + 10, make_outlier=True, gamma1=1.0, gamma2=0.01)
# %%
plot_dots_with_affinity_align(dots, 0.25*dots @ rotation_matrix + 10, make_outlier=True, gamma1=1.0, gamma2=1.0)
# plt.title('gamma1=1.0\ngamma2=1.0')
# %%
plot_dots_with_affinity_align(dots, 0.25*dots @ rotation_matrix + 10, make_outlier=True, gamma1=1.0, gamma2=0.01)
plt.title('set1: gamma=1.0\nset2: gamma=0.01')

# %%
def find_optimal_gamma(dots, c):
    mean_ds = []
    for gamma in np.arange(0.0001, 1.0, 0.001):
        mean_d = rbf_affinity_from_features(dots, dots, gamma=gamma).mean()
        mean_ds.append(mean_d)
    mean_ds = np.array(mean_ds)
    diff = np.abs(mean_ds - c)
    optimal_gamma = np.arange(0.0001, 1.0, 0.001)[np.argmin(diff)]
    return optimal_gamma
                                             
# %%
gamma1 = find_optimal_gamma(dots, 0.3)
print(gamma1)
dots2 = 0.25*dots @ rotation_matrix + 10
outlier_idx = [4, 12]
dots2[outlier_idx, 0] = 50
gamma2 = find_optimal_gamma(dots2, 0.3)
print(gamma2)
# %%
plot_dots_with_affinity_align(dots, dots2, gamma1=gamma1, gamma2=gamma2)
plt.title(f'set1: gamma={gamma1:.3f}\nset2: gamma={gamma2:.3f}')
# %%
# non-linear transformation
mlp = torch.nn.Sequential(
    torch.nn.Linear(2, 8),
    torch.nn.GELU(),
    torch.nn.Linear(8, 8),
    torch.nn.GELU(),
    torch.nn.Linear(8, 8),
    torch.nn.GELU(),
    torch.nn.Linear(8, 2)
)

mlp100 = []
mlp100 += [
    torch.nn.Linear(2, 8),
    torch.nn.GELU(),
]
for i in range(12):
    mlp100 += [
        torch.nn.Linear(8, 8),
        torch.nn.GELU(),
    ]
mlp100 += [
    torch.nn.Linear(8, 2)
]
mlp100 = torch.nn.Sequential(*mlp100)
# %%
with torch.no_grad():
    # z-score before applying non-linear transformation
    _dots = (dots - dots.mean(dim=0)) / dots.std(dim=0)
    dots2 = mlp(_dots)
plot_dots_with_affinity_align(dots, dots2)
plt.title(f'set1: gamma=1.0\nset2: gamma=1.0')
gamma1 = find_optimal_gamma(dots, 0.3)
gamma2 = find_optimal_gamma(dots2, 0.3)
plot_dots_with_affinity_align(dots, dots2, gamma1=gamma1, gamma2=gamma2)
plt.title(f'set1: gamma={gamma1:.3f}\nset2: gamma={gamma2:.3f}')
# %%
with torch.no_grad():
    # z-score before applying non-linear transformation
    _dots = (dots - dots.mean(dim=0)) / dots.std(dim=0)
    dots2 = mlp100(dots)
plot_dots_with_affinity_align(dots, dots2)
plt.title(f'set1: gamma=1.0\nset2: gamma=1.0')
gamma1 = find_optimal_gamma(dots, 0.3)
gamma2 = find_optimal_gamma(dots2, 0.3)
plot_dots_with_affinity_align(dots, dots2, gamma1=gamma1, gamma2=gamma2)
plt.title(f'set1: gamma={gamma1:.3f}\nset2: gamma={gamma2:.3f}')

# %%
dots2 = dots @ rotation_matrix * torch.tensor([0.25, 4]) + 10
gamma2 = find_optimal_gamma(dots2, 0.3)
plot_dots_with_affinity_align(dots, dots2, gamma1=gamma1, gamma2=gamma2)
plt.title(f'set1: gamma={gamma1:.3f}\nset2: gamma={gamma2:.3f}')
plot_dots_with_affinity_align(dots, dots2, gamma1=1.0, gamma2=1.0)
plt.title(f'set1: gamma=1.0\nset2: gamma=1.0')
# %%
dots2 = dots @ rotation_matrix * torch.tensor([0.001, 1]) + 10
gamma2 = find_optimal_gamma(dots2, 0.3)
plot_dots_with_affinity_align(dots, dots2, gamma1=gamma1, gamma2=gamma2)
plt.title(f'set1: gamma={gamma1:.3f}\nset2: gamma={gamma2:.3f}')
plot_dots_with_affinity_align(dots, dots2, gamma1=1.0, gamma2=1.0)
plt.title(f'set1: gamma=1.0\nset2: gamma=1.0')
# %%
