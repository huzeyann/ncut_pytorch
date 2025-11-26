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
    
def plot_dots_after_align(dots1, dots2, feat1, feat2, gammaA2=1.0):
    X = torch.cat([feat1, feat2], dim=0)
    A2 = rbf_affinity_from_features(X, gamma=gammaA2)
    fig, axes = _plot_two_set_dots_with_affinity(dots1, dots2, A2)
    return fig, axes
    
# %%

import pytorch_lightning as pl
from torch import nn

class MLPLayerAlign(pl.LightningModule):
    def __init__(self, n_classes: int, last_layer_params=None):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, n_classes),
        )
        if last_layer_params is not None:
            self.mlp[-1].weight.data = last_layer_params['weight']
            self.mlp[-1].bias.data = last_layer_params['bias']
            self.mlp[-1].weight.requires_grad = False
            self.mlp[-1].bias.requires_grad = False
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.mlp(x)
    
    def get_features(self, x):
        return self.mlp[:-2](x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_feat = self.mlp[:-1](x)
        y_hat = self.mlp[-1](y_feat)
        loss = self.loss(y_hat, y)
        
        affinity1 = rbf_affinity_from_features(x)
        affinity2 = rbf_affinity_from_features(y_feat)
        reg = torch.norm(affinity1 - affinity2, p=2)
        loss += reg
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)
        return optimizer
    
from torch.utils.data import DataLoader, TensorDataset

def make_dataloader(dots, y):
    dataset = TensorDataset(dots, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    return dataloader


def mlp_feature(dots, y, last_layer_params=None):
    dots = dots.clone()
    dots = (dots - dots.mean(dim=0)) / dots.std(dim=0)
    n_classes = y.max().item() + 1
    model = MLPLayerAlign(n_classes=n_classes, last_layer_params=last_layer_params)
    dataloader = make_dataloader(dots, y)
    trainer = pl.Trainer(max_epochs=3000)
    trainer.fit(model, dataloader)
    features = model.get_features(dots).detach()
    last_layer_params = {
        'weight': model.mlp[-1].weight.data,
        'bias': model.mlp[-1].bias.data,
    }
    return features, last_layer_params
# %%
    
dots = make_dots()
theta = np.radians(45)
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
dots2 = dots @ rotation_matrix * torch.tensor([0.25, 4]) + 10
dots2 = dots2.float()
# %%
# y = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
# y = torch.arange(30)
y = torch.randint(0, 3, (30,))
features, last_layer_params = mlp_feature(dots, y)
features2, _ = mlp_feature(dots2, y, last_layer_params)
# %%
plot_dots_after_align(dots, dots2, features, features2, gammaA2=1.0)
plt.suptitle('Z is ground truth 3 clusters')
# %%
