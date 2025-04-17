from collections import defaultdict
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ncut_pytorch import nystrom_ncut
from .affinity_gamma import find_gamma_by_degree_after_fps
from .math_utils import compute_riemann_curvature_loss, compute_boundary_loss, compute_repulsion_loss, compute_axis_align_loss


def _kway_ncut_loss(eigvec_gt, eigvec_hat, n_eig):
    _eigvec_gt = eigvec_gt[:, :n_eig]
    _eigvec_hat = eigvec_hat[:, :n_eig]
    loss = F.smooth_l1_loss(_eigvec_gt @ _eigvec_gt.T, _eigvec_hat @ _eigvec_hat.T)
    return loss

def flag_space_loss(eigvec_gt, eigvec_hat, n_eig, start=4, step_mult=2):
    if torch.all(eigvec_gt == 0) or torch.all(eigvec_hat == 0):
        return torch.tensor(0, device=eigvec_gt.device)
    
    loss = 0
    n_eig = start // step_mult
    while True:
        n_eig *= step_mult
        loss += _kway_ncut_loss(eigvec_gt, eigvec_hat, n_eig)
        if n_eig > eigvec_gt.shape[1] or n_eig > eigvec_hat.shape[1]:
            break
    return loss


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, n_layer=4, latent_dim=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.GELU(),
            *[nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU()) for _ in range(n_layer)],
            nn.Linear(latent_dim, out_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)


class CompressionModel(nn.Module):
    def __init__(self, in_dim, mood_dim=2, flag_space_n_eig=32, n_layer=4, latent_dim=512, 
                 eigvec_loss=1, recon_loss=1, riemann_curvature_loss=0, 
                 axis_align_loss=0, repulsion_loss=0.01, boundary_loss=0.01, lr=0.001, gamma=1.0):
        super().__init__()
        
        self.compress = MLP(in_dim, mood_dim, n_layer, latent_dim)
        self.uncompress = MLP(mood_dim, in_dim, n_layer, latent_dim)
                
        self.loss_history = defaultdict(list)

        self.flag_space_n_eig = flag_space_n_eig
        self.eigvec_loss = eigvec_loss
        self.recon_loss = recon_loss
        self.riemann_curvature_loss = riemann_curvature_loss
        self.axis_align_loss = axis_align_loss
        self.repulsion_loss = repulsion_loss
        self.boundary_loss = boundary_loss

        self.lr = lr
        self.gamma = gamma

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        return self.compress(x)


def compute_loss(feats, model):
    """
    Standalone version of compute_loss function.
    
    Args:
        feats: Input features tensor
        model: CompressionModel instance
        gamma: Optional gamma value for nystrom_ncut
        global_step: Current training step
        
    Returns:
        tuple: (total_loss, loss_dict, feats_compressed)
    """
    
    feats.requires_grad_(True)
    feats_compressed = model.compress(feats)
    print(feats_compressed.requires_grad)  # I don't know why the F, but it don't have grad here
    feats_uncompressed = model.uncompress(feats_compressed)
    
    device = feats.device
    eigvec_gt, eigval_gt = nystrom_ncut(feats, model.flag_space_n_eig, gamma=model.gamma, distance='rbf', device=device)
    eigvec_hat, eigval_hat = nystrom_ncut(feats_compressed, model.flag_space_n_eig, gamma=model.gamma, distance='rbf', device=device)

    total_loss = 0
    loss_dict = {}
    
    if model.eigvec_loss > 0:
        eigvec_loss = flag_space_loss(eigvec_gt, eigvec_hat, n_eig=model.flag_space_n_eig)
        loss_dict['eigvec'] = eigvec_loss.item()
        total_loss += eigvec_loss * model.eigvec_loss
        model.loss_history['eigvec'].append(eigvec_loss.item())

    if model.recon_loss > 0:
        recon_loss = F.smooth_l1_loss(feats, feats_uncompressed)
        loss_dict['recon'] = recon_loss.item()
        total_loss += recon_loss * model.recon_loss
        model.loss_history['recon'].append(recon_loss.item())

    if model.riemann_curvature_loss > 0:
        riemann_curvature_loss = compute_riemann_curvature_loss(feats_compressed)
        loss_dict['riemann_curvature'] = riemann_curvature_loss.item()
        total_loss += riemann_curvature_loss * model.riemann_curvature_loss

    if model.axis_align_loss > 0:
        axis_align_loss = compute_axis_align_loss(feats_compressed)
        loss_dict['axis_align'] = axis_align_loss.item()
        total_loss += axis_align_loss * model.axis_align_loss

    if model.repulsion_loss > 0:
        repulsion_loss = compute_repulsion_loss(feats_compressed)
        loss_dict['repulsion'] = repulsion_loss.item()
        total_loss += repulsion_loss * model.repulsion_loss

    if model.boundary_loss > 0:
        boundary_loss = compute_boundary_loss(feats_compressed)
        loss_dict['boundary'] = boundary_loss.item()
        total_loss += boundary_loss * model.boundary_loss

    loss_dict['total'] = total_loss.item()
    return total_loss


def fit_transform_mspace_model(features, flag_space_n_eig=32, mood_dim=2, 
                            n_layer=4, latent_dim=512, 
                            eigvec_loss=1, recon_loss=1, 
                            riemann_curvature_loss=0, axis_align_loss=0, 
                            repulsion_loss=0.01, boundary_loss=1, 
                            lr=0.001, training_steps=1000, grad_clip_val=1.0, 
                            batch_size=1000,
                            **kwargs):
    features = torch.tensor(features).float().cpu()  # [N, C]

    gamma = find_gamma_by_degree_after_fps(features, 0.1, distance='rbf')

    model = CompressionModel(features.shape[1], mood_dim=mood_dim, flag_space_n_eig=flag_space_n_eig, n_layer=n_layer, latent_dim=latent_dim, 
                             eigvec_loss=eigvec_loss, recon_loss=recon_loss, 
                             riemann_curvature_loss=riemann_curvature_loss, 
                             axis_align_loss=axis_align_loss, repulsion_loss=repulsion_loss, boundary_loss=boundary_loss, 
                             lr=lr, gamma=gamma)
    
    

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create optimizer
    optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    
    # Create data loader
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(features)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop - process exactly training_steps batches
    model.train()
    model = model.to(device)
    dataloader_iter = iter(dataloader)
    
    for step in range(training_steps):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
            
        feats = batch[0].to(device)
        # Forward pass and compute loss using the standalone function
        loss = compute_loss(feats, model)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            
        optimizer.step()
    
    # Prediction
    model.eval()
    
    predict_loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False)
    
    compressed_features = []
    with torch.no_grad():
        for batch in predict_loader:
            feats = batch[0].to(device)
            compressed = model.compress(feats)
            compressed_features.append(compressed.cpu())
    
    compressed = torch.cat(compressed_features, dim=0)
    return compressed
