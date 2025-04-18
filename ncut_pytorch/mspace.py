from collections import defaultdict
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import TensorDataset


from .ncut_pytorch import affinity_from_features, ncut
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

def ncut_wrapper(features, n_eig, distance='rbf', gamma=0.5):
    A = affinity_from_features(features, distance=distance, gamma=gamma)
    eigvec, eigval = ncut(A, n_eig)
    return eigvec, eigval


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, n_layer=2, latent_dim=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.GELU(),
            *[nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU()) for _ in range(n_layer)],
            nn.Linear(latent_dim, out_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)


class CompressionModel(pl.LightningModule):
    def __init__(self, in_dim, mood_dim=2, n_eig=32, 
                 n_layer=2, latent_dim=512, 
                 eigvec_loss=1, recon_loss=1, 
                 riemann_curvature_loss=0, axis_align_loss=0, 
                 repulsion_loss=0.01, boundary_loss=0.01, 
                 lr=0.001):
        super().__init__()
        
        self.compress = MLP(in_dim, mood_dim, n_layer, latent_dim)
        self.uncompress = MLP(mood_dim, in_dim, n_layer, latent_dim)
                
        self.loss_history = defaultdict(list)

        self.n_eig = n_eig
        self.eigvec_loss = eigvec_loss
        self.recon_loss = recon_loss
        self.riemann_curvature_loss = riemann_curvature_loss
        self.axis_align_loss = axis_align_loss
        self.repulsion_loss = repulsion_loss
        self.boundary_loss = boundary_loss

        self.lr = lr

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        return self.compress(x)

    def training_step(self, batch):
        input_feats, output_feats = batch

        if self.trainer.global_step == 0:
            self.gamma = find_gamma_by_degree_after_fps(input_feats, 0.1, distance='rbf')

        feats_compressed = self.compress(input_feats)
        feats_uncompressed = self.uncompress(feats_compressed)
        
        eigvec_gt, eigval_gt = ncut_wrapper(input_feats, self.n_eig, gamma=self.gamma, distance='rbf')
        eigvec_hat, eigval_hat = ncut_wrapper(feats_compressed, self.n_eig, gamma=self.gamma, distance='rbf')

        total_loss = 0
        if self.eigvec_loss > 0:
            eigvec_loss = flag_space_loss(eigvec_gt, eigvec_hat, n_eig=self.n_eig)
            self.log("loss/eigvec", eigvec_loss, prog_bar=True)
            total_loss += eigvec_loss * self.eigvec_loss
            self.loss_history['eigvec'].append(eigvec_loss.item())

        if self.recon_loss > 0:
            recon_loss = F.smooth_l1_loss(output_feats, feats_uncompressed)
            self.log("loss/recon", recon_loss, prog_bar=True)
            total_loss += recon_loss * self.recon_loss
            self.loss_history['recon'].append(recon_loss.item())

        if self.riemann_curvature_loss > 0:
            riemann_curvature_loss = compute_riemann_curvature_loss(feats_compressed)
            self.log("loss/riemann_curvature", riemann_curvature_loss, prog_bar=True)
            total_loss += riemann_curvature_loss * self.riemann_curvature_loss

        if self.axis_align_loss > 0:
            axis_align_loss = compute_axis_align_loss(feats_compressed)
            self.log("loss/axis_align", axis_align_loss, prog_bar=True)
            total_loss += axis_align_loss * self.axis_align_loss

        if self.repulsion_loss > 0:
            repulsion_loss = compute_repulsion_loss(feats_compressed)
            self.log("loss/repulsion", repulsion_loss, prog_bar=True)
            total_loss += repulsion_loss * self.repulsion_loss

        if self.boundary_loss > 0:
            boundary_loss = compute_boundary_loss(feats_compressed)
            self.log("loss/boundary", boundary_loss, prog_bar=True)
            total_loss += boundary_loss * self.boundary_loss

        loss = total_loss
        self.log("loss/total", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters(), lr=self.lr)
        return optimizer


def train_mspace_model(compress_feats, uncompress_feats, training_steps=1000, grad_clip_val=1.0, 
                    batch_size=1000, devices=[0], return_trainer=False, **model_kwargs):
    compress_feats = torch.tensor(compress_feats).float().cpu()
    uncompress_feats = torch.tensor(uncompress_feats).float().cpu()
    l, c = compress_feats.shape

    model = CompressionModel(c, **model_kwargs)
    

    is_cuda = torch.cuda.is_available()
    trainer = pl.Trainer(max_steps=training_steps,
                         gradient_clip_val=grad_clip_val,
                         accelerator="gpu" if is_cuda else "cpu", 
                         devices=devices if is_cuda else None,
                         enable_checkpointing=False,
                         enable_progress_bar=False,
                         enable_model_summary=False,
                         logger=False,
                         )
    dataset = TensorDataset(compress_feats, uncompress_feats)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    trainer.fit(model, dataloader)

    if return_trainer:
        return model, trainer

    return model

def mspace_viz_transform(feats, **kwargs):
    model, trainer = train_mspace_model(feats, feats, return_trainer=True, **kwargs)

    batch_size = kwargs.get('batch_size', 1000)
    test_loader = torch.utils.data.DataLoader(TensorDataset(feats), batch_size=batch_size, shuffle=False)
    compressed = trainer.predict(model, test_loader)
    compressed = torch.cat(compressed, dim=0)
    return compressed
