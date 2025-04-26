import logging
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import TensorDataset

# disable lightning logs
logging.getLogger('lightning').setLevel(0)
logging.getLogger("pytorch_lightning").setLevel(0)
class IgnorePLFilter(logging.Filter):
    def filter(self, record):
        keywords = ['available:', 'CUDA', 'LOCAL_RANK:']
        return not any(keyword in record.getMessage() for keyword in keywords)
logging.getLogger('pytorch_lightning.utilities.rank_zero').addFilter(IgnorePLFilter())
logging.getLogger('pytorch_lightning.accelerators.cuda').addFilter(IgnorePLFilter())


from .ncut_pytorch import affinity_from_features, ncut
from .affinity_gamma import find_gamma_by_degree_after_fps
from .math_utils import compute_riemann_curvature_loss, compute_boundary_loss, compute_repulsion_loss, compute_axis_align_loss, compute_attraction_loss, find_elbow

def _kway_ncut_loss(eigvec_gt, eigvec_hat, n_eig):
    _eigvec_gt = eigvec_gt[:, :n_eig]
    _eigvec_hat = eigvec_hat[:, :n_eig]
    loss = F.smooth_l1_loss(_eigvec_gt @ _eigvec_gt.T, _eigvec_hat @ _eigvec_hat.T)
    return loss

def flag_space_loss(eigvec_gt, eigvec_hat, n_eig, start=2, step_mult=2):
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

def filter_closeby_eigval(eigvec, eigval, threshold=1e-12):
    # filter out eigvals that are too close to each other
    # so the gradient is more stable
    eigval_diff = torch.diff(eigval).abs()
    keep_idx = torch.where(eigval_diff > threshold)[0]
    return eigvec[:, keep_idx], eigval[keep_idx]

def ncut_wrapper(features, n_eig, distance='rbf', gamma=0.5):
    A = affinity_from_features(features, distance=distance, gamma=gamma)
    eigvec, eigval = ncut(A, n_eig)
    eigvec, eigval = filter_closeby_eigval(eigvec, eigval)
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
    N_ITER_PER_STEP = 10
    
    def __init__(self, in_dim, mood_dim=2, n_eig=None, n_elbow=3,
                 n_layer=4, latent_dim=256, 
                 eigvec_loss=1, recon_loss=1, 
                 riemann_curvature_loss=0., axis_align_loss=0, 
                 repulsion_loss=0.1, attraction_loss=0., 
                 boundary_loss=0.01, zero_center_loss=0.,
                 lr=0.001, progress_bar=True, training_steps=3000,
                 degree=[0.05, 0.1, 0.2, 0.5]):
        super().__init__()
        
        self.compress = MLP(in_dim, mood_dim, n_layer, latent_dim)
        self.uncompress = MLP(mood_dim, in_dim, n_layer, latent_dim)
                
        self.loss_history = defaultdict(list)

        self.eigvec_loss = eigvec_loss
        self.recon_loss = recon_loss
        self.riemann_curvature_loss = riemann_curvature_loss
        self.axis_align_loss = axis_align_loss
        self.repulsion_loss = repulsion_loss
        self.attraction_loss = attraction_loss
        self.boundary_loss = boundary_loss
        self.zero_center_loss = zero_center_loss
        self.lr = lr
        self.degree = degree if isinstance(degree, list) else [degree]
        self.gamma = [None] * len(self.degree)
        self.n_eig = n_eig if isinstance(n_eig, list) else [n_eig] * len(degree)
        self.n_elbow = n_elbow

        self.progress_bar = progress_bar
        self.training_steps = training_steps
        if self.progress_bar:
            self.progress_bar = tqdm(total=training_steps, desc="M-space training")

        self.automatic_optimization = False


    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        return self.compress(x)

    def training_step(self, batch):
        if self.progress_bar and self.trainer.global_step % 10 == 0 and self.trainer.global_step != 0:
            self.progress_bar.update(10)
        if self.progress_bar and self.trainer.global_step >= self.training_steps - 10:
            self.progress_bar.update(10)
            self.progress_bar.close()

        input_feats, output_feats = batch
        
        stored_eigvec_gt = {}
        with torch.no_grad():
            # Compute eigvec_gt only once for each set of iterations
            if self.eigvec_loss != 0:
                    
                for i, degree, gamma, n_eig in zip(range(len(self.degree)), 
                                                self.degree, self.gamma, self.n_eig):
                    if gamma is None:
                        gamma = find_gamma_by_degree_after_fps(input_feats, degree, distance='rbf')
                        self.gamma[i] = gamma
                    if n_eig is None:
                        eigvec_gt, eigval_gt = ncut_wrapper(input_feats, input_feats.shape[0]//2, gamma=gamma, distance='rbf')
                        n_eig = find_elbow(eigval_gt, n_elbows=self.n_elbow)[-1]
                        self.n_eig[i] = n_eig
                    
                    # Compute and store eigvec_gt
                    key = f"{i}_{degree}_{gamma}_{n_eig}"
                    eigvec_gt, eigval_gt = ncut_wrapper(input_feats, n_eig, gamma=gamma, distance='rbf')
                    stored_eigvec_gt[key] = eigvec_gt
        
        # Run the same batch 10 times, updating parameters after each iteration
        for iteration in range(self.N_ITER_PER_STEP):
            feats_compressed = self.compress(input_feats)
            feats_uncompressed = self.uncompress(feats_compressed)
            
            total_loss = 0
            if self.eigvec_loss != 0:
                for i, degree, gamma, n_eig in zip(range(len(self.degree)), 
                                                  self.degree, self.gamma, self.n_eig):
                    # Reuse the stored eigvec_gt
                    key = f"{i}_{degree}_{gamma}_{n_eig}"
                    eigvec_gt = stored_eigvec_gt[key]
                    
                    # Compute eigvec_hat for each iteration
                    eigvec_hat, eigval_hat = ncut_wrapper(feats_compressed, n_eig, gamma=gamma, distance='rbf')
                    eigvec_loss = flag_space_loss(eigvec_gt, eigvec_hat, n_eig=n_eig)
                    self.log(f"loss/eigvec_d{self.degree[i]:.2f}", eigvec_loss, prog_bar=True)
                    total_loss += eigvec_loss * self.eigvec_loss
                    self.loss_history[f'eigvec_d{self.degree[i]:.2f}'].append(eigvec_loss.item())

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

            if self.attraction_loss > 0:
                attraction_loss = compute_attraction_loss(feats_compressed)
                self.log("loss/attraction", attraction_loss, prog_bar=True)
                total_loss += attraction_loss * self.attraction_loss

            if self.boundary_loss > 0:
                boundary_loss = compute_boundary_loss(feats_compressed)
                self.log("loss/boundary", boundary_loss, prog_bar=True)
                total_loss += boundary_loss * self.boundary_loss

            if self.zero_center_loss > 0:
                zero_center_loss = feats_compressed.abs().mean()
                self.log("loss/zero_center", zero_center_loss, prog_bar=True)
                total_loss += zero_center_loss * self.zero_center_loss

            # Log the loss for this iteration
            self.log(f"loss/total", total_loss, prog_bar=True)
            
            self.manual_backward(total_loss)
            self.clip_gradients(self.optimizers(), gradient_clip_val=0.1)
            self.optimizers().step()
            self.optimizers().zero_grad()
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters(), lr=self.lr)
        return optimizer


def train_mspace_model(compress_feats, uncompress_feats, training_steps=3000,
                    batch_size=1000, devices=[0], return_trainer=False, progress_bar=True, **model_kwargs):
    compress_feats = torch.tensor(compress_feats).float().cpu()
    uncompress_feats = torch.tensor(uncompress_feats).float().cpu()
    l, c = compress_feats.shape

    training_steps = training_steps // CompressionModel.N_ITER_PER_STEP

    model = CompressionModel(c, training_steps=training_steps, progress_bar=progress_bar, **model_kwargs)
    
    dataset = TensorDataset(compress_feats, uncompress_feats)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    is_cuda = torch.cuda.is_available()
    trainer = pl.Trainer(limit_train_batches=training_steps,
                         max_steps=training_steps,
                         accelerator="gpu" if is_cuda else "cpu", 
                         devices=devices if is_cuda else None,
                         enable_checkpointing=False,
                         enable_progress_bar=False,
                         enable_model_summary=False,
                         logger=False,
                         )
    trainer.fit(model, dataloader)

    if return_trainer:
        return model, trainer

    return model

def mspace_viz_transform(feats, return_model=False, **kwargs):
    model, trainer = train_mspace_model(feats, feats, return_trainer=True, **kwargs)

    batch_size = kwargs.get('batch_size', 1000)
    test_loader = torch.utils.data.DataLoader(TensorDataset(feats), batch_size=batch_size, shuffle=False)
    compressed = trainer.predict(model, test_loader)
    compressed = torch.cat(compressed, dim=0)
    if return_model:
        return compressed, model
    return compressed
