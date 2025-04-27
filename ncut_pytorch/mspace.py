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


from .ncut_pytorch import affinity_from_features, ncut, kway_ncut
from .affinity_gamma import find_gamma_by_degree_after_fps
from .math_utils import compute_riemann_curvature_loss, compute_boundary_loss, compute_repulsion_loss, compute_axis_align_loss, compute_attraction_loss, find_elbow

def _kway_ncut_loss(eigvec_gt, eigvec_hat, n_eig, weight):
    _eigvec_gt = eigvec_gt[:, :n_eig]
    _eigvec_hat = eigvec_hat[:, :n_eig]
    left = _eigvec_gt @ _eigvec_gt.T
    right = _eigvec_hat @ _eigvec_hat.T
    left = left * weight[:, None] * weight[None, :]
    right = right * weight[:, None] * weight[None, :]
    loss = F.l1_loss(left, right)
    return loss

def flag_space_loss(eigvec_gt, eigvec_hat, n_eig, start=2, step_mult=2, weight=None):
    if torch.all(eigvec_gt == 0) or torch.all(eigvec_hat == 0):
        return torch.tensor(0, device=eigvec_gt.device)
    
    if weight is None:
        weight = torch.ones(eigvec_gt.shape[0], device=eigvec_gt.device, dtype=eigvec_gt.dtype)
    
    loss = 0
    n_eig = start // step_mult
    while True:
        n_eig *= step_mult
        loss += _kway_ncut_loss(eigvec_gt, eigvec_hat, n_eig, weight)
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


class MspaceAutoEncoder(nn.Module):
    def __init__(self, in_dim, mood_dim, n_layer=4, latent_dim=256):
        super().__init__()
        self.encoder = MLP(in_dim, mood_dim, n_layer, latent_dim)
        self.decoder = MLP(mood_dim, in_dim, n_layer, latent_dim)
    

class TrainerDecoder(pl.LightningModule):
    # train the decoder after the encoder is trained
    def __init__(self, mspace_ae, lr=0.001, progress_bar=True, training_steps=1000):
        super().__init__()
        self.mspace_ae = mspace_ae
        self.lr = lr
        self.progress_bar = progress_bar
        self.training_steps = training_steps
        if self.progress_bar:
            self.progress_bar = tqdm(total=training_steps//10, desc="M-space decoder training")

    def training_step(self, batch):
        if self.progress_bar and self.trainer.global_step % 10 == 0 and self.trainer.global_step != 0:
            self.progress_bar.update(1)
        if self.progress_bar and self.trainer.global_step >= self.training_steps - 10:
            self.progress_bar.update(1)
            self.progress_bar.close()

        input_feats, output_feats = batch
        with torch.no_grad():
            feats_compressed = self.mspace_ae.encoder(input_feats)
        feats_uncompressed = self.mspace_ae.decoder(feats_compressed)
        # recon_loss = F.smooth_l1_loss(output_feats, feats_uncompressed, beta=0.1)
        recon_loss = F.mse_loss(output_feats, feats_uncompressed)
        self.log("loss/recon_separate", recon_loss, prog_bar=True)
        return recon_loss

    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.mspace_ae.decoder.parameters(), lr=self.lr)
        return optimizer


class TrainerAutoEncoder(pl.LightningModule):
    N_ITER_PER_STEP = 10
    
    def __init__(self, in_dim, mood_dim=2, n_eig=None, n_elbow=3,
                 n_layer=4, latent_dim=256, 
                 eigvec_loss=100, recon_loss=0, 
                 riemann_curvature_loss=0., axis_align_loss=0, 
                 repulsion_loss=0.1, attraction_loss=0., 
                 boundary_loss=0., zero_center_loss=0.01,
                 lr=0.001, progress_bar=True, training_steps=3000,
                 degree=[0.05, 0.1, 0.2, 0.5], kway_weight_gamma=5.0,
                 log_grad_norm=False):
        super().__init__()
        
        self.mspace_ae = MspaceAutoEncoder(in_dim, mood_dim, n_layer, latent_dim)
                
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
        self.kway_weight_gamma = kway_weight_gamma

        self.progress_bar = progress_bar
        self.training_steps = training_steps
        if self.progress_bar:
            self.progress_bar = tqdm(total=training_steps, desc="M-space encoder training")

        self.automatic_optimization = False
        self.log_grad_norm = log_grad_norm

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        return self.mspace_ae.encoder(x)
    

    def _log_loss(self, loss, name, log_grad_norm=False):
        self.logger.log_metrics({f'loss/{name}': loss.item()}, step=self.global_step)
        self.loss_history[f"loss/{name}"].append(loss.item())
        if log_grad_norm:
            grad_norm = 0
            self.manual_backward(loss, retain_graph=True)
            for param in self.mspace_ae.encoder.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            self.logger.log_metrics({f'grad/{name}': grad_norm}, step=self.global_step)
            self.loss_history[f"grad/{name}"].append(grad_norm)
            self.optimizers().zero_grad()

    def training_step(self, batch):
        if self.progress_bar and self.trainer.global_step % 10 == 0 and self.trainer.global_step != 0:
            self.progress_bar.update(10)
        if self.progress_bar and self.trainer.global_step >= self.training_steps - 10:
            self.progress_bar.update(10)
            self.progress_bar.close()

        input_feats, output_feats = batch
        
        stored_eigvec_gt = {}
        stored_eigvec_weight = {}
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

                    # Compute weight for each node, using kway ncut
                    eigvec_gt_kway = kway_ncut(eigvec_gt, return_continuous=True)
                    weight = eigvec_gt_kway.max(1).values.flatten()
                    weight = weight ** self.kway_weight_gamma
                    stored_eigvec_weight[key] = weight
        
        # Run the same batch 10 times, updating parameters after each iteration
        for iteration in range(self.N_ITER_PER_STEP):
            feats_compressed = self.mspace_ae.encoder(input_feats)

            log_grad_norm = iteration == 0 and self.log_grad_norm
            
            total_loss = 0
            if self.eigvec_loss != 0:
                for i, degree, gamma, n_eig in zip(range(len(self.degree)), 
                                                  self.degree, self.gamma, self.n_eig):
                    # Reuse the stored eigvec_gt
                    key = f"{i}_{degree}_{gamma}_{n_eig}"
                    eigvec_gt = stored_eigvec_gt[key]
                    weight = stored_eigvec_weight[key]

                    # Compute eigvec_hat for each iteration
                    eigvec_hat, eigval_hat = ncut_wrapper(feats_compressed, n_eig, gamma=gamma, distance='rbf')
                    eigvec_loss = flag_space_loss(eigvec_gt, eigvec_hat, n_eig=n_eig, weight=weight)
                    eigvec_loss = eigvec_loss * self.eigvec_loss
                    total_loss += eigvec_loss

                    _log_grad_norm = log_grad_norm and i == 0
                    self._log_loss(eigvec_loss, f"eigvec_d{self.degree[i]:.2f}", log_grad_norm=_log_grad_norm)


            if self.recon_loss > 0:
                feats_uncompressed = self.mspace_ae.decoder(feats_compressed)
                recon_loss = F.smooth_l1_loss(output_feats, feats_uncompressed, beta=0.1)
                recon_loss = recon_loss * self.recon_loss
                total_loss += recon_loss
                self._log_loss(recon_loss, "recon", log_grad_norm=log_grad_norm)


            if self.riemann_curvature_loss > 0:
                riemann_curvature_loss = compute_riemann_curvature_loss(feats_compressed)
                riemann_curvature_loss = riemann_curvature_loss * self.riemann_curvature_loss
                total_loss += riemann_curvature_loss
                self._log_loss(riemann_curvature_loss, "riemann_curvature", log_grad_norm=log_grad_norm)

            if self.axis_align_loss > 0:
                axis_align_loss = compute_axis_align_loss(feats_compressed)
                axis_align_loss = axis_align_loss * self.axis_align_loss
                total_loss += axis_align_loss
                self._log_loss(axis_align_loss, "axis_align", log_grad_norm=log_grad_norm)

            if self.repulsion_loss > 0:
                repulsion_loss = compute_repulsion_loss(feats_compressed)
                repulsion_loss = repulsion_loss * self.repulsion_loss
                total_loss += repulsion_loss
                self._log_loss(repulsion_loss, "repulsion", log_grad_norm=log_grad_norm)

            if self.attraction_loss > 0:
                attraction_loss = compute_attraction_loss(feats_compressed)
                attraction_loss = attraction_loss * self.attraction_loss
                total_loss += attraction_loss
                self._log_loss(attraction_loss, "attraction", log_grad_norm=log_grad_norm)

            if self.boundary_loss > 0:
                boundary_loss = compute_boundary_loss(feats_compressed)
                boundary_loss = boundary_loss * self.boundary_loss
                total_loss += boundary_loss
                self._log_loss(boundary_loss, "boundary", log_grad_norm=log_grad_norm)

            if self.zero_center_loss > 0:
                zero_center_loss = feats_compressed.abs().mean()
                zero_center_loss = zero_center_loss * self.zero_center_loss
                total_loss += zero_center_loss
                self._log_loss(zero_center_loss, "zero_center", log_grad_norm=log_grad_norm)

            # Log the loss for this iteration
            self.logger.log_metrics({'loss/total': total_loss.item()}, step=self.global_step)
            
            self.manual_backward(total_loss)
            self.clip_gradients(self.optimizers(), gradient_clip_val=0.1)
            self.optimizers().step()
            self.optimizers().zero_grad()
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters(), lr=self.lr)
        return optimizer


def train_mspace_model(compress_feats, uncompress_feats, training_steps=3000, decoder_training_steps=1000, decoder_lr=0.001,
                    batch_size=1000, devices=[0], return_trainer=False, progress_bar=True,
                    logger=None, use_wandb=False, **model_kwargs):
    compress_feats = torch.tensor(compress_feats).float().cpu()
    uncompress_feats = torch.tensor(uncompress_feats).float().cpu()
    l, c = compress_feats.shape

    training_steps = training_steps // TrainerAutoEncoder.N_ITER_PER_STEP

    model = TrainerAutoEncoder(c, training_steps=training_steps, progress_bar=progress_bar, **model_kwargs)
    
    dataset = TensorDataset(compress_feats, uncompress_feats)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    is_cuda = torch.cuda.is_available()

    trainer_args = {
        'accelerator': "gpu" if is_cuda else "cpu", 
        'devices': devices if is_cuda else None,
        'enable_checkpointing': False,
        'enable_progress_bar': False,
        'enable_model_summary': False,
    }

    if use_wandb and logger is None:
        logger = pl.loggers.WandbLogger(project='mspace', name='mspace')

    # train the autoencoder jointly
    trainer = pl.Trainer(max_steps=training_steps, logger=logger, **trainer_args)
    trainer.fit(model, dataloader)
    
    mspace_ae = model.mspace_ae

    if decoder_training_steps > 0:
        # train the decoder only
        trainer = pl.Trainer(max_steps=decoder_training_steps, logger=logger, **trainer_args)
        model2 = TrainerDecoder(mspace_ae, lr=decoder_lr, progress_bar=progress_bar, training_steps=decoder_training_steps)
        trainer.fit(model2, dataloader)
        model.mspace_ae = model2.mspace_ae

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
        return compressed, model.mspace_ae
    return compressed
