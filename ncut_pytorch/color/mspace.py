__all__ = ["train_mspace_model", "mspace_viz_transform"]

import logging
from collections import defaultdict
from functools import partial, wraps
import warnings
import pytorch_lightning as pl
from typing import List

import numpy as np  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from tqdm import tqdm

from ncut_pytorch.utils.device import auto_device
from ncut_pytorch import ncut_fn
from ncut_pytorch.utils.math import normalize_affinity, rbf_affinity, cosine_affinity
from ncut_pytorch.utils.grad import eigvec_outer_product
from ncut_pytorch.utils.sigma import find_sigma_by_degree


def get_eigvec_outer_product(features: torch.Tensor, n_eig_list: List[int]) -> torch.Tensor:
    sigma = find_sigma_by_degree(features)
    W = rbf_affinity(features, sigma=sigma)
    A = normalize_affinity(W)
    eigval_masks = torch.zeros(len(n_eig_list), A.shape[0], dtype=torch.bool)
    for i, n_eig in enumerate(n_eig_list):
        eigval_masks[i, :n_eig] = True
    P = eigvec_outer_product(A, eigval_masks)
    return P



class MovingMinMax(nn.Module):
    def __init__(self, dim, beta=0.1):
        super(MovingMinMax, self).__init__()
        self.beta = beta
        self.min_val = nn.Parameter(torch.zeros(dim))
        self.min_val.requires_grad = False
        self.max_val = nn.Parameter(torch.ones(dim))
        self.max_val.requires_grad = False
        

    def forward(self, x):
        self.min_val.data = self.min_val * self.beta + x.min(dim=0).values * (1 - self.beta)
        self.max_val.data = self.max_val * self.beta + x.max(dim=0).values * (1 - self.beta)

        x = (x - self.min_val) / (self.max_val - self.min_val)
        return x

class MLP(nn.Module):
    activation_map = {
        'leaky_relu': partial(nn.LeakyReLU, negative_slope=0.2),
        'relu': partial(nn.ReLU),
        'elu': partial(nn.ELU),
        'gelu': partial(nn.GELU),
        'tanh': partial(nn.Tanh),
        'sigmoid': partial(nn.Sigmoid),
        'identity': partial(nn.Identity),
    }
    def __init__(self, in_dim, out_dim, n_layer=2, latent_dim=4096, activation='gelu', final_activation='identity'):
        super().__init__()
        self.mlp = nn.Sequential(
            # MovingMinMaxAvg(in_dim),
            nn.Linear(in_dim, latent_dim),
            self.activation_map[activation](),
            *[nn.Sequential(nn.Linear(latent_dim, latent_dim), self.activation_map[activation]()) for _ in range(n_layer)],
            nn.Linear(latent_dim, out_dim),
            self.activation_map[final_activation](),
        )
    
    def forward(self, x):
        return self.mlp(x)


class MspaceAutoEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, z_dim, n_layer=4, latent_dim=256, encoder_activation='gelu', decoder_activation='gelu', final_activation='identity'):
        super().__init__()
        self.encoder = MLP(in_dim, z_dim, n_layer, latent_dim, encoder_activation, final_activation)
        self.decoder = nn.Sequential(
            MovingMinMax(z_dim),
            MLP(z_dim, out_dim, n_layer, latent_dim, decoder_activation)
        )
    

class TrainDecoder(pl.LightningModule):
    N_ITER_PER_STEP = 1

    # train the decoder after the encoder is trained, freeze the encoder
    def __init__(self, mspace_ae, decoder_lr=0.001, progress_bar=True, training_steps=1000, 
                 decoder_repulsion_loss=1e-1, decoder_zero_center_loss=1e-2,
                 log_grad_norm=False, **kwargs):
        super().__init__()
        self.mspace_ae = mspace_ae
        self.lr = decoder_lr
        self.progress_bar = progress_bar
        self.training_steps = training_steps
        if self.progress_bar:
            self.progress_bar = tqdm(total=training_steps, desc="[M-space decoder]")
        self.decoder_repulsion_loss = decoder_repulsion_loss
        self.decoder_zero_center_loss = decoder_zero_center_loss
        self.log_grad_norm = log_grad_norm

        self.loss_history = defaultdict(list)

        self.automatic_optimization = False

    def _log_loss(self, loss, name, log_grad_norm=False):
        if self.logger is None:
            return
        self.logger.log_metrics({f'decoder/loss/{name}': loss.item()}, step=self.global_step)
        self.loss_history[f"decoder/loss/{name}"].append(loss.item())
        if log_grad_norm:
            grad_norm = 0
            self.manual_backward(loss, retain_graph=True)
            for param in self.mspace_ae.decoder.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            self.logger.log_metrics({f'decoder/grad/{name}': grad_norm}, step=self.global_step)
            self.loss_history[f"decoder/grad/{name}"].append(grad_norm)
            self.optimizers().zero_grad()

    def training_step(self, batch):
        if self.progress_bar and self.trainer.global_step % 10 == 0 and self.trainer.global_step != 0:
            self.progress_bar.update(10)
            if hasattr(self, 'loss_history') and 'decoder/loss/recon' in self.loss_history and self.loss_history['decoder/loss/recon']:
                recon_loss = self.loss_history['decoder/loss/recon'][-1]
                self.progress_bar.set_description(f"[M-space decoder] recon  loss: {recon_loss:.2e}")
        if self.progress_bar and self.trainer.global_step % 10 == 0 and self.trainer.global_step >= self.training_steps - 10:
            self.progress_bar.update(10)
            self.progress_bar.close()

        input_feats, output_feats = batch
        with torch.no_grad():
            feats_compressed = self.mspace_ae.encoder(input_feats)
            dim_mins = feats_compressed.min(0).values
            dim_maxs = feats_compressed.max(0).values
            compressed_dim = feats_compressed.shape[1]


        for iteration in range(self.N_ITER_PER_STEP):

            total_loss = 0

            feats_uncompressed = self.mspace_ae.decoder(feats_compressed)
            recon_loss = F.mse_loss(output_feats, feats_uncompressed)
            self._log_loss(recon_loss, "recon", self.log_grad_norm and iteration == 0)
            total_loss = recon_loss

            if self.decoder_repulsion_loss > 0:
                ## repulsion regularization loss
                grid_size = 16
                radius_factor = 0.2
                # randomly shift the grid, to have better coverage
                dim_mins -= 0.25 * (dim_maxs - dim_mins) * torch.rand_like(dim_mins)
                dim_maxs += 0.25 * (dim_maxs - dim_mins) * torch.rand_like(dim_maxs)
                radius = (dim_maxs - dim_mins) / grid_size
                radius = (radius ** 2).sum() ** 0.5 * radius_factor
                
                # Create a grid of points in the compressed space
                if compressed_dim == 2:
                    # For 2D, use meshgrid for efficiency
                    grid1 = torch.linspace(dim_mins[0], dim_maxs[0], grid_size, device=feats_compressed.device)
                    grid2 = torch.linspace(dim_mins[1], dim_maxs[1], grid_size, device=feats_compressed.device)
                    X, Y = torch.meshgrid(grid1, grid2, indexing='ij')
                    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
                else:
                    # For higher dimensions, create a random sampling of points
                    num_samples = 256
                    grid_points = torch.rand(num_samples, compressed_dim, device=feats_compressed.device)
                    grid_points = grid_points * (dim_maxs - dim_mins) + dim_mins
                
                # Decompress the grid points
                grid_decompressed = self.mspace_ae.decoder(grid_points)
                
                # Find nearest neighbors in original data
                dist = torch.cdist(grid_decompressed, output_feats)
                nearest_dists = dist.min(dim=1).values
                repulsion = 1.0 / (nearest_dists + 0.01)  # the shift is to avoid big gradient
                
                ## filter out points that are too close to the existing data points
                grid_to_data_distances = torch.cdist(grid_points, feats_compressed)
                # Check if any data point is within the radius for each grid point
                has_data_nearby = (grid_to_data_distances < radius).any(dim=1)

                # Calculate regularization loss only for points without nearby data
                if (~has_data_nearby).any():
                    reg_loss = repulsion[~has_data_nearby].mean()
                else:
                    reg_loss = torch.tensor(0.0, device=repulsion.device)

                reg_loss = reg_loss * self.decoder_repulsion_loss
                self._log_loss(reg_loss, "reg", self.log_grad_norm and iteration == 0)
                total_loss = total_loss + reg_loss

                if self.decoder_zero_center_loss > 0:
                    ## zero center loss
                    zcenter = output_feats.mean(0)
                    zcenter_points = grid_decompressed[~has_data_nearby]
                    zcenter_loss = (zcenter_points - zcenter).abs().mean()
                    zcenter_loss = zcenter_loss * self.decoder_zero_center_loss
                    self._log_loss(zcenter_loss, "zcenter", self.log_grad_norm and iteration == 0)
                    total_loss = total_loss + zcenter_loss

            
            optimizer = self.optimizers()
            optimizer.zero_grad()
            self.manual_backward(total_loss)
            self.clip_gradients(optimizer, gradient_clip_val=0.1)
            optimizer.step()
        
        return total_loss

    def configure_optimizers(self):
        params = list(self.mspace_ae.decoder.parameters())
        params = [p for p in params if p.requires_grad]
        optimizer = torch.optim.NAdam(params, lr=self.lr)
        return optimizer


class TrainEncoder(pl.LightningModule):
    N_ITER_PER_STEP = 10
    
    def __init__(self, in_dim, out_dim, z_dim=2, n_eig_list=[4, 16, 64],
                 n_layer=4, latent_dim=256, 
                 flag_loss_mode='z_eigvec',  # 'z_eigvec' or 'z'
                 flag_loss=1.0, recon_loss=1e-3, 
                 repulsion_loss=1e-3, zero_center_loss=1e-3,
                 lr=1e-3, progress_bar=True, training_steps=1000,
                 encoder_activation='gelu', decoder_activation='gelu', 
                 final_activation='identity',
                 log_grad_norm=False, 
                 **kwargs):
        super().__init__()
        
        self.mspace_ae = MspaceAutoEncoder(in_dim, out_dim, z_dim, n_layer, latent_dim, encoder_activation, decoder_activation, final_activation)
                
        self.loss_history = defaultdict(list)

        self.flag_loss_mode = flag_loss_mode
        self.flag_loss = flag_loss
        self.recon_loss = recon_loss
        self.repulsion_loss = repulsion_loss
        self.zero_center_loss = zero_center_loss
        self.lr = lr
        self.n_eig_list = n_eig_list

        self.progress_bar = progress_bar
        self.training_steps = training_steps
        if self.progress_bar:
            self.progress_bar = tqdm(total=training_steps, desc="[M-space encoder]")

        self.automatic_optimization = False
        self.log_grad_norm = log_grad_norm

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        return self.mspace_ae.encoder(x)
    

    def _log_loss(self, loss, name, log_grad_norm=False, iteration=0):
        # if self.logger is None:
            # return
        # self.logger.log_metrics({f'loss/{name}': loss.item()}, step=self.global_step)
        # self.loss_history[f"loss/{name}"].append(loss.item())
        if log_grad_norm:
            grad_norm = 0
            self.manual_backward(loss, retain_graph=True)
            for param in self.mspace_ae.encoder.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            # self.logger.log_metrics({f'grad/{name}': grad_norm}, step=self.global_step)
            # self.loss_history[f"grad/{name}"].append(grad_norm)
            if iteration % 10 == 0:
                print(f"loss/{name} grad_norm: {grad_norm:.2e}, iteration: {iteration}")
            self.optimizers().zero_grad()

    def training_step(self, batch):
        if self.progress_bar and self.trainer.global_step % 10 == 0 and self.trainer.global_step != 0:
            self.progress_bar.update(10)
        if self.progress_bar and self.trainer.global_step % 10 == 0 and self.trainer.global_step >= self.training_steps - 10:
            self.progress_bar.update(10)
            self.progress_bar.close()

        input_feats, output_feats = batch
        
        P_gt = {}
        # Compute eigvec_gt only once for each set of iterations
        if self.flag_loss != 0:
            P_gt = get_eigvec_outer_product(input_feats, self.n_eig_list)

        
        # Run the same batch 10 times, updating parameters after each iteration
        for iteration in range(self.N_ITER_PER_STEP):

            feats_compressed = self.mspace_ae.encoder(input_feats)

            log_grad_norm = iteration == 0 and self.log_grad_norm
            
            total_loss = 0
            if self.flag_loss != 0:
                if self.flag_loss_mode == 'z_eigvec':
                    P_hat = get_eigvec_outer_product(feats_compressed, self.n_eig_list)
                    flag_loss = F.mse_loss(P_gt, P_hat)
                elif self.flag_loss_mode == 'z':
                    P_hat = feats_compressed @ feats_compressed.T
                    flag_loss = F.mse_loss(P_gt.mean(0), P_hat)
                flag_loss = flag_loss * self.flag_loss
                total_loss += flag_loss
                self._log_loss(flag_loss, "flag_loss", log_grad_norm=log_grad_norm, iteration=self.trainer.global_step)

            if self.recon_loss > 0:
                feats_uncompressed = self.mspace_ae.decoder(feats_compressed)
                recon_loss = F.mse_loss(output_feats, feats_uncompressed)
                recon_loss = recon_loss * self.recon_loss
                total_loss += recon_loss
                self._log_loss(recon_loss, "recon_loss", log_grad_norm=log_grad_norm, iteration=self.trainer.global_step)

            if self.zero_center_loss > 0:
                # zero_center_loss = feats_compressed.abs().mean()
                zero_center_loss = (feats_compressed ** 2).mean()
                zero_center_loss = zero_center_loss * self.zero_center_loss
                total_loss += zero_center_loss
                self._log_loss(zero_center_loss, "zero_center_loss", log_grad_norm=log_grad_norm, iteration=self.trainer.global_step)

            if self.repulsion_loss > 0:
                repulsion_loss = compute_repulsion_loss(feats_compressed)
                repulsion_loss = repulsion_loss * self.repulsion_loss
                total_loss += repulsion_loss
                self._log_loss(repulsion_loss, "repulsion_loss", log_grad_norm=log_grad_norm, iteration=self.trainer.global_step)

            # Log the loss for this iteration
            if self.logger is not None:
                self.logger.log_metrics({'loss/total': total_loss.item()}, step=self.global_step)
            self.loss_history['loss/total'].append(total_loss.item())
            
            optimizer = self.optimizers()
            optimizer.zero_grad()
            self.manual_backward(total_loss)
            self.clip_gradients(optimizer, gradient_clip_val=0.1)
            optimizer.step()

        return total_loss
    
    def configure_optimizers(self):
        params = list(self.mspace_ae.parameters())
        params = [p for p in params if p.requires_grad]
        optimizer = torch.optim.NAdam(params, lr=self.lr)
        return optimizer


def compute_repulsion_loss(
    points: torch.Tensor,      # [N, D]
) -> torch.Tensor:
    """Computes repulsion loss between points to prevent collapse."""
    dist_matrix = torch.cdist(points, points)
    mask = torch.eye(points.shape[0], device=points.device).bool()
    dist_matrix = dist_matrix + mask * 1e10
    nearest_dists, _ = torch.min(dist_matrix, dim=1)
    repulsion = 1.0 / (nearest_dists + 1)
    return torch.mean(repulsion)


def suppress_lightning_logs(func):
    """Temporarily suppresses noisy PyTorch Lightning startup logs."""
    class IgnorePLFilter(logging.Filter):
        def filter(self, record):
            keywords = ['available:', 'CUDA', 'LOCAL_RANK:']
            return not any(keyword in record.getMessage() for keyword in keywords)

    @wraps(func)
    def wrapper(*args, **kwargs):
        original_lightning_level = logging.getLogger('lightning').level
        original_pytorch_lightning_level = logging.getLogger("pytorch_lightning").level
        logging.getLogger('lightning').setLevel(0)
        logging.getLogger("pytorch_lightning").setLevel(0)
        pl_filter = IgnorePLFilter()
        logger_rank_zero = logging.getLogger('pytorch_lightning.utilities.rank_zero')
        logger_cuda = logging.getLogger('pytorch_lightning.accelerators.cuda')
        logger_rank_zero.addFilter(pl_filter)
        logger_cuda.addFilter(pl_filter)
        try:
            return func(*args, **kwargs)
        finally:
            logging.getLogger('lightning').setLevel(original_lightning_level)
            logging.getLogger("pytorch_lightning").setLevel(original_pytorch_lightning_level)
            logger_rank_zero.removeFilter(pl_filter)
            logger_cuda.removeFilter(pl_filter)

    return wrapper


@suppress_lightning_logs
def train_mspace_model(compress_feats, uncompress_feats, encoder_training_steps=3000, decoder_training_steps=3000,
                    batch_size=1000, return_trainer=False, progress_bar=False,
                    logger=False, use_wandb=False, **model_kwargs):
    # check args
    valid_keys = ['z_dim', 'flag_loss', 'flag_loss_mode', 'recon_loss', 'repulsion_loss', 'zero_center_loss', 'n_eig_list', 'n_layer', 'latent_dim', 'lr']
    for key in model_kwargs.keys():
        if key not in valid_keys:
            raise ValueError(f"Invalid argument key: {key}. Valid keys: {valid_keys}")

    compress_feats = compress_feats.float().cpu()
    uncompress_feats = uncompress_feats.float().cpu()
    l, c_in = compress_feats.shape
    c_out = uncompress_feats.shape[1]

    model = TrainEncoder(c_in, c_out, training_steps=encoder_training_steps, progress_bar=progress_bar, **model_kwargs)
    
    dataset = TensorDataset(compress_feats, uncompress_feats)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    trainer_args = {
        'accelerator': auto_device(),
        'devices': 1,
        'enable_checkpointing': False,
        'enable_progress_bar': False,
        'enable_model_summary': False,
    }

    if use_wandb and not logger:
        logger = pl.loggers.WandbLogger(project='mspace', name='mspace')

    # train the autoencoder jointly
    trainer = pl.Trainer(max_steps=encoder_training_steps, logger=logger, **trainer_args)
    trainer.fit(model, dataloader)
    
    mspace_ae = model.mspace_ae

    if decoder_training_steps > 0:
        # train the decoder only
        trainer = pl.Trainer(max_steps=decoder_training_steps, logger=logger, **trainer_args)
        model2 = TrainDecoder(mspace_ae, progress_bar=progress_bar, training_steps=decoder_training_steps, **model_kwargs)
        trainer.fit(model2, dataloader)
        model.mspace_ae = model2.mspace_ae

    if return_trainer:
        result = model, trainer
    else:
        result = model
    
    return result

def mspace_viz_transform(X, return_model=False, **kwargs):
    X = X.float().cpu()
    model, trainer = train_mspace_model(X, X, return_trainer=True, **kwargs)

    batch_size = kwargs.get('batch_size', 1000)
    test_loader = torch.utils.data.DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False, num_workers=0)
    compressed = trainer.predict(model, test_loader)
    compressed = torch.cat(compressed, dim=0)
    if return_model:
        return compressed, model.mspace_ae
    return compressed

