__all__ = ["train_mspace_model", "mspace_viz_transform"]

import logging
from collections import defaultdict
from functools import partial
import warnings

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from tqdm import tqdm

from ncut_pytorch.utils.device import auto_device

# disable lightning logs
logging.getLogger('lightning').setLevel(0)
logging.getLogger("pytorch_lightning").setLevel(0)
class IgnorePLFilter(logging.Filter):
    def filter(self, record):
        keywords = ['available:', 'CUDA', 'LOCAL_RANK:']
        return not any(keyword in record.getMessage() for keyword in keywords)
logging.getLogger('pytorch_lightning.utilities.rank_zero').addFilter(IgnorePLFilter())
logging.getLogger('pytorch_lightning.accelerators.cuda').addFilter(IgnorePLFilter())


from ncut_pytorch.utils.math import rbf_affinity
from ncut_pytorch.ncuts.ncut_nystrom import _plain_ncut
from ncut_pytorch.ncuts.ncut_kway import kway_ncut
from ncut_pytorch.utils.gamma import find_gamma_by_degree_after_fps
from ncut_pytorch.utils.math import compute_riemann_curvature_loss, compute_boundary_loss, compute_repulsion_loss, compute_axis_align_loss, compute_attraction_loss, find_elbow


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

def filter_closeby_eigval(eigvec, eigval, threshold=1e-10):
    # filter out eigvals that are too close to each other
    # so the gradient is more stable
    eigval_diff = torch.diff(eigval).abs()
    keep_idx = torch.where(eigval_diff > threshold)[0]
    return eigvec[:, keep_idx], eigval[keep_idx]

def ncut_wrapper(features, n_eig, gamma=0.5):
    A = rbf_affinity(features, gamma=gamma)
    eigvec, eigval = _plain_ncut(A, n_eig)
    eigvec, eigval = filter_closeby_eigval(eigvec, eigval)
    return eigvec, eigval


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
    def __init__(self, in_dim, out_dim, mood_dim, n_layer=4, latent_dim=256, encoder_activation='gelu', decoder_activation='gelu', final_activation='identity'):
        super().__init__()
        self.encoder = MLP(in_dim, mood_dim, n_layer, latent_dim, encoder_activation, final_activation)
        self.decoder = nn.Sequential(
            MovingMinMax(mood_dim),
            MLP(mood_dim, out_dim, n_layer, latent_dim, decoder_activation)
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

            
            self.optimizers().zero_grad()
            self.manual_backward(total_loss)
            self.clip_gradients(self.optimizers(), gradient_clip_val=0.1)
            self.optimizers().step()
        
        return total_loss

    def configure_optimizers(self):
        params = list(self.mspace_ae.decoder.parameters())
        params = [p for p in params if p.requires_grad]
        optimizer = torch.optim.NAdam(params, lr=self.lr)
        return optimizer


class TrainEncoder(pl.LightningModule):
    N_ITER_PER_STEP = 10
    
    def __init__(self, in_dim, out_dim, mood_dim=2, n_eig=None, n_elbow=3,
                 n_layer=4, latent_dim=256, 
                 eigvec_loss=100, recon_loss=0, 
                 riemann_curvature_loss=0., axis_align_loss=0, 
                 repulsion_loss=0.1, attraction_loss=0., 
                 boundary_loss=0., zero_center_loss=0.01,
                 lr=0.001, progress_bar=True, training_steps=500,
                 degree=[0.05, 0.1, 0.2, 0.5], kway_weight_gamma=5.0,
                 encoder_activation='gelu', decoder_activation='gelu', 
                 final_activation='identity',
                 log_grad_norm=False, 
                 **kwargs):
        super().__init__()
        
        self.mspace_ae = MspaceAutoEncoder(in_dim, out_dim, mood_dim, n_layer, latent_dim, encoder_activation, decoder_activation, final_activation)
                
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
            self.progress_bar = tqdm(total=training_steps, desc="[M-space encoder]")

        self.automatic_optimization = False
        self.log_grad_norm = log_grad_norm

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        return self.mspace_ae.encoder(x)
    

    def _log_loss(self, loss, name, log_grad_norm=False):
        if self.logger is None:
            return
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
            # Update progress bar description with average eigvec loss
            if hasattr(self, 'loss_history'):
                # Calculate average eigvec loss from all degree values
                eigvec_losses = []
                for i, degree in enumerate(self.degree):
                    loss_key = f"loss/eigvec_d{degree:.2f}"
                    if loss_key in self.loss_history and self.loss_history[loss_key]:
                        eigvec_losses.append(self.loss_history[loss_key][-1])
                if eigvec_losses:
                    avg_eigvec_loss = sum(eigvec_losses) / len(eigvec_losses)
                    self.progress_bar.set_description(f"[M-space encoder] eigvec loss: {avg_eigvec_loss:.2e}")
        if self.progress_bar and self.trainer.global_step % 10 == 0 and self.trainer.global_step >= self.training_steps - 10:
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
                        gamma = find_gamma_by_degree_after_fps(input_feats, degree)
                        self.gamma[i] = gamma
                    if n_eig is None:
                        eigvec_gt, eigval_gt = ncut_wrapper(input_feats, input_feats.shape[0]//2, gamma=gamma)
                        n_eig = find_elbow(eigval_gt, n_elbows=self.n_elbow)[-1]
                        self.n_eig[i] = n_eig
                    
                    # Compute and store eigvec_gt
                    key = f"{i}_{degree}_{gamma}_{n_eig}"
                    eigvec_gt, eigval_gt = ncut_wrapper(input_feats, n_eig, gamma=gamma)
                    stored_eigvec_gt[key] = eigvec_gt

                    # Compute weight for each node, using kway ncut
                    eigvec_gt_kway = kway_ncut(eigvec_gt)
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
                    eigvec_hat, eigval_hat = ncut_wrapper(feats_compressed, n_eig, gamma=gamma)
                    eigvec_loss = flag_space_loss(eigvec_gt, eigvec_hat, n_eig=n_eig, weight=weight)
                    eigvec_loss = eigvec_loss * self.eigvec_loss
                    total_loss += eigvec_loss

                    _log_grad_norm = log_grad_norm and i == 0
                    self._log_loss(eigvec_loss, f"eigvec_d{self.degree[i]}", log_grad_norm=_log_grad_norm)


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
                # zero_center_loss = feats_compressed.abs().mean()
                zero_center_loss = (feats_compressed ** 2).mean()
                zero_center_loss = zero_center_loss * self.zero_center_loss
                total_loss += zero_center_loss
                self._log_loss(zero_center_loss, "zero_center", log_grad_norm=log_grad_norm)

            # Log the loss for this iteration
            if self.logger is not None:
                self.logger.log_metrics({'loss/total': total_loss.item()}, step=self.global_step)
            self.loss_history['loss/total'].append(total_loss.item())
            
            self.manual_backward(total_loss)
            self.clip_gradients(self.optimizers(), gradient_clip_val=0.1)
            self.optimizers().step()
            self.optimizers().zero_grad()

        return total_loss
    
    def configure_optimizers(self):
        params = list(self.mspace_ae.parameters())
        params = [p for p in params if p.requires_grad]
        optimizer = torch.optim.NAdam(params, lr=self.lr)
        return optimizer


# Moving Average Callback
class BestModelsAvgCallback(pl.Callback):
    """
    Callback to store the top models with the lowest total loss and average them at the end.
    """
    def __init__(self, top_k=10):
        super().__init__()
        self.top_k = top_k
        self.best_models = []  # List of tuples (loss, params)
        self.best_loss = float('inf')
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Store the model if it has a lower loss than the current best models."""

        # skip if the global step is on the frist 80% of the training steps
        if trainer.global_step < trainer.max_steps * 0.8:
            return
        
        # Get current model parameters
        current_params = {}
        for name, param in pl_module.named_parameters():
            current_params[name] = param.data.clone()
        
        # Get the current total loss from the outputs
        # The outputs from training_step is the total loss
        current_loss = outputs['loss'].item()
        
        # If we have fewer than top_k models or this is a better model
        if len(self.best_models) < self.top_k or current_loss < self.best_loss:
            # Add to best models
            self.best_models.append((current_loss, current_params))
            
            # Sort by loss (ascending)
            self.best_models.sort(key=lambda x: x[0])
            
            # Keep only top_k models
            if len(self.best_models) > self.top_k:
                self.best_models = self.best_models[:self.top_k]
            
            # Update best loss
            self.best_loss = self.best_models[0][0]
    
    def on_train_end(self, trainer, pl_module):
        """Average the best model parameters and apply them to the model at the end of training."""
        if self.best_models:
            # Initialize averaged parameters
            avg_params = {}
            
            # Get parameter names from the first model
            param_names = self.best_models[0][1].keys()
            
            # Initialize with zeros of the same shape
            for name in param_names:
                avg_params[name] = torch.zeros_like(self.best_models[0][1][name])
            
            # Sum all parameters from best models
            for _, params in self.best_models:
                for name in param_names:
                    avg_params[name] += params[name]
            
            # Divide by the number of models to get the average
            for name in param_names:
                avg_params[name] /= len(self.best_models)
            
            # Apply averaged parameters to the model
            for name, param in pl_module.named_parameters():
                if name in avg_params:
                    param.data.copy_(avg_params[name])
            
            # Log the best loss and average loss
            if pl_module.logger is not None:
                pl_module.logger.log_metrics({
                    'best_loss': self.best_loss,
                    'avg_loss': sum(loss for loss, _ in self.best_models) / len(self.best_models)
                }, step=trainer.global_step)

            self.best_models = []
            self.best_loss = float('inf')
            

def train_mspace_model(compress_feats, uncompress_feats, training_steps=500, decoder_training_steps=1000,
                    batch_size=1000, return_trainer=False, progress_bar=True,
                    logger=False, use_wandb=False, model_avg_window=3, **model_kwargs):
    compress_feats = compress_feats.float().cpu()
    uncompress_feats = uncompress_feats.float().cpu()
    l, c_in = compress_feats.shape
    c_out = uncompress_feats.shape[1]

    model = TrainEncoder(c_in, c_out, training_steps=training_steps, progress_bar=progress_bar, **model_kwargs)
    
    dataset = TensorDataset(compress_feats, uncompress_feats)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    trainer_args = {
        'accelerator': auto_device(),
        'devices': 1,
        'enable_checkpointing': False,
        'enable_progress_bar': False,
        'enable_model_summary': False,
        'callbacks': [BestModelsAvgCallback(top_k=model_avg_window)],
    }

    if use_wandb and not logger:
        logger = pl.loggers.WandbLogger(project='mspace', name='mspace')

    # train the autoencoder jointly
    trainer = pl.Trainer(max_steps=training_steps, logger=logger, **trainer_args)
    trainer.fit(model, dataloader)
    
    mspace_ae = model.mspace_ae

    if decoder_training_steps > 0:
        # train the decoder only
        trainer = pl.Trainer(max_steps=decoder_training_steps, logger=logger, **trainer_args)
        model2 = TrainDecoder(mspace_ae, progress_bar=progress_bar, training_steps=decoder_training_steps, **model_kwargs)
        trainer.fit(model2, dataloader)
        model.mspace_ae = model2.mspace_ae

    if return_trainer:
        return model, trainer

    return model


def try_train_mspace(*args, **kwargs):
    # TODO: msapce training sometimes fails into nan, why?
    for i in range(3):
        try:
            model, trainer = train_mspace_model(*args, **kwargs)
            return model, trainer
        except Exception as e:
            warnings.warn(f"Error in training mspace model: {e}\nTrying again...")
            continue
    raise Exception("Failed to train mspace model after 3 times")
    # model, trainer = train_mspace_model(*args, **kwargs)
    # return model, trainer

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
