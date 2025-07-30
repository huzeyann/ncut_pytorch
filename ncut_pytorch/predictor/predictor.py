import torch
from PIL import Image
from typing import List, Tuple, Callable
import torch.nn as nn
import numpy as np
from torchvision import transforms
from ncut_pytorch import Ncut, kway_ncut
from ncut_pytorch.ncuts.ncut_click import ncut_click_prompt
from ncut_pytorch.utils.math_utils import chunked_matmul
from ncut_pytorch.ncuts.ncut_kway import axis_align
from ncut_pytorch.ncuts.ncut_nystrom import _nystrom_propagate
from ncut_pytorch.dino import hires_dino_256, hires_dino_512, hires_dino_1024
from ncut_pytorch.dino import LowResDINO, HighResDINO
from ncut_pytorch import mspace_color
from ncut_pytorch.predictor.utils import draw_segments_boundaries, image_xy_to_tensor_index


class NotInitializedError(Exception):
    """Raised when trying to use predictor methods before calling initialize()"""
    pass


MODEL_REGISTRY = {
    "dino_256": hires_dino_256,
    "dino_512": hires_dino_512,
    "dino_1024": hires_dino_1024,
}

class NcutPredictor:
    _initialized : bool = False
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        self._features : torch.Tensor
        self._hierarchy_assign : List[torch.Tensor]
        self._eigvecs : torch.Tensor
        self._color_palette : torch.Tensor
        
    
    def initialize(self, features: torch.Tensor, n_segments: List[int] | int = [5, 10, 20, 40, 80]) -> None:
        self._features = features
        if isinstance(n_segments, int):
            n_segments = [n_segments]
        self.refresh_eigvecs(max(n_segments))
        self.cache_hierarchy(n_segments)
        self._initialized = True

    def refresh_eigvecs(self, n_eig: int) -> None:
        eigvecs = Ncut(self._features, n_eig=n_eig, device=self.device)
        self._eigvecs = eigvecs

    def get_n_eigvecs(self, n_eig: int) -> torch.Tensor:
        cache_hit = n_eig <= self._eigvecs.shape[1]
        if not cache_hit:
            self.refresh_eigvecs(n_eig)
        return self._eigvecs[:, :n_eig]

    def cache_hierarchy(self, n_segments: List[int]) -> None:
        hierarchy_assign = []
        for n_eig in n_segments:
            eigvecs = self.get_n_eigvecs(n_eig)
            kway_eigvecs = kway_ncut(eigvecs, device=self.device)
            cluster_assignment = kway_eigvecs.argmax(dim=1).cpu()
            hierarchy_assign.append(cluster_assignment)
        self._hierarchy_assign = hierarchy_assign
    
    def get_n_segments(self, n_cluster: int):
        self.__check_initialized()
        eigvecs = self.get_n_eigvecs(n_cluster)
        kway_eigvec = kway_ncut(eigvecs, device=self.device)
        cluster_assignment = kway_eigvec.argmax(dim=1).cpu()
        return cluster_assignment
    
    def get_hierarchy_masks(self, point_index: int) -> List[torch.Tensor]:
        self.__check_initialized()
        masks: List[torch.Tensor] = []
        for cluster_assignment in self._hierarchy_assign:
            cluster_idx = cluster_assignment[point_index].item()
            mask = cluster_assignment == cluster_idx
            masks.append(mask)
        return masks
    
    def predict_clicks(self, 
                             fg_indices: torch.Tensor,
                             bg_indices: torch.Tensor,
                             click_weight: float,
                             **kwargs):
        self.__check_initialized()
        eigvecs, eigval, nystrom_indices, gamma = ncut_click_prompt(
            self._features,
            fg_indices,
            bg_indices,
            return_indices_and_gamma=True,
            click_weight=click_weight,
            **kwargs,
        )

        eigvecs = kway_ncut(eigvecs, device=self.device)
        R = axis_align(eigvecs, device=self.device)
        kway_eigvecs = chunked_matmul(eigvecs, R, device=self.device, large_device=eigvecs.device)
        
        # find which cluster is the foreground
        fg_eigvecs = kway_eigvecs[fg_indices]
        fg_idx = fg_eigvecs.mean(0).argmax().item()
        bg_idx = 1 if fg_idx == 0 else 0
        
        # discretize the eigvecs
        mask = kway_eigvecs.argmax(dim=-1) == fg_idx
        heatmap = kway_eigvecs[:, fg_idx] - kway_eigvecs[:, bg_idx]      

        # save for inference use
        self.__nystrom_indices = nystrom_indices
        self.__gamma = gamma
        self.__click_eigvecs = eigvecs
        self.__R = R
        self.__fg_idx = fg_idx
        self.__bg_idx = bg_idx

        return mask, heatmap
    
    def inference_new_features(self, new_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.__check_initialized()
        if not hasattr(self, "__nystrom_indices"):
            raise ValueError("please call predict_clicks() before inference_new_features()")

        nystrom_X = self._features[self.__nystrom_indices]
        nystrom_out = self.__click_eigvecs[self.__nystrom_indices]
        eigvecs = _nystrom_propagate(nystrom_out, new_features, nystrom_X, 
                                     gamma=self.__gamma, device=self.device)
        eigvecs = chunked_matmul(eigvecs, self.__R, device=self.device, large_device=eigvecs.device)
        mask = eigvecs.argmax(dim=-1) == self.__fg_idx
        heatmap = eigvecs[:, self.__fg_idx] - eigvecs[:, self.__bg_idx]
        return mask, heatmap
    
    def get_color_palette(self, n_eig: int = 50) -> torch.Tensor:
        cache_hit = hasattr(self, '_color_palette') and len(self._color_palette) > 0
        if not cache_hit:
            self.refresh_color_palette(n_eig)
        return self._color_palette
    
    def refresh_color_palette(self, n_eig: int = 50) -> None:
        self.__check_initialized()
        self._color_palette = mspace_color(self._eigvecs[:, :n_eig])

    def __check_initialized(self):
        if not self._initialized or not hasattr(self, '_features') or not hasattr(self, '_hierarchy_assign') or not hasattr(self, '_eigvecs'):
            raise NotInitializedError("Not initialized, please call initialize() first")

    def to(self, device: str):
        self.device = device    

