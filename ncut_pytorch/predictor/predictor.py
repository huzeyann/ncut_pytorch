from typing import List, Tuple, Union

import numpy as np
import torch

from ncut_pytorch import kway_ncut, ncut_fn
from ncut_pytorch import mspace_color
from ncut_pytorch.ncuts.ncut_click import ncut_click_prompt
from ncut_pytorch.ncuts.ncut_kway import axis_align
from ncut_pytorch.ncuts.ncut_nystrom import nystrom_propagate
from ncut_pytorch.utils.sample import farthest_point_sampling
from ncut_pytorch.utils.math import chunked_matmul


class NotInitializedError(Exception):
    """Raised when trying to use predictor methods before calling initialize()"""
    pass

class NcutPredictor:
    _initialized: bool = False
    device: str = 'cpu'

    def __init__(self):
        self._features: torch.Tensor
        self._hierarchy_assign: List[torch.Tensor]
        self._eigvecs: torch.Tensor
        self._color_palette: torch.Tensor

        # inference states
        self._nystrom_indices: torch.Tensor
        self._gamma: float
        self._click_eigvecs: torch.Tensor
        self._R: torch.Tensor
        self._fg_idx: int
        self._bg_idx: int
        
        # kway ncut states
        self._kway_sample_idx: torch.Tensor

    def initialize(self,
                   features: torch.Tensor,
                   n_segments: Union[List[int], int] = (5, 25, 50, 100, 250)
                   ) -> None:
        self._features = features
        if isinstance(n_segments, int):
            n_segments = [n_segments]
        self.refresh_eigvecs(max(n_segments))
        self._initialized = True
        self.cache_hierarchy(n_segments)
        self._color_palette = []

    def refresh_eigvecs(self, n_eig: int) -> None:
        eigvecs, eigval = ncut_fn(self._features, n_eig=n_eig, device=self.device)
        self._eigvecs = eigvecs
        self._kway_sample_idx = farthest_point_sampling(eigvecs, 10240, device=self.device)

    def get_n_eigvecs(self, n_eig: int) -> torch.Tensor:
        cache_hit = n_eig <= self._eigvecs.shape[1]
        if not cache_hit:
            self.refresh_eigvecs(n_eig)
        return self._eigvecs[:, :n_eig]

    def cache_hierarchy(self, n_segments: List[int]) -> None:
        hierarchy_assign = []
        for n_eig in n_segments:
            hierarchy_assign.append(self.get_n_segments(n_eig))
        self._hierarchy_assign = hierarchy_assign

    def get_n_segments(self, n_cluster: int) -> torch.Tensor:
        self.__check_initialized()
        eigvecs = self.get_n_eigvecs(n_cluster)
        kway_eigvec = kway_ncut(eigvecs, device=self.device, sample_idx=self._kway_sample_idx)
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
                       fg_indices: np.ndarray,
                       bg_indices: np.ndarray,
                       click_weight: float,
                       **kwargs
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self._nystrom_indices = nystrom_indices
        self._gamma = gamma
        self._click_eigvecs = eigvecs
        self._R = R
        self._fg_idx = fg_idx
        self._bg_idx = bg_idx

        return mask, heatmap

    def inference_new_features(self, new_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.__check_initialized()
        if not hasattr(self, "_nystrom_indices") or len(self._nystrom_indices) == 0:
            raise NotInitializedError("please call predict_clicks() before inference_new_features()")

        nystrom_X = self._features[self._nystrom_indices]
        nystrom_out = self._click_eigvecs[self._nystrom_indices]
        eigvecs = nystrom_propagate(nystrom_out, new_features, nystrom_X,
                                    gamma=self._gamma, device=self.device)
        eigvecs = chunked_matmul(eigvecs, self._R, device=self.device, large_device=eigvecs.device)
        mask = eigvecs.argmax(dim=-1) == self._fg_idx
        heatmap = eigvecs[:, self._fg_idx] - eigvecs[:, self._bg_idx]
        return mask, heatmap

    def get_color_palette(self, n_eig: int = 50) -> torch.Tensor:
        cache_hit = hasattr(self, '_color_palette') and len(self._color_palette) > 0
        if not cache_hit:
            self.refresh_color_palette(n_eig)
        return self._color_palette

    def refresh_color_palette(self, n_eig: int = 50) -> None:
        self.__check_initialized()
        self._color_palette = mspace_color(self._eigvecs[:, :n_eig])
        
    def inference_new_color_palette(self, new_image: torch.Tensor) -> torch.Tensor:
        ... # TODO: implement this

    def __check_initialized(self) -> None:
        if not self._initialized or not hasattr(self, '_features') or \
            not hasattr(self, '_eigvecs'):
            raise NotInitializedError("Not initialized, please call initialize() first")

    def to(self, device: Union[str, torch.device]):
        self.device = device
        return self
