import torch
from PIL import Image
from typing import List, Tuple
import torch.nn as nn
import numpy as np
from ncut_pytorch import Ncut, kway_ncut
from ncut_pytorch.ncuts.ncut_click import ncut_click_prompt, get_mask_and_heatmap
from ncut_pytorch.dino import hires_dino_256, hires_dino_512, hires_dino_1024


MODEL_REGISTRY = {
    "dino_256": hires_dino_256,
    "dino_512": hires_dino_512,
    "dino_1024": hires_dino_1024,
}


class NcutPredictor(nn.Module):
    def __init__(self, backbone: str = "dino_512", 
                 n_eig_hierarchy: List[int] = [5, 10, 20, 40, 80], 
                 device: str = 'cuda'):
        super().__init__()
        self.model, self.transform = MODEL_REGISTRY[backbone]()
        self.model = self.model.to(device)
        self.n_eig_hierarchy = n_eig_hierarchy
        
        self._images = None
        self._features = None
        self.device = device

    def set_images(self, images: List[Image.Image]):
        self._images = images
        transformed_images = [self.transform(image) for image in images]
        transformed_images = torch.stack(transformed_images)
        transformed_images = transformed_images.to(self.device)
        features = self.model(transformed_images)
        self._features = features  # (b, c, h, w)
        
        self._cache_hierarchical_eigvecs()
    
    def _cache_hierarchical_eigvecs(self):
        b, c, h, w = self._features.shape
        _inp = self._features.permute(0, 2, 3, 1).reshape(-1, self._features.shape[1])
        eigvecs = Ncut(_inp, n_eig=max(self.n_eig_hierarchy), device=self.device)
        
        hierarchy_eigvecs = []
        for n_eig in self.n_eig_hierarchy:
            _eigvecs = kway_ncut(eigvecs[:, :n_eig], device=self.device)
            hierarchy_eigvecs.append(_eigvecs)
        self.hierarchy_eigvecs = hierarchy_eigvecs
        
    def preview(self,
                point_coord: Tuple[int, int],
                image_indice: int):
        b, c, h, w = self._features.shape
        
        point_coords = np.array([point_coord])
        image_indices = np.array([image_indice])
        point_index = self._point_to_tensor(point_coords, image_indices)[0]
        heatmaps = []
        masks = []
        for eigvec in self.hierarchy_eigvecs:
            cluster_idx = eigvec[point_index].argmax()
            mask = eigvec.argmax(dim=1) == cluster_idx
            mask = mask.reshape(b, h, w)
            masks.append(mask)
            heatmap = eigvec[:, cluster_idx].reshape(b, h, w)
            heatmaps.append(heatmap)
        return heatmaps, masks

    def predict(self, 
                point_coords: np.ndarray, 
                point_labels: np.ndarray, 
                image_indices: np.ndarray,
                **kwargs):
        fg_indices = self._point_to_tensor(point_coords[point_labels == 1], image_indices[point_labels == 1])
        bg_indices = self._point_to_tensor(point_coords[point_labels == 0], image_indices[point_labels == 0])
        _inp = self._features.permute(0, 2, 3, 1).reshape(-1, self._features.shape[1])
        n_cluster = kwargs.pop('n_clusters', 2)
        eigvecs, eigval = ncut_click_prompt(
            _inp,
            fg_indices,
            bg_indices,
            n_eig=n_cluster,
            **kwargs,
        )
        mask, heatmap = get_mask_and_heatmap(eigvecs, fg_indices, n_cluster=n_cluster)
        b, c, h, w = self._features.shape
        mask = mask.reshape(b, h, w)
        heatmap = heatmap.reshape(b, h, w)

        return eigvecs, mask, heatmap
    
    def to(self, device: str):
        self.model = self.model.to(device)
        self.device = device
        return self
    
    def _point_to_tensor(self, point_coords: np.ndarray, image_indices: np.ndarray):
        if len(point_coords) == 0:
            return torch.tensor([], dtype=torch.long)
        image_wh = [image.size for image in self._images]
        image_wh = np.array(image_wh)
        wh = image_wh[image_indices]
        point_coords = point_coords / wh
        
        point_coords = np.flip(point_coords, axis=1) # (x, y) -> (y, x)
        
        feat_hws = np.array(self._features.shape[2:])
        point_coords = point_coords * feat_hws
        point_coords = point_coords.astype(np.int32)
        
        point_indices = point_coords[:, 0] * feat_hws[0] + point_coords[:, 1]
        
        offset_perimg = np.prod(feat_hws)
        offsets = image_indices * offset_perimg
        point_indices = point_indices + offsets
        point_indices = torch.from_numpy(point_indices).to(dtype=torch.long)
        return point_indices
