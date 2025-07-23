from gettext import find
from fpsample import np
from numpy.random import f
import torch
from PIL import Image
from typing import List
import torch.nn as nn
from ncut_pytorch.ncut import Ncut
from ncut_pytorch.ncuts.ncut_click import ncut_click_prompt, get_mask_and_heatmap
from ncut_pytorch.dino import hires_dino_256, hires_dino_512, hires_dino_1024
from ncut_pytorch.dino.transform import unnormalize
from ncut_pytorch import mspace_color
from ncut_pytorch import kway_ncut
from ncut_pytorch.utils.math_utils import pca_lowrank


MODEL_REGISTRY = {
    "dino_256": hires_dino_256,
    "dino_512": hires_dino_512,
    "dino_1024": hires_dino_1024,
}

import time
def timeit_decorator(name: str = None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            if name is None:
                print(f"Time taken: {end_time - start_time} seconds")
            else:
                print(f"{name} Time taken: {end_time - start_time} seconds")
            return result
        return wrapper
    return decorator

class NcutPredictor(nn.Module):
    def __init__(self, backbone: str = "dino_1024", n_eig_cache: int = 50, device: str = 'cuda', 
                 eigvec_reweight: bool = False, pca_dim: int = None):
        super().__init__()
        self.model, self.transform = MODEL_REGISTRY[backbone]()
        self.model = self.model.to(device)
        self.n_eig_cache = n_eig_cache
        
        self._images = None
        self._features = None
        self.eigvecs = None
        self.device = device
        self.eigvec_reweight = eigvec_reweight
        self.pca_dim = pca_dim

    # @timeit_decorator(name="set_images")
    def set_images(self, images: List[Image.Image]):
        self._images = images
        transformed_images = [self.transform(image) for image in images]
        transformed_images = torch.stack(transformed_images)
        transformed_images = transformed_images.to(self.device)
        features = self.model(transformed_images)
        self._features = features  # (b, c, h, w)
        b, c, h, w = self._features.shape
        if self.pca_dim is not None:
            _inp = self._features.permute(0, 2, 3, 1).reshape(-1, self._features.shape[1])
            _inp = pca_lowrank(_inp, self.pca_dim)
            _inp = _inp.reshape(b, h, w, self.pca_dim).permute(0, 3, 1, 2)
            self._features = _inp
            
        # _inp = self._features.permute(0, 2, 3, 1).reshape(-1, self._features.shape[1])
        # eigvecs = Ncut(_inp, n_eig=self.n_eig_cache, device=self.device)
        # eigvecs = eigvecs.reshape(b, h, w, self.n_eig_cache)
        # self.eigvecs = eigvecs

    @timeit_decorator(name="predict")
    def predict(self, 
                point_coords: np.ndarray, 
                point_labels: np.ndarray, 
                image_indices: np.ndarray,
                **kwargs):
        fg_indices = self.point_to_tensor(point_coords[point_labels == 1], image_indices[point_labels == 1])
        bg_indices = self.point_to_tensor(point_coords[point_labels == 0], image_indices[point_labels == 0])
        # _inp = self.eigvecs.reshape(-1, self.n_eig_cac2he)
        _inp = self._features.permute(0, 2, 3, 1).reshape(-1, self._features.shape[1])
        eigvecs, eigval = ncut_click_prompt(
            _inp,
            fg_indices,
            bg_indices,
            n_eig=self.n_eig_cache,
            **kwargs,
        )
        mask, heatmap = get_mask_and_heatmap(eigvecs, fg_indices, num_cluster=kwargs['n_clusters'])
        b, c, h, w = self._features.shape
        mask = mask.reshape(b, h, w)
        heatmap = heatmap.reshape(b, h, w)

        return eigvecs, mask, heatmap
    
    def to(self, device: str):
        self.model = self.model.to(device)
        self.device = device
        return self
    
    def point_to_tensor(self, point_coords: np.ndarray, image_indices: np.ndarray):
        if len(point_coords) == 0:
            return torch.tensor([], dtype=torch.long)
        fg_indices, bg_indices = [], []
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

from ncut_pytorch.utils.sample_utils import farthest_point_sampling
from ncut_pytorch.utils.gamma import find_gamma_by_degree_after_fps
from ncut_pytorch.ncuts.ncut_click import _build_nystrom_graph
from ncut_pytorch.ncuts.ncut_click import ncut_click_prompt_cached
from ncut_pytorch.utils.math_utils import chunked_matmul

class NcutPredictorCached(NcutPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_weights = None

    def set_images(self, images: List[Image.Image]):
        self._images = images
        transformed_images = [self.transform(image) for image in images]
        transformed_images = torch.stack(transformed_images)
        transformed_images = transformed_images.to(self.device)
        features = self.model(transformed_images)
        self._features = features  # (b, c, h, w)
        b, c, h, w = self._features.shape
        
        _inp = self._features.permute(0, 2, 3, 1).reshape(-1, self._features.shape[1])
        nystrom1_indices = farthest_point_sampling(_inp, 10240, device=self.device)   
        gamma = find_gamma_by_degree_after_fps(_inp[nystrom1_indices], d_gamma=0.1)
        self.gamma = gamma
        eigvecs = Ncut(_inp[nystrom1_indices], n_eig=self.n_eig_cache, device=self.device, gamma=gamma)
        nystrom2_indices = farthest_point_sampling(eigvecs, 1024, device=self.device)
        nystrom_indices = nystrom1_indices[nystrom2_indices]
        self.nystrom_indices = nystrom_indices
        self.cached_weights = _build_nystrom_graph(_inp, _inp[nystrom_indices], gamma=gamma, device=self.device)

    @timeit_decorator(name="predict")
    def predict(self, 
                point_coords: np.ndarray, 
                point_labels: np.ndarray, 
                image_indices: np.ndarray,
                **kwargs):
        fg_indices = self.point_to_tensor(point_coords[point_labels == 1], image_indices[point_labels == 1])
        bg_indices = self.point_to_tensor(point_coords[point_labels == 0], image_indices[point_labels == 0])
        # _inp = self.eigvecs.reshape(-1, self.n_eig_cac2he)
        _inp = self._features.permute(0, 2, 3, 1).reshape(-1, self._features.shape[1])
        eigvecs, eigval = ncut_click_prompt_cached(
            self.nystrom_indices,
            self.gamma,
            _inp,
            fg_indices,
            bg_indices,
            n_eig=self.n_eig_cache,
            **kwargs,
        )
        eigvecs = self.cached_weights.float() @ eigvecs
        # eigvecs = chunked_matmul(self.cached_weights.float(), eigvecs, device=self.device)
        mask, heatmap = get_mask_and_heatmap(eigvecs, fg_indices, num_cluster=kwargs['n_clusters'])
        b, c, h, w = self._features.shape
        mask = mask.reshape(b, h, w)
        heatmap = heatmap.reshape(b, h, w)

        return eigvecs, mask, heatmap