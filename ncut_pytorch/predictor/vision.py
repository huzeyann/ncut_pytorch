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

from .predictor import NcutPredictor, NotInitializedError

class NcutVisionPredictor:
    _initialized : bool = False
    
    def __init__(self, 
                 model: nn.Module,
                 transform: transforms.Compose,
                 device: str = 'cuda',
                 batch_size: int = 32):
        self.model = model
        self.transform = transform
        self.model = self.model.to(device)
        
        self.device = device
        self.batch_size = batch_size
        
        self._images : List[Image.Image]
        self._image_whs : List[Tuple[int, int]]
        self._feat_hws : Tuple[int, int]
        
        self.predictor = NcutPredictor(device=device)
        

    def set_images(self, 
                   images: List[Image.Image],
                   n_segments: List[int] = [5, 10, 20, 40, 80]):
        """
        set the images and save its features in the cache.
        
        Args:
            images (List[Image.Image]): List of images to set.
            n_segments (List[int], optional): Number of segments to cache. Defaults to [5, 10, 20, 40, 80]. 
                n_segments is showed in the preview function.
        """
        features = self.forward_model(images)
        self._images = images
        self._image_whs = [image.size for image in images]
        self._feat_hws = (features.shape[2], features.shape[3])
        
        flat_features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        self.predictor.initialize(flat_features, n_segments)
        self._initialized = True

    @torch.no_grad()
    def forward_model(self, images: List[Image.Image]) -> torch.Tensor:
        all_features = []
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            transformed_images = torch.stack([self.transform(image) for image in batch_images])
            transformed_images = transformed_images.to(self.device)
            features = self.model(transformed_images)
            features = features.to('cpu')
            all_features.append(features)
        return torch.cat(all_features, dim=0)

    def generate(self, n_cluster: int) -> torch.Tensor:
        """
        generate the cluster assignment for the images.
        
        Args:
            n_cluster (int): Number of clusters to generate.
            
        Returns:
            torch.Tensor: Cluster assignment for the images. (b, h, w)
        """
        self.__check_initialized()
        cluster_assignment = self.predictor.get_n_segments(n_cluster)
        b, h, w = len(self._images), self._feat_hws[0], self._feat_hws[1]
        cluster_assignment = cluster_assignment.reshape(b, h, w)
        return cluster_assignment
    
    def preview(self,
                point_coord: Tuple[int, int],
                image_indice: int) -> List[torch.Tensor]:
        """
        preview the hierarchy cluster assignment for the images.
        
        Args:
            point_coord (Tuple[int, int]): The coordinate of the point to preview.
            image_indice (int): The index of the image to preview.
            
        Returns:
            List[torch.Tensor]: List of masks for each hierarchy level. each mask is (b, h, w)
        """
        self.__check_initialized()
        b, h, w = len(self._images), self._feat_hws[0], self._feat_hws[1]
        
        point_index = image_xy_to_tensor_index(self._image_whs, self._feat_hws, 
                                               [point_coord], [image_indice])[0]
        masks = self._get_mask_preview(point_index)
        masks = [mask.reshape(b, h, w) for mask in masks]
        return masks

    def predict(self, 
                point_coords: np.ndarray, 
                point_labels: np.ndarray, 
                image_indices: np.ndarray,
                click_weight: float = 0.5,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        predict the mask and heatmap for the images, based on the clicks.
        
        Args:
            point_coords (np.ndarray): The coordinates of the points to predict. (n, 2)
            point_labels (np.ndarray): The labels of the points to predict. (n, )
            image_indices (np.ndarray): The indices of the images to predict. (n, )
            click_weight (float, optional): The weight of the click. Defaults to 0.5.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mask and heatmap for the images. (b, h, w)
        """
        self.__check_initialized()
        fg_indices = image_xy_to_tensor_index(self._image_whs, self._feat_hws, 
                                              point_coords[point_labels == 1], 
                                              image_indices[point_labels == 1])
        bg_indices = image_xy_to_tensor_index(self._image_whs, self._feat_hws, 
                                              point_coords[point_labels == 0], 
                                              image_indices[point_labels == 0])
        b, h, w = len(self._images), self._feat_hws[0], self._feat_hws[1]
        
        mask, heatmap = self.predictor.predict_clicks(fg_indices, bg_indices, click_weight, **kwargs)
    
        mask = mask.reshape(b, h, w)
        heatmap = heatmap.reshape(b, h, w)
        
        return mask, heatmap
    
  
    def inference(self, images: List[Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        inference the mask and heatmap for new images, based on the saved states in the predict function.
        
        Args:
            images (List[Image.Image]): List of images to inference.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mask and heatmap for the images. (b, h, w)
        """
        self.__check_initialized()
        new_features = self.forward_model(images)
        b, c, h, w = new_features.shape
        new_features = new_features.permute(0, 2, 3, 1).reshape(-1, c)
                
        mask, heatmap = self.predictor.inference_new_features(new_features)
        
        mask = mask.reshape(b, h, w)
        heatmap = heatmap.reshape(b, h, w)
        return mask, heatmap
    
    def color_continues(self) -> np.ndarray:
        """
        color the features by continues mspace color palette.
        
        Returns:
            np.ndarray: RGB image. (b, h, w, 3)
        """
        self.__check_initialized()
        b, h, w = len(self._images), self._feat_hws[0], self._feat_hws[1]
        color_palette = self.predictor.get_color_palette()
        rgb = color_palette.reshape(b, h, w, 3)
        rgb = (rgb * 255).to(torch.uint8).cpu().numpy()
        return rgb
        
    def color_discrete(self, cluster_assignment: torch.Tensor, draw_boundaries: bool = True) -> np.ndarray:
        """
        color the features by discrete mspace color palette.
        
        Args:
            cluster_assignment (torch.Tensor): Cluster assignment for the images. (b, h, w)
            draw_boundaries (bool, optional): Whether to draw boundaries. Defaults to True.
            
        Returns:
            np.ndarray: RGB image. (b, h, w, 3)
        """
        self.__check_initialized()
        b, h, w = cluster_assignment.shape
        color_palette = self.predictor.get_color_palette()
        n_cluster = cluster_assignment.max() + 1
        cluster_assignment = cluster_assignment.flatten()
        discrete_rgb = np.zeros((b*h*w, 3), dtype=np.uint8)
        for i in range(n_cluster):
            mask = cluster_assignment == i
            color = color_palette[mask].mean(0)
            color = (color * 255).cpu().numpy()
            color = np.clip(color, 0, 255).astype(np.uint8)
            discrete_rgb[mask] = color
        discrete_rgb = discrete_rgb.reshape(b, h, w, 3)
        
        if draw_boundaries:
            discrete_rgb = draw_segments_boundaries(discrete_rgb)
            
        return discrete_rgb
    
    def __check_initialized(self):
        if not self._initialized:
            raise NotInitializedError("Not initialized, please call set_images() first")
        
        try:
            self.predictor.__check_initialized()
        except NotInitializedError:
            raise NotInitializedError("Not initialized, please call set_images() first")

    def to(self, device: str):
        self.model = self.model.to(device)
        self.predictor.to(device)
        self.device = device