from typing import List, Tuple, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .predictor import NcutPredictor, NotInitializedError


class NcutVisionPredictor:
    _initialized: bool = False

    def __init__(self,
                 model: nn.Module,
                 transform: transforms.Compose,
                 batch_size: int):
        self.model = model
        self.transform = transform

        self.batch_size = batch_size

        self._images: List[Image.Image]
        self._image_whs: List[Tuple[int, int]]
        self._feat_hws: Tuple[int, int]

        self.predictor = NcutPredictor()

    def set_images(self,
                   images: List[Image.Image],
                   n_segments: List[int] = (5, 10, 20, 40, 80)):
        """
        set the images and save its features in the cache.
        
        Args:
            images (List[Image.Image]): List of images to set.
            n_segments (List[int], optional): Number of segments to cache. Defaults to [5, 10, 20, 40, 80]. 
                n_segments is showed in the preview function.
        """
        features = self.forward_model(images)
        self._images = images
        self._image_whs = np.array([image.size for image in images])
        self._feat_hws = (features.shape[2], features.shape[3])

        flat_features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        self.predictor.initialize(flat_features, n_segments)
        self._initialized = True

    @torch.no_grad()
    def forward_model(self, images: List[Image.Image]) -> torch.Tensor:
        device = next(self.model.parameters()).device

        all_features = []
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            transformed_images = torch.stack([self.transform(image) for image in batch_images])
            transformed_images = transformed_images.to(device)
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
                image_indices: int) -> List[torch.Tensor]:
        """
        preview the hierarchy cluster assignment for the images.
        
        Args:
            point_coord (Tuple[int, int]): The coordinate of the point to preview.
            image_indices (int): The index of the image to preview.
            
        Returns:
            List[torch.Tensor]: List of masks for each hierarchy level. each mask is (b, h, w)
        """
        self.__check_initialized()
        b, h, w = len(self._images), self._feat_hws[0], self._feat_hws[1]

        point_index = image_xy_to_tensor_index(self._image_whs, self._feat_hws,
                                               [point_coord], [image_indices])[0]
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
        discrete_rgb = np.zeros((b * h * w, 3), dtype=np.uint8)
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

    def to(self, device: Union[str, torch.device] = "cpu"):
        self.model = self.model.to(device)
        self.predictor = self.predictor.to(device)
        return self


def image_xy_to_tensor_index(image_whs: np.array,
                             feat_hws: np.array,
                             point_coords: np.ndarray,
                             image_indices: np.ndarray) -> np.ndarray:
    """
    Convert image xy coordinates to tensor index.
    Args:
        image_whs: List of image width and height, (n_images, 2)
        feat_hws: Feature width and height, (2, )
        point_coords: Point coordinates, (n_points, 2)
        image_indices: Image indices for each point, (n_points, )
    Returns:
        Point indices
    """
    if len(point_coords) == 0:
        return np.array([], dtype=np.int64)

    wh = image_whs[image_indices]
    point_coords = point_coords / wh

    point_coords = np.flip(point_coords, axis=1)  # (x, y) -> (y, x)

    point_coords = point_coords * feat_hws
    point_coords = point_coords.astype(np.int64)

    point_indices = point_coords[:, 0] * feat_hws[0] + point_coords[:, 1]

    offset_perimg = np.prod(feat_hws)
    offsets = image_indices * offset_perimg
    point_indices = point_indices + offsets
    point_indices = point_indices.astype(np.int64)
    return point_indices


def draw_segments_boundaries(images: List[np.ndarray], min_area: int = 100):
    # images: (n_images, h, w, 3)
    assert images.ndim == 4
    n_images = images.shape[0]
    output = []
    for i in range(n_images):
        output.append(draw_segments_boundaries_one_image(images[i], min_area))
    output = np.stack(output)
    return output


def draw_segments_boundaries_one_image(image: np.ndarray, min_area: int = 100):
    # image: (h, w, 3)
    assert image.ndim == 3
    # Get unique colors (excluding black as background if necessary)
    unique_colors = np.unique(image.reshape(-1, 3), axis=0)

    # Create a copy of the original image to draw boundaries
    output = image.copy()

    # Iterate through each unique color
    for color in unique_colors:
        # Create a mask for the current color
        mask = np.all(image == color, axis=-1).astype(np.uint8) * 255

        # Find contours of the component
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw black boundary only if the component is large enough
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                cv2.drawContours(output, [contour], -1, (0, 0, 0), 1)

    return output

