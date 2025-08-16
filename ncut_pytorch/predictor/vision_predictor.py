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
                   n_segments: List[int] = (5, 25, 50, 100, 250)):
        """
        set the images and save its features in the cache.
        
        Args:
            images (List[Image.Image]): List of images to set.
            n_segments (List[int], optional): Number of segments to cache.
                n_segments is showed in the preview function.
        """
        features = self.forward_model(images)  # (b, c, h, w)
        self._images = images
        self._image_whs = np.array([image.size for image in images])
        self._feat_hws = (features.shape[2], features.shape[3])

        flat_features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        self.predictor.initialize(flat_features, n_segments)
        self._initialized = True

    @torch.inference_mode()
    def forward_model(self, images: List[Image.Image]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        with torch.autocast(device_type=device.type, enabled=True):
            all_features = []
            for i in range(0, len(images), self.batch_size):
                batch_images = images[i:i + self.batch_size]
                transformed_images = torch.stack([self.transform(image) for image in batch_images])
                transformed_images = transformed_images.to(device)
                features = self.model(transformed_images)
                features = features.to('cpu')
                all_features.append(features)
        return torch.cat(all_features, dim=0)

    def generate(self, n_segment: int) -> torch.Tensor:
        """
        generate the cluster assignment for the images.
        
        Args:
            n_cluster (int): Number of clusters to generate.
            
        Returns:
            torch.Tensor: Cluster assignment for the images. (b, h, w)
        """
        self.__check_initialized()
        cluster_assignment = self.predictor.get_n_segments(n_segment)
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

        point_index = self._image_xy_to_tensor_index(self._image_whs, 
                                                     self._feat_hws,
                                                     np.array([point_coord]), 
                                                     np.array([image_indices])
                                                     )[0]
        masks = self.predictor.get_hierarchy_masks(point_index)
        masks = [mask.reshape(b, h, w) for mask in masks]
        return masks

    def summary(self,
                n_segments: List[int] = (5, 25, 50, 100, 250),
                draw_border: bool = True,
                ) -> List[torch.Tensor]:
        """
        summary the cluster assignment for the images.
        
        Args:
            n_segments (List[int]): Number of segments to summary.
        """
        self.__check_initialized()
        display_hw = 512
        
        colors = []
        colors.append(self._images)
        for n_segment in n_segments:
            cluster_assignment = self.generate(n_segment)
            color = self.color_discrete(cluster_assignment, draw_border=draw_border)
            colors.append(color)
        color = self.color_continues()
        colors.append(color)
        
        # make a grid of images
        n_rows = len(self._images)
        n_cols = len(n_segments) + 2
        grid_image = Image.new('RGB', size=(n_cols * display_hw, n_rows * display_hw))
        for i in range(n_rows):
            for j in range(n_cols):
                img = colors[j][i]
                img = Image.fromarray(np.array(img))
                img = img.resize((display_hw, display_hw), Image.Resampling.NEAREST)
                grid_image.paste(img, box=(j * display_hw, i * display_hw))
        return grid_image

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
        fg_indices = self._image_xy_to_tensor_index(self._image_whs, self._feat_hws,
                                                    point_coords[point_labels == 1],
                                                    image_indices[point_labels == 1])
        bg_indices = self._image_xy_to_tensor_index(self._image_whs, self._feat_hws,
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

    def color_discrete(self, 
                       cluster_assignment: torch.Tensor,
                       draw_border: bool = True,
                       ) -> List[Image.Image]:
        """
        color the features by discrete mspace color palette.
        
        Args:
            cluster_assignment (torch.Tensor): Cluster assignment for the images. (b, h, w)
            draw_boundaries (bool, optional): Whether to draw boundaries. Defaults to True.
            
        Returns:
            List[Image.Image]: List of RGB images.
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
        
        # convert to PIL image and resize to the original size
        pil_images = []
        for i in range(b):
            img = Image.fromarray(discrete_rgb[i])
            img = img.resize(self._images[i].size, Image.Resampling.NEAREST)
            if draw_border:
                img = self._draw_segments_border(img)
            pil_images.append(img)
        
        return pil_images
    
    def refresh_color_palette(self):
        self.predictor.refresh_color_palette()
    
    @staticmethod
    def _draw_segments_border(img: Image.Image, min_area_ratio: float = 0.0005) -> Image.Image:
        img = np.array(img)
        src_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        src_gray = src_gray.astype(np.int32)
    
        contours, hierarchy = cv2.findContours(src_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
        area_threshold = src_gray.shape[0] * src_gray.shape[1] * min_area_ratio
        drawing = img.copy()
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area >= area_threshold:
                color = (0, 0, 0)
                cv2.drawContours(drawing, contours, i, color, 1, cv2.LINE_8, hierarchy, 0)
        return Image.fromarray(drawing)

    @staticmethod
    def _image_xy_to_tensor_index(image_whs: np.array,
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

    def __check_initialized(self):
        if not self._initialized:
            raise NotInitializedError("Not initialized, please call set_images() first")

    def to(self, device: Union[str, torch.device]):
        self.model = self.model.to(device)
        self.predictor = self.predictor.to(device)
        return self