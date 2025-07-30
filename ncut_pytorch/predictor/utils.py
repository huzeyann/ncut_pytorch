import cv2
import torch
import numpy as np
from typing import List, Tuple

def image_xy_to_tensor_index(image_whs: List[Tuple[int, int]], 
                             feat_hws: Tuple[int, int], 
                             point_coords: np.ndarray, 
                             image_indices: np.ndarray):
    """
    Convert image xy coordinates to tensor index.
    Args:
        image_whs: List of image width and height, (n_images, 2)
        feat_hws: Feature width and height, (2,)
        point_coords: Point coordinates, (n_points, 2)
        image_indices: Image indices for each point, (n_points,)
    Returns:
        Point indices
    """
    if len(point_coords) == 0:
        return np.array([], dtype=np.int64)
    
    image_whs = np.array(image_whs)
    wh = image_whs[image_indices]
    point_coords = point_coords / wh
    
    point_coords = np.flip(point_coords, axis=1) # (x, y) -> (y, x)
    
    feat_hws = np.array(feat_hws)
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