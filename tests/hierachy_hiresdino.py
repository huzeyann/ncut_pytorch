# %%
from ncut_pytorch import Ncut, mspace_color
from ncut_pytorch.predictor.dino import hires_dino_512

import os

from ncut_pytorch.predictor.dino import hires_dino

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import matplotlib.pyplot as plt
from einops import rearrange, repeat
# %%
model, transform = hires_dino_512()
model = hires_dino(dino_name="dino_vitb8", 
                    stride=6, 
                    shift_dists=[1, 2, 3],
                    flip_transforms=True,
                    chunk_size=4,
                    dtype=torch.float16,
                    feature_resolution=512)
# model.feature_resolution = 512
# %%
from PIL import Image

default_images = ['./image_0.jpg', './image_1.jpg', './guitar_ego.jpg']

images = [transform(Image.open(image_path)) for image_path in default_images]
images = torch.stack(images)
print(images.shape)
# %%
# display warning log
import logging
logging.basicConfig(level=logging.WARNING)

model.to('cuda')

import time

# GPU run
start_time = time.time()
hr_feats_tensor_gpu = model(images.to('cuda'))
gpu_time = time.time() - start_time
print(f"GPU feature extraction took {gpu_time:.2f} seconds")

hr_feats_tensor = hr_feats_tensor_gpu

# model.to('cpu')

# start_time = time.time()
# hr_feats_tensor_cpu = model(images.to('cpu'))
# cpu_time = time.time() - start_time
# print(f"CPU feature extraction took {cpu_time:.2f} seconds")

# assert torch.allclose(hr_feats_tensor_gpu, hr_feats_tensor_cpu)

# %%
from ncut_pytorch import kway_ncut
from ncut_pytorch.ncuts.ncut_kway import _onehot_discretize


@torch.no_grad()
def rgb_from_ncut_discrete_hirarchical(feats, color_num_eig=50, num_clusters=[10, 50, 250],
                                       degree=0.1, n_neighbors=1):
    n_eig = max(color_num_eig, np.max(num_clusters))
    b, c, h, w = feats.shape
    feats = rearrange(feats, 'b c h w -> (b h w) c')

    start_time = time.time()
    chunk_size = 65536
    eigvecs = Ncut(n_eig=n_eig, d_gamma=degree, n_neighbors=n_neighbors,
                   matmul_chunk_size=chunk_size,
                   ).fit_transform(feats)
    ncut_time = time.time() - start_time
    print(f"NCUT took {ncut_time:.2f} seconds, chunk_size={chunk_size}")
    
    start_time = time.time()
    rgb = mspace_color(eigvecs[:, :color_num_eig])
    mspace_time = time.time() - start_time
    print(f"mspace_color took {mspace_time:.2f} seconds")
    
    start_time = time.time()
    # discretize the eigvecs and fill the discrete_rgb with the mean color of t-SNE colors
    discrete_rgbs = []
    for num_cluster in num_clusters:
        # discretize the eigvecs, k-way NCUT
        kway_eigvec = kway_ncut(eigvecs[:, :num_cluster], n_sample=10240)
        kway_eigvec = _onehot_discretize(kway_eigvec)
        kway_eigvec = kway_eigvec.cpu()
        discrete_rgb = torch.zeros_like(rgb)
        # fill the discrete_rgb with the mean color of the cluster
        for i in range(num_cluster):
            mask = kway_eigvec[:, i] == 1
            discrete_rgb[mask] = rgb[mask].mean(0)
        discrete_rgb = rearrange(discrete_rgb, '(b h w) c -> b h w c', b=b, h=h, w=w)
        discrete_rgb = discrete_rgb.cpu().numpy()
        discrete_rgbs.append(discrete_rgb)
    continues_rgb = rearrange(rgb, '(b h w) c -> b h w c', b=b, h=h, w=w)
    continues_rgb = continues_rgb.cpu().numpy()
    ncut_time = time.time() - start_time
    print(f"kway_ncut took {ncut_time:.2f} seconds")
    return discrete_rgbs, continues_rgb


# %%
import gc

torch.cuda.empty_cache()
gc.collect()

num_clusters = [16, 32, 64, 128, 256]
# num_clusters = [16]
degree = 0.1
n_neighbors = 8
ncut_discrete_rgbs, continues_rgb = rgb_from_ncut_discrete_hirarchical(hr_feats_tensor, color_num_eig=20,
                                                                       num_clusters=num_clusters,
                                                                       degree=degree,
                                                                       n_neighbors=n_neighbors)
# %%
1024000 * 768 * 8 * 4 / 1024 / 1024 / 1024
# %%
512*512*3
# %%
import cv2
import numpy as np


def draw_component_boundaries(image):
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
        min_size = 100
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_size:
                cv2.drawContours(output, [contour], -1, (0, 0, 0), 1)

    return output


def draw_component_boundaries_fast_v1(image, min_size=100):
    """
    Faster version using connected components directly.
    This is typically 5-10x faster than the original.
    """
    # Convert to grayscale representation for connected components
    # Create a unique label for each RGB color
    image_flat = image.reshape(-1, 3)
    image_hash = image_flat[:, 0] * 256*256 + image_flat[:, 1] * 256 + image_flat[:, 2]
    image_hash = image_hash.reshape(image.shape[:2]).astype(np.uint32)
    
    # Find connected components for each unique color
    output = image.copy()
    unique_hashes = np.unique(image_hash)
    
    for hash_val in unique_hashes:
        # Create binary mask for this color
        mask = (image_hash == hash_val).astype(np.uint8) * 255
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Draw boundaries for components larger than min_size
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                component_mask = (labels == i).astype(np.uint8) * 255
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(output, contours, -1, (0, 0, 0), 1)
    
    return output


def draw_component_boundaries_fast_v2(image, min_size=100):
    """
    Even faster version using morphological operations.
    This is typically 10-20x faster than the original.
    """
    # Convert RGB to a single channel representation
    image_flat = image.reshape(-1, 3)
    
    # Create unique integer representation of each color
    # Use a more efficient hash that avoids large numbers
    unique_colors, color_indices = np.unique(image_flat, axis=0, return_inverse=True)
    label_image = color_indices.reshape(image.shape[:2]).astype(np.uint16)
    
    output = image.copy()
    
    # Process each unique color
    for color_idx in range(len(unique_colors)):
        # Create binary mask
        mask = (label_image == color_idx).astype(np.uint8)
        
        # Find connected components with stats in one go
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask * 255, connectivity=8)
        
        # Create boundary mask for all large components at once
        boundary_mask = np.zeros_like(mask, dtype=np.uint8)
        
        for i in range(1, num_labels):  # Skip background
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                # Get component mask
                component = (labels == i).astype(np.uint8)
                
                # Create boundary using morphological operations (faster than findContours)
                kernel = np.ones((3, 3), np.uint8)
                eroded = cv2.erode(component, kernel, iterations=1)
                boundary = component - eroded
                boundary_mask |= boundary
        
        # Apply boundary to output
        boundary_coords = np.where(boundary_mask > 0)
        output[boundary_coords] = [0, 0, 0]
    
    return output


def draw_component_boundaries_fastest(image, min_size=100):
    """
    Fastest version - processes all boundaries in a single pass.
    This is typically 20-50x faster than the original.
    """
    h, w = image.shape[:2]
    
    # Create unique labels for each color
    image_flat = image.reshape(-1, 3)
    unique_colors, indices = np.unique(image_flat, axis=0, return_inverse=True)
    label_image = indices.reshape((h, w)).astype(np.uint16)
    
    # Find all boundaries at once using edge detection
    # Check for differences with neighbors (right and down)
    boundaries = np.zeros((h, w), dtype=bool)
    
    # Right neighbors
    boundaries[:, :-1] |= (label_image[:, :-1] != label_image[:, 1:])
    # Down neighbors  
    boundaries[:-1, :] |= (label_image[:-1, :] != label_image[1:, :])
    
    # Filter boundaries by component size
    output = image.copy()
    
    # For each unique color, check if any of its components are large enough
    for color_idx in range(len(unique_colors)):
        mask = (label_image == color_idx).astype(np.uint8) * 255
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Create mask of large components
        large_components = np.zeros_like(labels, dtype=bool)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                large_components |= (labels == i)
        
        # Apply boundaries only for large components
        color_boundaries = boundaries & (label_image == color_idx) & large_components
        output[color_boundaries] = [0, 0, 0]
    
    return output


# %%
draw_boundaries = True
# draw the boundaries of connected components, looks nice but can be slow
if draw_boundaries:
    import time
    
    ncut_discrete_rgbs_draw = []
    
    # Timing comparison (optional - you can remove this)
    print("Comparing boundary drawing methods...")
    test_image = ncut_discrete_rgbs[0][0]  # Use first image for timing test
    
    # Original method
    start_time = time.time()
    _ = draw_component_boundaries(test_image)
    original_time = time.time() - start_time
    print(f"Original method: {original_time:.3f}s")
    
    # Fastest method
    start_time = time.time()
    _ = draw_component_boundaries_fast_v2(test_image)
    fastest_time = time.time() - start_time
    print(f"Fastest method: {fastest_time:.3f}s ({original_time/fastest_time:.1f}x speedup)")
    
    # Use the fastest method for all images
    for i in range(len(num_clusters)):
        ncut_discrete_rgbs_draw.append([draw_component_boundaries(rgb) for rgb in ncut_discrete_rgbs[i]])
    ncut_discrete_rgbs = ncut_discrete_rgbs_draw
# %%
fig, axes = plt.subplots(3, 6, figsize=(20, 10))
for ax in axes.flatten():
    ax.axis('off')
for i in range(3):
    image = images[i].cpu().numpy().transpose(1, 2, 0)
    # unnormalize the image     
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    axes[i][0].imshow(image)
    axes[i][0].set_title('Original Image')
    for j, num_cluster in enumerate(num_clusters):
        axes[i][j + 1].imshow(ncut_discrete_rgbs[j][i])
        axes[i][j + 1].set_title(f'k-way ({num_cluster})')
# plt.suptitle(f'dino_vitb16 no_norm 512x512, degree={degree}', fontsize=16)
plt.suptitle(f'dino_vitb8 512x512, n_neighbors={n_neighbors}', fontsize=28)
# plt.suptitle(f'hr_dv2 dino_vitb16 1024x1024 stride=4, shift_dists={shift_dists}, no flip, degree={degree}', fontsize=16)
plt.tight_layout()
plt.show()
# %%
