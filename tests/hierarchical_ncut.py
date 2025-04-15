# %%

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from ncut_pytorch import NCUT, rgb_from_tsne_3d, convert_to_lab_color
# %%
from PIL import Image
# image = Image.open("/workspace/data/egoexo_0001.jpg")

default_images = ['/images/image_0.jpg', '/images/image_1.jpg', 
                  '/images/guitar_ego.jpg']


from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
images = []
for image_path in default_images:
    image = Image.open(image_path)
    image = transform(image)
    images.append(image)
images = torch.stack(images)
print(images.shape)
# %%

@torch.no_grad()
def my_highreso_feature(upsampler, images):
    upsampler = upsampler.eval().cuda()
    featups = []
    for i_image in range(images.shape[0]):
        featup = upsampler(images[i_image].unsqueeze(0).cuda())
        featup = featup.cpu()
        featups.append(featup)
    featup = torch.cat(featups, dim=0)
    return featup
    
upsampler = torch.hub.load("huzeyann/FeatUp", 'dino', use_norm=False, force_reload=True)
featup1 = my_highreso_feature(upsampler, images)

## uncomment this to use DINO+SAM features
# upsampler = torch.hub.load("huzeyann/FeatUp", 'sam', use_norm=False, force_reload=False)
# featup2 = my_highreso_feature(upsampler, images)
# featup = torch.cat([featup1, featup2], dim=1)


featup = featup1
print(featup.shape)
# torch.Size([3, 384, 1024, 1024])


# %%
from ncut_pytorch import kway_ncut


@torch.no_grad()
def rgb_from_ncut_discrete_hirarchical(feats, color_num_eig=50, num_clusters=[10, 50, 250], 
                                       num_sample=10000, degree=0.1, distance='rbf'):
    
    num_eig = max(color_num_eig, np.max(num_clusters))
    b, c, h, w = feats.shape
    feats = rearrange(feats, 'b c h w -> (b h w) c')

    # num_sample should be at least 1/4 of the number of features, 
    # otherwise the NCUT would not be balanced and high-quality
    num_sample = min(num_sample, feats.shape[0]//4)

    # find the optimal gamma for the given degree
    # gamma = find_gamma_by_degree_after_fps(feats, degree, distance=distance, num_sample=num_sample)
    # print(f'gamma: {gamma}')

    # run NCUT
    eigvecs, eigvals = NCUT(num_eig=num_eig, device='cuda:0', move_output_to_cpu=False,
                            # affinity_focal_gamma=gamma, distance=distance,
                            # num_sample=10000, num_sample2=1024,
                            knn=10,
                            ).fit_transform(feats)
    # return None, None
    
    # use t-SNE to fill a base color palette
    x3d, rgb = rgb_from_tsne_3d(eigvecs[:, :color_num_eig], device='cuda:1', num_sample=1000, perplexity=500)
    rgb = convert_to_lab_color(rgb)
    rgb = torch.from_numpy(rgb)
    

    # discretize the eigvecs and fill the discrete_rgb with the mean color of t-SNE colors
    discrete_rgbs = []
    for num_cluster in num_clusters:
        # discretize the eigvecs, k-way NCUT
        kway_onehot = kway_ncut(eigvecs[:, :num_cluster])
        kway_onehot = kway_onehot.cpu()
        discrete_rgb = torch.zeros_like(rgb)
        # fill the discrete_rgb with the mean color of the cluster
        for i in range(num_cluster):
            mask = kway_onehot[:, i] == 1
            discrete_rgb[mask] = rgb[mask].mean(0)
        discrete_rgb = rearrange(discrete_rgb, '(b h w) c -> b h w c', b=b, h=h, w=w)
        discrete_rgb = discrete_rgb.cpu().numpy()
        discrete_rgbs.append(discrete_rgb)
    continues_rgb = rearrange(rgb, '(b h w) c -> b h w c', b=b, h=h, w=w)
    continues_rgb = continues_rgb.cpu().numpy()
    return discrete_rgbs, continues_rgb
# %%
import gc
torch.cuda.empty_cache()
gc.collect()
# %%
num_clusters = [16, 32, 64, 128, 256]
ncut_discrete_rgbs, continues_rgb = rgb_from_ncut_discrete_hirarchical(featup, color_num_eig=50, num_clusters=num_clusters, 
                                                           num_sample=10000, degree=0.1, distance='rbf')

# try a different degree=0.1, 
# smaller degree means less connected affinity and sharper NCUT eigvecs
# smaller degree can counter the blurriness from FeatUp features

# ncut_discrete_rgbs_degree, continues_rgb_degree = rgb_from_ncut_discrete_hirarchical(featup, color_num_eig=50, num_clusters=num_clusters, 
#                                                            num_sample=10000, degree=0.05, distance='rbf')
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
        min_size = 50
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_size:
                cv2.drawContours(output, [contour], -1, (0, 0, 0), 2)

    return output
# %%
draw_boundaries = True
# draw the boundaries of connected components, looks nice but can be slow
if draw_boundaries:
    ncut_discrete_rgbs_draw = []
    for i in range(len(num_clusters)):
        ncut_discrete_rgbs_draw.append([draw_component_boundaries(rgb) for rgb in ncut_discrete_rgbs[i]])
    ncut_discrete_rgbs = ncut_discrete_rgbs_draw
# %%
fig, axes = plt.subplots(3, 6, figsize=(20,10))
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
        axes[i][j+1].imshow(ncut_discrete_rgbs[j][i])
        axes[i][j+1].set_title(f'k-way ({num_cluster})')
plt.suptitle('DiNO Features (sample=10000,1024,p**2+1/D+gamma, knn=100)', fontsize=16)
plt.tight_layout()
plt.show()
    

# %%
