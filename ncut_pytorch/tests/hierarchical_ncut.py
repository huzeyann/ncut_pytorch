# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from tqdm import tqdm
from ncut_pytorch.backbone import load_model, DiNO
from ncut_pytorch.visualize_utils import rgb_from_3d_rgb_cube, quantile_normalize
from ncut_pytorch import NCUT, rgb_from_tsne_3d, rgb_from_umap_3d, rgb_from_tsne_2d
from ncut_pytorch import convert_to_lab_color

# %%
from PIL import Image
# image = Image.open("/workspace/data/egoexo_0001.jpg")

default_images = ['/images/image_0.jpg', '/images/image_1.jpg', 
                #   '/images/image_5.jpg',]
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
# class DinoLastLayer(torch.nn.Module):
#     def __init__(self):
#         super(DinoLastLayer, self).__init__()
#         model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
#         model = model.cuda()
#         self.model = model

#     def forward(self, x):
#         feat = self.model.get_intermediate_layers(images.cuda())[0]
#         hw = int(np.sqrt(feat.shape[1]))
#         feat = rearrange(feat[:, 1:], 'b (h w) c -> b c h w', h=hw, w=hw)
#         return feat

# from my_featup import my_highreso_feature
# with torch.no_grad():
#     model = DinoLastLayer()
#     featups = []
#     batch_size = 4
#     for i in range(0, images.shape[0], batch_size):
#         batch = images[i:i + batch_size].cuda()
#         featup = my_highreso_feature(model, batch)
#         featups.append(featup)
#     featup = torch.cat(featups, dim=0)
#     print(featup.shape)
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
# upsampler = torch.hub.load("huzeyann/FeatUp", 'sam', use_norm=False, force_reload=False)
# featup2 = my_highreso_feature(upsampler, images)
# featup = torch.cat([featup1, featup2], dim=1)
featup = featup1
print(featup.shape)
# %%
def rgb_from_ncut_tsne(feats, num_eig=50):
    b, c, h, w = feats.shape
    feats = rearrange(feats, 'b c h w -> (b h w) c')
    eigvecs, eigvals = NCUT(num_eig=num_eig, device='cuda:1', move_output_to_cpu=True,
                            affinity_focal_gamma=0.1, distance='rbf').fit_transform(feats)
    # eigvecs = torch.nan_to_num(eigvecs)
    x3d, rgb = rgb_from_tsne_3d(eigvecs, device='cuda:1', num_sample=1000, perplexity=500)
    rgb = rgb.cpu().numpy()
    # rgb = convert_to_lab_color(rgb)
    rgb = rearrange(rgb, '(b h w) c -> b h w c', b=b, h=h, w=w)
    return rgb

#%%
def plot_two_row(images, rgbs):
    fig, ax = plt.subplots(2, 3, figsize=(10, 5))
    for i in range(3):
        image = images[i].cpu().numpy().transpose(1, 2, 0)
        # unnormalize the image
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = np.clip(image, 0, 1)
        ax[0][i].imshow(image)
        ax[1][i].imshow(rgbs[i])
    plt.show()


# %%
from ncut_pytorch import kway_ncut

@torch.no_grad()
def rgb_from_ncut_discrete(feats, color_num_eig=50, num_clusters=[10, 50, 250]):
    num_eig = max(color_num_eig, np.max(num_clusters))
    b, c, h, w = feats.shape
    feats = rearrange(feats, 'b c h w -> (b h w) c')
    eigvecs, eigvals = NCUT(num_eig=num_eig, device='cuda:1', move_output_to_cpu=False,
                            affinity_focal_gamma=0.1, distance='rbf').fit_transform(feats)
    # x3d, rgb = rgb_from_tsne_2d(eigvecs[:, :color_num_eig], device='cuda:1', num_sample=1000, perplexity=500)
    # rgb = rgb.cpu().numpy()
    x3d, rgb = rgb_from_tsne_3d(eigvecs[:, :color_num_eig], device='cuda:1', num_sample=1000, perplexity=500)
    rgb = convert_to_lab_color(rgb)
    rgb = torch.from_numpy(rgb)
    
    discrete_rgbs = []
    for num_cluster in num_clusters:
        kway = kway_ncut(eigvecs[:, :num_cluster])
        kway = kway.cpu()
        discrete_rgb = torch.zeros_like(rgb)
        for i in range(num_cluster):
            mask = kway[:, i] == 1
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
ncut_discrete_rgbs, continues_rgb = rgb_from_ncut_discrete(featup, color_num_eig=50, num_clusters=num_clusters)
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
ncut_discrete_rgbs_draw = []
for i in range(len(num_clusters)):
    ncut_discrete_rgbs_draw.append([draw_component_boundaries(rgb) for rgb in ncut_discrete_rgbs[i]])
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
        axes[i][j+1].imshow(ncut_discrete_rgbs_draw[j][i])
        axes[i][j+1].set_title(f'k-way ({num_cluster})')
plt.suptitle('DiNO+SAM Features', fontsize=16)
plt.tight_layout()
plt.show()
    

# %%
@torch.no_grad()
def rgb_from_sam(image_paths):
    
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint="/data/download/sam_vit_h_4b8939.pth")
    sam.to(device='cuda:0')
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    color_images = []
    for image_path in tqdm(image_paths):
        color_image = run_sam(mask_generator, image_path)
        color_images.append(color_image)
    color_images = np.stack(color_images)
    del mask_generator
    del sam
    return color_images

def run_sam(mask_generator, image_path):
    _img = Image.open(image_path).convert("RGB").resize((1024, 1024))
    masks = mask_generator.generate(np.array(_img))
    segments = [mask['segmentation'] for mask in masks]
    segments = np.stack(segments)
    num_segments = segments.shape[0]
    # get unique colors for each segment, from hue 0 to 1
    hues = np.linspace(0, 1, num_segments)
    colors = plt.cm.hsv(hues)[:, :3]
    colors = colors * 255
    colors = colors.astype(np.uint8)
    # create a color image
    color_image = np.zeros((segments.shape[1], segments.shape[2], 3), dtype=np.uint8)
    for i in range(num_segments):
        color_image[segments[i]] = colors[i]
    # convert to PIL image
    color_image = Image.fromarray(color_image).convert("RGB")
    # return numpy array
    color_image = np.array(color_image)
    return color_image
# %%
sam_rgbs = rgb_from_sam(default_images)

plot_two_row(images, sam_rgbs)
# %%
num_clusters = [16, 32, 64, 128, 256]
ncut_discrete_rgbs, continues_rgb = rgb_from_ncut_discrete(featup, color_num_eig=100, num_clusters=num_clusters)

# %%
fig, axes = plt.subplots(3, 8, figsize=(20, 9.5))
for ax in axes.flatten():
    ax.axis('off')
for i in range(3):
    image = images[i].cpu().numpy().transpose(1, 2, 0)
    # unnormalize the image
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    axes[i][0].imshow(image)
    for j, num_cluster in enumerate(num_clusters):
        axes[i][j+1].imshow(ncut_discrete_rgbs[j][i])
        if i == 0:
            axes[i][j+1].set_title(f'DiNO NCUT ({num_cluster})', fontsize=16)
    
    axes[i][6].imshow(continues_rgb[i])
    if i == 0:
        axes[i][6].set_title('DiNO NCUT (*)', fontsize=16)
    
    axes[i][7].imshow(sam_rgbs[i])
    if i == 0:
        axes[i][7].set_title('SAM', fontsize=16)
    
    
plt.tight_layout()
plt.show()
# %%
fig, axes = plt.subplots(3, 7, figsize=(20, 9.5))
for ax in axes.flatten():
    ax.axis('off')
for i in range(3):
    image = images[i].cpu().numpy().transpose(1, 2, 0)
    # unnormalize the image
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    axes[i][0].imshow(image)
    for j, num_cluster in enumerate(num_clusters):
        axes[i][j+1].imshow(ncut_discrete_rgbs[j][i])
        if i == 0:
            axes[i][j+1].set_title(f'DiNO NCUT ({num_cluster})', fontsize=16)
    
    axes[i][6].imshow(sam_rgbs[i])
    if i == 0:
        axes[i][6].set_title('SAM', fontsize=16)
    
plt.tight_layout()
plt.savefig('/nfscc/ncut_pytorch/docs/images/ncut_hierarchy_vs_sam.jpg')
plt.show()
# %%
fig, axes = plt.subplots(3, 5, figsize=(12, 8))
for ax in axes.flatten():
    ax.axis('off')
for i in range(3):
    image = images[i].cpu().numpy().transpose(1, 2, 0)
    # unnormalize the image
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    axes[i][0].imshow(image)
    for j, num_cluster in enumerate([16, 64, 256]):
        axes[i][j+1].imshow(ncut_discrete_rgbs[j][i])
        if i == 0:
            axes[i][j+1].set_title(f'NCUT ({num_cluster})', fontsize=16)
    
    axes[i][4].imshow(sam_rgbs[i])
    if i == 0:
        axes[i][4].set_title('SAM', fontsize=16)
    
plt.tight_layout(pad=0.5)
# plt.savefig('/nfscc/ncut_pytorch/docs/images/ncut_hierarchy_vs_sam.png', bbox_inches='tight', pad_inches=0, dpi=72)
plt.show()
# %%
sd = torch.hub.load_state_dict_from_url("https://raw.githubusercontent.com/huzeyann/FeatUp/refs/heads/main/ckpts/sam_no_norm.ckpt")
# %%
