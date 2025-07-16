# %%
import torch

from ncut_pytorch.sample_utils import farthest_point_sampling, nystrom_propagate, auto_divice
from ncut_pytorch.affinity_gamma import find_gamma_by_degree_after_fps
from ncut_pytorch.math_utils import get_affinity, normalize_affinity, svd_lowrank, correct_rotation
from ncut_pytorch.kway_ncut import kway_ncut

# %%
from einops import rearrange
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

from ncut_pytorch.biased_ncut import get_mask_and_heatmap, bias_ncut_soft
# %%

default_images = ['/images/image_0.jpg', '/images/image_1.jpg', 
                  '/images/guitar_ego.jpg']


from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
transform_inv = transforms.Compose([
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    transforms.ToPILImage(),
])
images = []
for image_path in default_images:
    image = Image.open(image_path)
    image = transform(image)
    images.append(image)
images = torch.stack(images)
print(images.shape)
b, c, h_img, w_img = images.shape
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
features = my_highreso_feature(upsampler, images)
print(features.shape)
# torch.Size([3, 384, 1024, 1024])

# %%
b, c, h, w = features.shape

clicks1 = torch.tensor([[0.41699219, 0.50292969], [0.48925781, 0.25585938], [0.59472656, 0.19238281]])
clicks1 = (clicks1[:, 0] * h).long() * w + (clicks1[:, 1] * w).long()
clicks2 = torch.tensor([[0.33398438, 0.21386719], [0.63671875, 0.59082031]])
clicks2 = (clicks2[:, 0] * h).long() * w + (clicks2[:, 1] * w).long()
click_list = [clicks1, clicks2]

b, c, h, w = features.shape

bias_factor = 0.5
input_features = rearrange(features, 'b c h w -> (b h w) c')

eigvecs, eigvals = bias_ncut_soft(input_features, clicks1, clicks2, bg_factor=0.1, bias_factor=0.8)
mask, heatmap = get_mask_and_heatmap(eigvecs, clicks1)
heatmap = rearrange(heatmap, '(b h w) -> b h w', b=b, h=h, w=w)
mask = rearrange(mask, '(b h w) -> b h w', b=b, h=h, w=w)
# %%
fig, axs = plt.subplots(3, 3, figsize=(10, 10))
for ax in axs.flatten():
    ax.axis('off')
for i_image in range(3):
    axs[0, i_image].imshow(transform_inv(images[i_image]))
    axs[1, i_image].imshow(heatmap[i_image].cpu().numpy(), cmap='bwr', vmin=-1, vmax=1)
    axs[2, i_image].imshow(mask[i_image].cpu().numpy())
plt.tight_layout()
plt.show()
# %%
