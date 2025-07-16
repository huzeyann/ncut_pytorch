# %%

from ncut_pytorch.sample_utils import farthest_point_sampling, nystrom_propagate, auto_divice
from ncut_pytorch.affinity_gamma import find_gamma_by_degree_after_fps
from ncut_pytorch.math_utils import get_affinity, normalize_affinity, svd_lowrank, correct_rotation
from ncut_pytorch.ncuts.ncut_kway import kway_ncut

def bias_ncut_multiclass(features, click_list, 
                        num_eig=50, bias_factor=0.5, 
                        num_sample=10240,
                        distance='rbf', degree=0.1, 
                        device=None):
    """
    Args:
        features: (n_nodes, n_features)
        click_idx: (n_clicks) indices of the clicked points
        bias_factor: (float) the factor of the bias term, decrease it to grow the mask bigger, need to be tuned for different images
        device: (torch.device)
        distance: (str) 'rbf' or 'cosine'
        degree: (float) the degree of the affinity matrix, 0.1 is good for most cases, decrease it will sharpen the affinity matrix
        num_sample: (int) increasing it does not necessarily improve the result
        num_eig: (int) does not matter since we only need a binary mask
    """
    n_nodes, n_features = features.shape
    num_sample = min(num_sample, n_nodes//4)
    # farthest point sampling
    fps_idx = farthest_point_sampling(features, n_sample=num_sample)
    fps_idx = torch.tensor(fps_idx, dtype=torch.long)
    # remove pos_idx and neg_idx from fps_idx
    click_idx = torch.cat(click_list)
    fps_idx = fps_idx[~torch.isin(fps_idx, click_idx)]
    # add pos_idx and neg_idx to fps_idx
    fps_idx = torch.cat([click_idx, fps_idx])

    new_click_list = []
    count = 0
    for click_idx in click_list:
        new_click_list.append(torch.arange(len(click_idx)) + count)
        count += len(click_idx)
    
    device = auto_divice(features.device, device)
    _input = features[fps_idx].to(device)

    gamma = find_gamma_by_degree_after_fps(_input, degree=degree, distance=distance)
    affinity = get_affinity(_input, distance=distance, gamma=gamma)
    affinity = normalize_affinity(affinity)
    
    # modify the affinity from the clicks
    click_fs = []
    for click_idx in new_click_list:
        click_f = 1 * affinity[click_idx].mean(0)
        click_fs.append(click_f)
    click_f = torch.stack(click_fs, dim=1)
    click_affinity = get_affinity(click_f, distance=distance, gamma=gamma)
    click_affinity = normalize_affinity(click_affinity)
    
    _A = bias_factor * click_affinity + (1 - bias_factor) * affinity
        
    eigvecs, eigvals, _ = svd_lowrank(_A, q=num_eig)
    eigvecs = correct_rotation(eigvecs)

    # propagate the eigenvectors to the full graph
    eigvecs = nystrom_propagate(eigvecs, features, features[fps_idx], distance=distance, device=device)
        
    return eigvecs, eigvals


# %%
from einops import rearrange
from PIL import Image
import torch
import matplotlib.pyplot as plt

from ncut_pytorch.ncuts.ncut_biased import get_mask_and_heatmap
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
    
upsampler = torch.hub.load("huzeyann/FeatUp", 'dino', use_norm=False, force_reload=False)
features = my_highreso_feature(upsampler, images)
print(features.shape)
# torch.Size([3, 384, 1024, 1024])

# %%
clicks = torch.tensor([[0.41699219, 0.50292969], [0.48925781, 0.25585938], [0.59472656, 0.19238281]])
b, c, h, w = features.shape
click_idx = (clicks[:, 0] * h).long() * w + (clicks[:, 1] * w).long()
print(click_idx)

bias_factor = 0.5
input_features = rearrange(features, 'b c h w -> (b h w) c')
eigvecs, eigvals = bias_ncut_multiclass(input_features, click_list, bias_factor=bias_factor, num_sample2=1024)

mask, heatmap = get_mask_and_heatmap(eigvecs, click_idx)

mask = rearrange(mask, '(b h w) -> b h w', b=b, h=h, w=w)
heatmap = rearrange(heatmap, '(b h w) -> b h w', b=b, h=h, w=w)
# %%
fig, axs = plt.subplots(3, 3, figsize=(10, 6))
for ax in axs.flatten():
    ax.axis('off')
for i_image in range(3):
    axs[0, i_image].imshow(transform_inv(images[i_image]))
    if i_image == 0:  # I only clicked on the first image
        axs[0, i_image].scatter(clicks[:, 1] * w_img, clicks[:, 0] * h_img, color='red', s=10)
    axs[1, i_image].imshow(mask[i_image].cpu().numpy())
    axs[2, i_image].imshow(heatmap[i_image].cpu().numpy(), cmap='bwr', vmin=-0.1, vmax=0.1)
    
    # find the centroid of the cluster
    centroid = heatmap[i_image].argmax().item()
    axs[2, i_image].scatter(centroid % w, centroid // h, color='green', s=100, marker='+', edgecolor='black')
plt.tight_layout()
plt.show()

# %%
clicks1 = torch.tensor([[0.41699219, 0.50292969], [0.48925781, 0.25585938], [0.59472656, 0.19238281]])
clicks1 = (clicks1[:, 0] * h).long() * w + (clicks1[:, 1] * w).long()
clicks2 = torch.tensor([[0.33398438, 0.21386719], [0.63671875, 0.59082031]])
clicks2 = (clicks2[:, 0] * h).long() * w + (clicks2[:, 1] * w).long()
# click_list = [clicks1, clicks2]
click_list = [clicks1]

b, c, h, w = features.shape

bias_factor = 0.5
input_features = rearrange(features, 'b c h w -> (b h w) c')

eigvecs, eigvals = bias_ncut_multiclass(input_features, click_list, bias_factor=bias_factor)

# %%
num_clusters = 2
kway_onehot = kway_ncut(eigvecs[:, :num_clusters])
kway_onehot = kway_onehot.cpu()
kway_indices = kway_onehot.argmax(1)

cluster_images = rearrange(kway_indices, '(b h w) -> b h w', b=b, h=h, w=w)
# %%
fig, axs = plt.subplots(3, 3, figsize=(10, 6))
for ax in axs.flatten():
    ax.axis('off')
for i_image in range(3):
    axs[0, i_image].imshow(transform_inv(images[i_image]))
    axs[1, i_image].imshow(cluster_images[i_image].cpu().numpy())
# %%