# %%
from einops import rearrange
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

from ncut_pytorch import affinity_from_features, kway_ncut
from ncut_pytorch.ncut_pytorch import find_gamma_by_degree
from ncut_pytorch.nystrom_utils import farthest_point_sampling, propagate_knn


def normalize_affinity(A):
    D = A.abs().sum(dim=-1).detach().clone()
    D = D + 1e-6
    # print(D.max(), D.min())
    A = A / torch.sqrt(D)[:, None]
    A = A / torch.sqrt(D)[None, :]
    return A


@torch.no_grad()
def correct_rotation(eigen_vector):
    # correct the random rotation (flipping sign) of eigenvectors
    rand_w = torch.ones(
        eigen_vector.shape[0], device=eigen_vector.device, dtype=eigen_vector.dtype
    )
    s = rand_w[None, :] @ eigen_vector
    s = s.sign()
    return eigen_vector * s


def my_bias_ncut(features, click_idx, bias_factor=0.5, device='cuda', 
                 distance='rbf', degree=0.1, num_sample=10000, num_eig=30):
    """
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
    fps_idx = farthest_point_sampling(features, num_sample=num_sample)
    fps_idx = torch.tensor(fps_idx, dtype=torch.long)
    # remove pos_idx and neg_idx from fps_idx
    fps_idx = fps_idx[~torch.isin(fps_idx, click_idx)]
    # add pos_idx and neg_idx to fps_idx
    fps_idx = torch.cat([click_idx, fps_idx])
    click_idx = torch.arange(len(click_idx))
    
    
    _input = features[fps_idx].to(device)

    gamma = find_gamma_by_degree(_input, degree=degree, distance=distance)

    affinity = affinity_from_features(_input, distance=distance, affinity_focal_gamma=gamma)
    affinity = normalize_affinity(affinity)
    
    # modify the affinity from the clicks
    click_f = 1 * affinity[click_idx].sum(0)
    click_affinity = affinity_from_features(click_f.unsqueeze(1), distance=distance, affinity_focal_gamma=gamma)
    click_affinity = normalize_affinity(click_affinity)
    
    _A = bias_factor * click_affinity + (1 - bias_factor) * affinity
        
    eigvecs, eigvals, _ = torch.svd_lowrank(_A, q=num_eig+6)
    eigvecs = correct_rotation(eigvecs)

    # propagate the eigenvectors to the full graph
    eigvecs = propagate_knn(eigvecs, features, features[fps_idx], knn=10, distance=distance, device=device)
        
    return eigvecs, eigvals


def kway_ncut_axis_align(eigvecs, num_cluster=2, num_sample=1000, device='cuda'):
    inp = eigvecs[:, :num_cluster]
    fps_idx = farthest_point_sampling(inp, num_sample=num_sample)
    _inp = inp[fps_idx].to(device)
    _, R = kway_ncut(_inp, return_rotation=True)
    R = R.to(eigvecs.device)
    continues_eigvecs = eigvecs[:, :num_cluster] @ R
    return continues_eigvecs


def get_mask_and_heatmap(eigvecs, click_idx, num_cluster=2, num_sample=1000, device='cuda'):
    eigvecs = kway_ncut_axis_align(eigvecs, num_cluster=num_cluster, num_sample=num_sample, device=device)
    # find which cluster is the foreground
    fg_eigvecs = eigvecs[click_idx]
    fg_idx = fg_eigvecs.mean(0).argmax().item()
    bg_idx = 1 if fg_idx == 0 else 0
    
    # discretize the eigvecs
    mask = eigvecs.argmax(dim=-1) == fg_idx

    heatmap = eigvecs[:, fg_idx] - eigvecs[:, bg_idx]
    
    return mask, heatmap


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
eigvecs, eigvals = my_bias_ncut(input_features, click_idx, bias_factor=bias_factor)

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
clicks2 = torch.tensor([[0.33398438, 0.21386719], [0.63671875, 0.59082031]])

for clicks in [clicks1, clicks2]:
    b, c, h, w = features.shape
    click_idx = (clicks[:, 0] * h).long() * w + (clicks[:, 1] * w).long()
    print(click_idx)

    masks, heatmaps = [], []

    bias_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for bias_factor in bias_factors:
        input_features = rearrange(features, 'b c h w -> (b h w) c')
        eigvecs, eigvals = my_bias_ncut(input_features, click_idx, bias_factor=bias_factor)

        mask, heatmap = get_mask_and_heatmap(eigvecs, click_idx)

        mask = rearrange(mask, '(b h w) -> b h w', b=b, h=h, w=w)
        heatmap = rearrange(heatmap, '(b h w) -> b h w', b=b, h=h, w=w)

        masks.append(mask)
        heatmaps.append(heatmap)

    # make plot
    fig, axs = plt.subplots(3, 11, figsize=(12, 6))
    for ax in axs.flatten():
        ax.axis('off')
    for i_image in range(3):
        axs[i_image, 0].imshow(transform_inv(images[i_image]))
        if i_image == 0:
            axs[i_image, 0].scatter(clicks[:, 1] * w, clicks[:, 0] * h, color='red', s=10)
        for i_bias in range(10):
            axs[i_image, i_bias+1].imshow(masks[i_bias][i_image].cpu().numpy())
            centroid = heatmaps[i_bias][i_image].argmax().item()
            axs[i_image, i_bias+1].scatter(centroid % w, centroid // h, color='green', s=100, marker='+', edgecolor='black')
            if i_image == 0:
                axs[i_image, i_bias+1].set_title(f'bias={bias_factors[i_bias]:.2f}')

    plt.tight_layout()
    plt.show()
# %%
