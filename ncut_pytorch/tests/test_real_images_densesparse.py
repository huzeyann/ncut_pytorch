# %%

from einops import rearrange
from ncut_pytorch.new_ncut_pytorch import NewNCUT


from typing import List, Tuple
from ncut_pytorch import affinity_from_features, ncut, nystrom_ncut
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt


def plot_eigvecs_dynamic(eigvec1, eigvec2, eigvec3, sample_index, x_2d, n_cols=6, tol=1e-3):
    n_eig = eigvec1.shape[1]

    n_rows = n_eig // n_cols * 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))    
    
    # vmaxvmin_fn = lambda x: max(abs(x.min()), abs(x.max()))
    
    def vmaxvmin_fn(x):
        quantile = torch.tensor([0.01, 0.99])
        quantiles = x.flatten().quantile(quantile)
        return max(abs(quantiles[0]), abs(quantiles[1]))
        
    # plot one row of eigvec1, one row of eigvec2, repeat
    for i in range(n_eig):
        
        ax = axs[i // n_cols * 3, i % n_cols]
        _v = vmaxvmin_fn(eigvec1[:, i])
        sc = ax.scatter(x_2d[:, 0], x_2d[:, 1], c=eigvec1[:, i], cmap="bwr", vmin=-_v, vmax=_v)
        fig.colorbar(sc, ax=ax)
        ax.set_title(f"exact NCUT eig{i}")
            
        ax = axs[i // n_cols * 3 + 1, i % n_cols]
        _v = vmaxvmin_fn(eigvec2[:, i])
        sc = ax.scatter(x_2d[:, 0], x_2d[:, 1], c=eigvec2[:, i], cmap="bwr", vmin=-_v, vmax=_v)
        fig.colorbar(sc, ax=ax)
        ax.set_title(f"KNN nystrom NCUT eig{i}")
        
        ax = axs[i // n_cols * 3 + 2, i % n_cols]
        _v = vmaxvmin_fn(eigvec3[:, i])
        sc = ax.scatter(x_2d[:, 0], x_2d[:, 1], c=eigvec3[:, i], cmap="bwr", vmin=-_v, vmax=_v)
        fig.colorbar(sc, ax=ax)
        ax.set_title(f"original nystrom NCUT eig{i}")
        
    axs[0, 0].scatter(x_2d[sample_index, 0], x_2d[sample_index, 1], c="k", s=10)
    
    return fig

from ncut_pytorch.ncut_pytorch import correct_rotation
def non_appriximation_ncut(features, num_eig, **config_kwargs):
    aff = affinity_from_features(features, **config_kwargs)
    eigvec, eigval = ncut(aff, num_eig)
    eigvec = correct_rotation(eigvec)
    return eigvec, eigval



def densesparse_ncut(features, num_eig, precomputed_sampled_indices, **config_kwargs):
    aff = affinity_from_features(features, **config_kwargs)
    not_sampled_indices = np.setdiff1d(np.arange(features.shape[0]), precomputed_sampled_indices)
    indices = np.concatenate([precomputed_sampled_indices, not_sampled_indices])
    reverse_indices = np.argsort(indices)
    A = aff[precomputed_sampled_indices][:, precomputed_sampled_indices]
    B = aff[precomputed_sampled_indices][:, not_sampled_indices]
    C = aff[not_sampled_indices][:, not_sampled_indices]
    # take top10 knn on each column of B
    mask = torch.full_like(B, True, dtype=torch.bool)
    mask[B.topk(10, dim=0).indices, torch.arange(B.shape[1])] = False
    B[mask] = 0
    C = torch.zeros_like(C)
    C[torch.arange(C.shape[0]), torch.arange(C.shape[1])] = 1
        
    W = torch.cat([torch.cat([A, B], dim=1), torch.cat([B.T, C], dim=1)], dim=0)
    eigvec, eigval = ncut(W, num_eig)
    eigvec = eigvec[reverse_indices]
    eigvec = correct_rotation(eigvec)
    return eigvec, eigval

from ncut_pytorch.new_ncut_pytorch import NewNCUT
def original_nystrom_ncut(features, num_eig, precomputed_sampled_indices, distance="rbf"):
    eigvec, eigval = NewNCUT(num_eig=num_eig, distance=distance).fit_transform(
        features, precomputed_sampled_indices=precomputed_sampled_indices)
    eigvec = correct_rotation(eigvec)
    return eigvec, eigval

from ncut_pytorch.ncut_pytorch import NCUT
def knn_nystrom_ncut(features, num_eig, precomputed_sampled_indices, distance="rbf"):
    eigvec, eigval = NCUT(num_eig=num_eig, knn=10, distance=distance,
                          indirect_connection=False,
                          ).fit_transform(
        features, precomputed_sampled_indices=precomputed_sampled_indices)
    eigvec = correct_rotation(eigvec)
    return eigvec, eigval

from myold_nystrom import nystrom_ncut
from bruteforce_nystrom import force_nystrom_ncut


import torchvision
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def get_voc_images(num_images=10):
    dataset_voc = torchvision.datasets.VOCSegmentation(
        "/data/pascal_voc/",
        year="2012",
        download=True,
        image_set="val",
    )
    print("number of images in the dataset:", len(dataset_voc))

    images = [dataset_voc[i][0] for i in range(num_images)]
    return images

def transform_images(images):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((672, 672)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    images = [transform(img) for img in images]
    images = torch.stack(images)
    return images

from ncut_pytorch.backbone import load_model
from ncut_pytorch.backbone import extract_features
from ncut_pytorch.backbone import resample_clip_positional_embedding

def get_model_features(images: List[Image.Image]):
    images = transform_images(images)
    model = load_model('DiNO(dino_vitb16_448)')
    features = extract_features(images, model, batch_size=1, layer=11, node_type="block")
    # model = load_model('CLIP(ViT-B-16/openai)')
    # resample_clip_positional_embedding(model.model, 224, 224, 16)
    # features = extract_features(images, model, batch_size=2, layer=5, node_type="attn")
    
    return features


if __name__ == "__main__":
    

    images = get_voc_images(num_images=20)
    features = get_model_features(images)
    print(features.shape)
    
    b, h, w, c = features.shape
    features = rearrange(features, "b h w c -> (b h w) c")
    
    
# %%
num_eig = 100

# 00) all sample
precomputed_sampled_indices = torch.arange(0, features.shape[0], 2).long()

# 0) fps sample
# from ncut_pytorch.ncut_pytorch import farthest_point_sampling
# precomputed_sampled_indices = farthest_point_sampling(features, 1000)

# 1) random sample
# n_samples = 1000
# precomputed_sampled_indices = np.random.choice(features.shape[0], n_samples, replace=False)
# precomputed_sampled_indices = torch.tensor(precomputed_sampled_indices).long()

# # 2) manual sample
# precomputed_sampled_indices = torch.arange(0, 84*84*10, 43).long()
# precomputed_sampled_indices += 84*84*5
# 
# precomputed_sampled_indices = torch.arange(0, 14*14*1000, 14*7).long()
# precomputed_sampled_indices = torch.arange(0, 14*14*100, 2).long()

print(precomputed_sampled_indices.shape)

distance = "cosine"
eigvec1, eigval1 = non_appriximation_ncut(features, num_eig, distance=distance)

eigvec2, eigval2 = densesparse_ncut(features, num_eig, precomputed_sampled_indices, distance=distance)

eigvec3, eigval3 = knn_nystrom_ncut(features, num_eig, precomputed_sampled_indices, distance=distance)
# %%
eigvec4, eigval4 = force_nystrom_ncut(
    features,
    precomputed_sampled_indices,
    distance=distance,
    num_eig=num_eig,
    n_inv=-1,
    device="cuda:0",
    chunk_size=8096,
)

# %%
def vmaxvmin_fn(x):
    quantile = torch.tensor([0.02, 0.98])
    x = x.flatten()
    if len(x) > 1000:
        x = x[torch.randperm(len(x))[:1000]]
    quantiles = x.quantile(quantile)
    return max(abs(quantiles[0]), abs(quantiles[1]))

# plot 20 raw eigenvectors
def plot_raw_eigvecs(eigvecs):
    v = vmaxvmin_fn(eigvecs)
    eigvecs = rearrange(eigvecs, "(b h w) n -> b n h w", b=b, h=h, w=w)
    fig, axs = plt.subplots(21, 20, figsize=(20, 20))
    for j in range(20):
        axs[0, j].imshow(images[j])
        axs[0, j].axis("off")
        for i in range(20):
            axs[i + 1, j].imshow(eigvecs[j][i].reshape(h, w), cmap="bwr", vmin=-v, vmax=v)
            axs[i + 1, j].axis("off")
    fig.tight_layout()
    return fig

# plot_raw_eigvecs(eigvec1)
# plt.suptitle("quadrtic NCUT eigvecs")
# plt.tight_layout()
# plt.savefig("/nfscc/figs/ncut/test_real_images_exact_ncut.png")
# plt.close()
plot_raw_eigvecs(eigvec2)
plt.suptitle("densesparse nystrom NCUT eigvecs")
plt.tight_layout()
plt.show()
#%%
# plt.savefig("/nfscc/figs/ncut/test_real_images_densesparse_ncut.png")
# plt.close()
# plot_raw_eigvecs(eigvec3)
# plt.suptitle("(huzheng) knn nystrom NCUT eigvecs")
# plt.tight_layout()
# plt.savefig("/nfscc/figs/ncut/test_real_images_original_nystrom_ncut_huzheng.png")
# plt.close()
plot_raw_eigvecs(eigvec4)
plt.suptitle("burteforce nystrom NCUT eigvecs")
plt.tight_layout()
# plt.savefig("/nfscc/figs/ncut/test_real_images_original_nystrom_ncut_burteforce.png")
# plt.close()
plt.show()
# %%
