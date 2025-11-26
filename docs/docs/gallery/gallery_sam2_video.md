

<details>

<summary>
Click to expand full code

``` py
class SAM2(torch.nn.Module):
```

</summary>

```py linenums="1"

# %%
import os
from einops import repeat, rearrange
import numpy as np
import torch
from PIL import Image
import requests
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch import nn


N_FRAMES = 600


class SAM2(nn.Module):

    def __init__(self):
        super().__init__()
    
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_checkpoint = "/data/sam_model/sam2_hiera_large.pt"
        # model_cfg = "/data/sam_model/sam2_hiera_b+.yaml"
        model_cfg = "sam2_hiera_l"

        device = 'cuda:0'
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

        image_encoder = sam2_model.image_encoder
        image_encoder.eval()
        
        self.image_encoder = image_encoder
        
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.image_encoder(x)
        features = output['vision_features']
        features = rearrange(features, 'b c h w -> b h w c')
        return features
        
# %%


def transform_images(frames, size=(1024, 1024)):
    resized = []
    length = len(frames)
    for i in range(length):
        frame = frames[i]
        # image = Image.fromarray((frame * 255).astype(np.uint8))
        image = Image.fromarray(frame)
        image = image.resize(size, Image.LANCZOS)
        image = np.array(image) / 255.0
        resized.append(np.array(image))
    frames = np.stack(resized, axis=0)
    frames = frames.transpose(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    frames = torch.tensor(frames, dtype=torch.float32)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    return frames


def read_video(video_path: str) -> torch.Tensor:
    try:
        from decord import VideoReader
    except ImportError:
        raise ImportError("Please install the decord library: pip install decord")

    vr = VideoReader(video_path)
    print(f"Total frames: {len(vr)}")
    # frames = vr.get_batch(range(len(vr))).asnumpy()
    lenth = len(vr)
    lenth = N_FRAMES if lenth > N_FRAMES else lenth
    frames = vr.get_batch(np.arange(lenth)).asnumpy()
    # if less than N_FRAMES frames, repeat the last frame
    if lenth < N_FRAMES:
        last_frame = frames[-1]
        for i in range(N_FRAMES - lenth):
            frames = np.append(frames, last_frame.reshape(1, *last_frame.shape), axis=0)
    return frames


@torch.no_grad()
def video_sam_feature(video_path, checkpoint="/data/sam_model/sam_vit_b_01ec64.pth"):
    frames = read_video(video_path)

    feat_extractor = SAM2()

    features = []
    for i in range(frames.shape[0]):
        frame = frames[i]
        frame = transform_images([frame])
        feature = feat_extractor(frame.cuda())
        feature = torch.nn.functional.normalize(feature, dim=-1)
        features.append(feature.cpu())
    features = torch.cat(features, dim=0)
    return features


# %%
video_paths = [
    "/workspace/ego4d_dog_264.mp4",
    "/workspace/ego4d_tractor_264.mp4",
    "/workspace/ego4d_drum_264.mp4",
]
# %%
features = []
for video_path in video_paths:
    features.append(video_sam_feature(video_path))
features = torch.cat(features, dim=0)
# %%
print(features.shape)
# %%
num_nodes = np.prod(features.shape[:-1])
print("Number of nodes:", num_nodes)

# %%
from ncut_pytorch import NCUT, rgb_from_tsne_3d

eigvectors, eigenvalues = NCUT(
    num_eig=100, num_sample=30000, device="cuda:0", normalize_features=False,
).fit_transform(features.reshape(-1, features.shape[-1]))
# %%
_, rgb = rgb_from_tsne_3d(eigvectors, num_sample=50000, perplexity=500, knn=10, device="cuda:0")


# %%
image_rgb = rgb.reshape(*features.shape[:-1], 3)

import matplotlib.pyplot as plt


frames1 = read_video(video_paths[0])
frames2 = read_video(video_paths[1])
frames3 = read_video(video_paths[2])

save_dir = "/tmp/ncut_video_sam2/"
import shutil
shutil.rmtree(save_dir, ignore_errors=True)
import os
os.makedirs(save_dir, exist_ok=True)

def resize_image(image, size=(540, 540)):
    image = Image.fromarray(image)
    image = image.resize(size, Image.LANCZOS)
    image = np.array(image)
    return image

for i_frame in range(0, N_FRAMES):
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax in axes.flatten():
        ax.axis("off")
    
    offsets = [0, N_FRAMES, 2 * N_FRAMES]
    for i, frames in enumerate([frames1, frames2, frames3]):
        axes[0, i].imshow(resize_image(frames[i_frame]))
        offset = offsets[i]
        np_image = image_rgb[i_frame+offset].cpu().numpy()
        np_image = (np_image * 255).astype(np.uint8)
        axes[1, i].imshow(resize_image(np_image))
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/frame_{i_frame:04d}.png")
    # plt.show()
    plt.close()

# %%
# make video
save_dir = "/nfscc/tmp/ncut_video_sam2/"

def make_video_from_images(image_dir, video_path):
    import cv2
    import os

    images = sorted(os.listdir(image_dir))
    frame = cv2.imread(os.path.join(image_dir, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_dir, image)))

    cv2.destroyAllWindows()
    video.release()
    
make_video_from_images(save_dir, "/workspace/output/ncut_video_sam2.mp4")
# %%


```

</details>


This video example use image model without temporal dimension

1. extract image feature for every frame, independently

2. concatenate all the image features and compute NCUT

<div  style="text-align: center;">
<video width="100%" controls muted autoplay loop>
  <source src="/images/gallery_gallery_sam2_video/ncut_video_sam2_264.mp4" type="video/mp4">
</video>
</div>