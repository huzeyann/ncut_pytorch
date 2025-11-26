

<details>

<summary>
Click to expand full code

``` py
class DiNOv2(torch.nn.Module):
```

</summary>

```py linenums="1"

# %%
from einops import rearrange
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch import nn
import numpy as np

N_FRAMES = 600

# %%
class DiNOv2(torch.nn.Module):
    def __init__(self, ver="dinov2_vitb14_reg"):
        super().__init__()
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", ver)
        self.dinov2.requires_grad_(False)
        self.dinov2.eval()
        self.dinov2 = self.dinov2.cuda()
        
        def new_block_forward(self, x: torch.Tensor) -> torch.Tensor:
            def attn_residual_func(x):
                return self.ls1(self.attn(self.norm1(x)))

            def ffn_residual_func(x):
                return self.ls2(self.mlp(self.norm2(x)))

            attn_output = attn_residual_func(x)
            self.attn_output = attn_output.clone()
            x = x + attn_output
            mlp_output = ffn_residual_func(x)
            self.mlp_output = mlp_output.clone()
            x = x + mlp_output
            block_output = x
            self.block_output = block_output.clone()
            return x
        
        setattr(self.dinov2.blocks[0].__class__, "forward", new_block_forward)

    @torch.no_grad()
    def forward(self, x): 

        out = self.dinov2(x)

        attn_outputs, mlp_outputs, block_outputs = [], [], []
        for i, blk in enumerate(self.dinov2.blocks):
            attn_outputs.append(blk.attn_output)
            mlp_outputs.append(blk.mlp_output)
            block_outputs.append(blk.block_output)

        attn_outputs = torch.stack(attn_outputs)
        mlp_outputs = torch.stack(mlp_outputs)
        block_outputs = torch.stack(block_outputs)
        # return attn_outputs, mlp_outputs, block_outputs
        return block_outputs



def transform_images(frames, size=(896, 896)):
    resized = []
    length = len(frames)
    for i in range(length):
        frame = frames[i]
        # image = Image.fromarray((frame * 255).astype(np.uint8))
        image = Image.fromarray(frame)
        image = image.resize(size, Image.ANTIALIAS)
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

    feat_extractor = DiNOv2()

    attn_outputs, mlp_outputs, block_outputs = [], [], []
    for i in range(frames.shape[0]):
        frame = frames[i]
        frame = transform_images([frame])
        block_output = feat_extractor(frame.cuda())
        block_outputs.append(block_output[-1].cpu())
    block_outputs = torch.stack(block_outputs)
    return block_outputs


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
# remove reg tokens
features = features[:, :, 5:].reshape(-1, 64, 64, 768)
# %%
num_nodes = np.prod(features.shape[:-1])
print("Number of nodes:", num_nodes)

# %%
from ncut_pytorch import NCUT, rgb_from_tsne_3d

eigvectors, eigenvalues = NCUT(
    num_eig=100, num_sample=30000, device="cuda:0"
).fit_transform(features.reshape(-1, features.shape[-1]))
# %%
_, rgb = rgb_from_tsne_3d(eigvectors, num_sample=50000, perplexity=500, knn=10, device="cuda:0")


# %%
image_rgb = rgb.reshape(*features.shape[:-1], 3)

import matplotlib.pyplot as plt


frames1 = read_video(video_paths[0])
frames2 = read_video(video_paths[1])
frames3 = read_video(video_paths[2])

save_dir = "/tmp/ncut_video_dinov2/"
import shutil
shutil.rmtree(save_dir, ignore_errors=True)
import os
os.makedirs(save_dir, exist_ok=True)

def resize_image(image, size=(540, 540)):
    image = Image.fromarray(image)
    image = image.resize(size, Image.ANTIALIAS)
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
save_dir = "/tmp/ncut_video_dinov2/"

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
    
make_video_from_images(save_dir, "/workspace/output/ncut_video_dinov2.mp4")
# %%


```

</details>

This video example use image model without temporal dimension

1. extract image feature for every frame, independently

2. concatenate all the image features and compute NCUT

<div  style="text-align: center;">
<video width="100%" controls muted autoplay loop>
  <source src="/images/gallery_gallery_dinov2_video/ncut_video_dinov2_264.mp4" type="video/mp4">
</video>
</div>