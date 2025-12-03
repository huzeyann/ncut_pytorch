

<details>

<summary>
Click to expand full code

``` py
class SAM(torch.nn.Module):
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


class SAM(torch.nn.Module):
    def __init__(self, checkpoint="/data/sam_model/sam_vit_b_01ec64.pth", **kwargs):
        super().__init__(**kwargs)
        from segment_anything import sam_model_registry, SamPredictor
        from segment_anything.modeling.sam import Sam

        sam: Sam = sam_model_registry["vit_b"](checkpoint=checkpoint)

        from segment_anything.modeling.image_encoder import (
            window_partition,
            window_unpartition,
        )

        def new_block_forward(self, x: torch.Tensor) -> torch.Tensor:
            shortcut = x
            x = self.norm1(x)
            # Window partition
            if self.window_size > 0:
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, self.window_size)

            x = self.attn(x)
            # Reverse window partition
            if self.window_size > 0:
                x = window_unpartition(x, self.window_size, pad_hw, (H, W))
            self.attn_output = x.clone()

            x = shortcut + x
            mlp_outout = self.mlp(self.norm2(x))
            self.mlp_output = mlp_outout.clone()
            x = x + mlp_outout
            self.block_output = x.clone()

            return x

        setattr(sam.image_encoder.blocks[0].__class__, "forward", new_block_forward)

        self.image_encoder = sam.image_encoder
        self.image_encoder.eval()
        self.image_encoder = self.image_encoder.cuda()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = torch.nn.functional.interpolate(x, size=(1024, 1024), mode="bilinear")
        out = self.image_encoder(x)

        attn_outputs, mlp_outputs, block_outputs = [], [], []
        for i, blk in enumerate(self.image_encoder.blocks):
            attn_outputs.append(blk.attn_output)
            mlp_outputs.append(blk.mlp_output)
            block_outputs.append(blk.block_output)
            # print(f"block {i} attn_output shape: {blk.attn_output.shape}")
            # print(f"block {i} mlp_output shape: {blk.mlp_output.shape}")
            # print(f"block {i} block_output shape: {blk.block_output.shape}")
        attn_outputs = torch.stack(attn_outputs)
        mlp_outputs = torch.stack(mlp_outputs)
        block_outputs = torch.stack(block_outputs)
        return attn_outputs, mlp_outputs, block_outputs


def image_sam_feature(
    images, resolution=(1024, 1024), checkpoint="/data/sam_model/sam_vit_b_01ec64.pth"
):
    if isinstance(images, list):
        assert isinstance(images[0], Image.Image), "Input must be a list of PIL images."
    else:
        assert isinstance(images, Image.Image), "Input must be a PIL image."
        images = [images]

    transform = transforms.Compose(
        [
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    feat_extractor = SAM(checkpoint=checkpoint)

    attn_outputs, mlp_outputs, block_outputs = [], [], []
    for i, image in enumerate(images):
        torch_image = transform(image)
        # feat = feat_extractor(torch_image.unsqueeze(0).cuda()).cpu()
        attn_output, mlp_output, block_output = feat_extractor(
            torch_image.unsqueeze(0).cuda()
        )
        # feats.append(feat)
        attn_outputs.append(attn_output.cpu())
        mlp_outputs.append(mlp_output.cpu())
        block_outputs.append(block_output.cpu())
    attn_outputs = torch.cat(attn_outputs, dim=1)
    mlp_outputs = torch.cat(mlp_outputs, dim=1)
    block_outputs = torch.cat(block_outputs, dim=1)

    # feats = torch.cat(feats, dim=1)
    # feats = rearrange(feats, "l b c h w -> l b h w c")
    return attn_outputs, mlp_outputs, block_outputs


# %%
from torchvision.datasets import ImageFolder

dataset = ImageFolder("/data/coco/")
print("number of images in the dataset:", len(dataset))
# %%
images = [dataset[i][0] for i in range(20)]
attn_outputs, mlp_outputs, block_outputs = image_sam_feature(images)
# %%
print(attn_outputs.shape, mlp_outputs.shape, block_outputs.shape)
# %%
num_nodes = np.prod(attn_outputs.shape[1:4])


# %%
from ncut_pytorch import NCUT, rgb_from_tsne_3d

i_layer = 9

for i_layer in range(12):

    attn_eig, _ = NCUT(num_eig=100, device="cuda:0").fit_transform(
        attn_outputs[i_layer].reshape(-1, attn_outputs[i_layer].shape[-1])
    )
    _, attn_rgb = rgb_from_tsne_3d(attn_eig, device="cuda:0")
    attn_rgb = attn_rgb.reshape(attn_outputs[i_layer].shape[:3] + (3,))
    mlp_eig, _ = NCUT(num_eig=100, device="cuda:0").fit_transform(
        mlp_outputs[i_layer].reshape(-1, mlp_outputs[i_layer].shape[-1])
    )
    _, mlp_rgb = rgb_from_tsne_3d(mlp_eig, device="cuda:0")
    mlp_rgb = mlp_rgb.reshape(mlp_outputs[i_layer].shape[:3] + (3,))
    block_eig, _ = NCUT(num_eig=100, device="cuda:0").fit_transform(
        block_outputs[i_layer].reshape(-1, block_outputs[i_layer].shape[-1])
    )
    _, block_rgb = rgb_from_tsne_3d(block_eig, device="cuda:0")
    block_rgb = block_rgb.reshape(block_outputs[i_layer].shape[:3] + (3,))

    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(4, 10, figsize=(10, 5))
    for ax in axs.flatten():
        ax.axis("off")
    for i_col in range(10):
        axs[0, i_col].imshow(images[i_col])
        axs[1, i_col].imshow(attn_rgb[i_col])
        axs[2, i_col].imshow(mlp_rgb[i_col])
        axs[3, i_col].imshow(block_rgb[i_col])

    axs[1, 0].set_title("attention layer output", ha="left")
    axs[2, 0].set_title("MLP layer output", ha="left")
    axs[3, 0].set_title("sum of residual stream", ha="left")

    plt.suptitle(f"SAM layer {i_layer} NCUT spectral-tSNE", fontsize=16)
    # plt.show()
    save_dir = "/workspace/output/gallery/sam"
    import os
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/sam_layer_{i_layer}.jpg", bbox_inches="tight")
    plt.close()
    
exit(0)
# %%

```

</details>


<div style="text-align: center;">
    <img src="../images/gallery_gallery_sam/sam_layer_0.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery_gallery_sam/sam_layer_1.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery_gallery_sam/sam_layer_2.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery_gallery_sam/sam_layer_3.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery_gallery_sam/sam_layer_4.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery_gallery_sam/sam_layer_5.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery_gallery_sam/sam_layer_6.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery_gallery_sam/sam_layer_7.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery_gallery_sam/sam_layer_8.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery_gallery_sam/sam_layer_9.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery_gallery_sam/sam_layer_10.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery_gallery_sam/sam_layer_11.jpg" style="width:100%;">
</div>
