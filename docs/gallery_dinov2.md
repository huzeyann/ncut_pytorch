

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
        return attn_outputs, mlp_outputs, block_outputs


def image_dino_feature(
    images, resolution=(448, 448),
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

    feat_extractor = DiNOv2()

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
attn_outputs, mlp_outputs, block_outputs = image_dino_feature(images)
# %%
print(attn_outputs.shape, mlp_outputs.shape, block_outputs.shape)
# remove 1 cls token 4 reg tokens
def reshape_output(outputs):
    from einops import rearrange
    outputs = rearrange(outputs[:, :, 5:, :], "l b (h w) c -> l b h w c", h=32, w=32)
    return outputs
attn_outputs = reshape_output(attn_outputs)
mlp_outputs = reshape_output(mlp_outputs)
block_outputs = reshape_output(block_outputs)
# %%
num_nodes = np.prod(attn_outputs.shape[1:4])


# %%
from ncut_pytorch import NCUT, rgb_from_tsne_3d

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

    plt.suptitle(f"DiNOv2reg layer {i_layer} NCUT spectral-tSNE", fontsize=16)
    # plt.show()
    save_dir = "/workspace/output/gallery/dinov2reg"
    import os
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/dinov2reg_layer_{i_layer}.jpg", bbox_inches="tight")
    plt.close()
    
exit(0)

```

</details>


<div style="text-align: center;">
    <img src="../images/gallery/dinov2reg/dinov2reg_layer_0.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/dinov2reg/dinov2reg_layer_1.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/dinov2reg/dinov2reg_layer_2.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/dinov2reg/dinov2reg_layer_3.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/dinov2reg/dinov2reg_layer_4.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/dinov2reg/dinov2reg_layer_5.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/dinov2reg/dinov2reg_layer_6.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/dinov2reg/dinov2reg_layer_7.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/dinov2reg/dinov2reg_layer_8.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/dinov2reg/dinov2reg_layer_9.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/dinov2reg/dinov2reg_layer_10.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/dinov2reg/dinov2reg_layer_11.jpg" style="width:100%;">
</div>
