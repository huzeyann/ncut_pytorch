

<details>

<summary>
Click to expand full code

``` py
class MAE(torch.nn.Module):
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

import timm

# %%
class MAE(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, **kwargs):
        super(MAE, self).__init__(**kwargs)

        sd = torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
        )

        checkpoint_model = sd["model"]
        state_dict = self.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        self.requires_grad_(False)
        self.eval()
        self.cuda()
        
        def forward(self, x):
            self.saved_attn_node = self.ls1(self.attn(self.norm1(x)))
            x = x + self.saved_attn_node.clone()
            self.saved_mlp_node = self.ls2(self.mlp(self.norm2(x)))
            x = x + self.saved_mlp_node.clone()
            self.saved_block_output = x.clone()
            return x
        
        setattr(self.blocks[0].__class__, "forward", forward)
        
    def forward(self, x):
        out = super().forward(x)
        attn_nodes = [block.saved_attn_node for block in self.blocks]
        mlp_nodes = [block.saved_mlp_node for block in self.blocks]
        block_outputs = [block.saved_block_output for block in self.blocks]
        attn_nodes = torch.stack(attn_nodes, dim=0)
        mlp_nodes = torch.stack(mlp_nodes, dim=0)
        block_outputs = torch.stack(block_outputs, dim=0)
        return attn_nodes, mlp_nodes, block_outputs

def image_mae_feature(
    images, resolution=(224, 224),
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

    feat_extractor = MAE()

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
attn_outputs, mlp_outputs, block_outputs = image_mae_feature(images)
# %%
print(attn_outputs.shape, mlp_outputs.shape, block_outputs.shape)
# %%
# remove 1 cls token
def reshape_output(outputs):
    from einops import rearrange
    outputs = rearrange(outputs[:, :, 1:, :], "l b (h w) c -> l b h w c", h=14, w=14)
    return outputs
attn_outputs = reshape_output(attn_outputs)
mlp_outputs = reshape_output(mlp_outputs)
block_outputs = reshape_output(block_outputs)
print(attn_outputs.shape, mlp_outputs.shape, block_outputs.shape)
# %%
num_nodes = np.prod(attn_outputs.shape[1:4])


# %%
from ncut_pytorch import NCUT, rgb_from_tsne_3d

for i_layer in range(12):

    attn_eig, _ = NCUT(num_eig=100, device="cuda:0").fit_transform(
        attn_outputs[i_layer].reshape(-1, attn_outputs[i_layer].shape[-1])
    )
    _, attn_rgb = rgb_from_tsne_3d(attn_eig, device="cuda:0", seed=42)
    attn_rgb = attn_rgb.reshape(attn_outputs[i_layer].shape[:3] + (3,))
    mlp_eig, _ = NCUT(num_eig=100, device="cuda:0").fit_transform(
        mlp_outputs[i_layer].reshape(-1, mlp_outputs[i_layer].shape[-1])
    )
    _, mlp_rgb = rgb_from_tsne_3d(mlp_eig, device="cuda:0", seed=42)
    mlp_rgb = mlp_rgb.reshape(mlp_outputs[i_layer].shape[:3] + (3,))
    block_eig, _ = NCUT(num_eig=100, device="cuda:0").fit_transform(
        block_outputs[i_layer].reshape(-1, block_outputs[i_layer].shape[-1])
    )
    _, block_rgb = rgb_from_tsne_3d(block_eig, device="cuda:0", seed=42)
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

    plt.suptitle(f"MAE layer {i_layer} NCUT spectral-tSNE", fontsize=16)
    # plt.show()
    save_dir = "/workspace/output/gallery/mae"
    import os
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/mae_layer_{i_layer}.jpg", bbox_inches="tight")
    plt.close()
    
exit(0)

# %%

```

</details>


<div style="text-align: center;">
    <img src="../images/gallery/mae/mae_layer_0.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/mae/mae_layer_1.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/mae/mae_layer_2.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/mae/mae_layer_3.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/mae/mae_layer_4.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/mae/mae_layer_5.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/mae/mae_layer_6.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/mae/mae_layer_7.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/mae/mae_layer_8.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/mae/mae_layer_9.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/mae/mae_layer_10.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/mae/mae_layer_11.jpg" style="width:100%;">
</div>
