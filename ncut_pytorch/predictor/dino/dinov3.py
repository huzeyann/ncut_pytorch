URLS = {
    "dinov3_vits16": "https://huggingface.co/huzey/mydv3/resolve/master/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vits16plus": "https://huggingface.co/huzey/mydv3/resolve/master/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "dinov3_vitb16": "https://huggingface.co/huzey/mydv3/resolve/master/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": "https://huggingface.co/huzey/mydv3/resolve/master/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vith16plus": "https://huggingface.co/huzey/mydv3/resolve/master/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    "dinov3_vitl16_sat493m": "https://huggingface.co/huzey/mydv3/resolve/master/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    "dinov3_vitl16_dinotxt": "https://huggingface.co/huzey/mydv3/resolve/master/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
}

import torch
from torch import nn
import os
import subprocess


class Dinov3Backbone(nn.Module):
    def __init__(self, config="dinov3_vitl16"):
        super().__init__()
        # ckpt_path = download_by_wget(URLS[config])
        # dinov3 = torch.hub.load("facebookresearch/dinov3", config, weights=ckpt_path)
        if config == "dinov3_vitl16_sat493m":
            c = "dinov3_vitl16"
        else:
            c = config
        # if config == "dinov3_vitl16_dinotxt":
        #     config = "dinov3_vitl16"
        dinov3 = torch.hub.load("facebookresearch/dinov3", c, weights=URLS[config])
        self.model = dinov3

    def forward(self, x: torch.Tensor):
        return self.model.get_intermediate_layers(x, reshape=True)[0]  # b, c, h, w


def download_by_wget(url):
    # Get the torch hub directory
    torch_hub_dir = torch.hub._get_torch_home()
    cache_dir = os.path.join(torch_hub_dir, 'checkpoints')
    os.makedirs(cache_dir, exist_ok=True)

    model_filename = url.split('/')[-1].split('?')[0]
    output_file = os.path.join(cache_dir, model_filename)

    # Download the file using wget (with quotes around the URL to handle special characters)
    # print(f"Downloading model to {output_file}...")
    subprocess.run(['wget', '-O', output_file, url], check=True)
    return output_file