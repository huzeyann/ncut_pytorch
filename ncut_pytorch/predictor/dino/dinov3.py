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


class Dinov3Backbone(nn.Module):
    def __init__(self, config="dinov3_vitl16"):
        super().__init__()
        if config == "dinov3_vitl16_sat493m":
            config = "dinov3_vitl16"
        dinov3 = torch.hub.load("facebookresearch/dinov3", config, weights=URLS[config])
        self.model = dinov3
        
        self.keep_idx = None

    def forward(self, x: torch.Tensor):
        out = self.model.get_intermediate_layers(x, reshape=True)[0]  # b, c, h, w
        if self.keep_idx is None:
            self.keep_idx = self.remove_high_variance_channels(out)
        out = out[:, self.keep_idx, :, :]
        return out
    
    def remove_high_variance_channels(self, out: torch.Tensor, n_remove: int = 8):
        _out = out.permute(0, 2, 3, 1)
        _out = _out.reshape(-1, _out.shape[-1])
        var = torch.var(_out, dim=0)
        var_sorted_idx = torch.argsort(var, descending=True)
        keep_idx = var_sorted_idx[n_remove:]
        return keep_idx
