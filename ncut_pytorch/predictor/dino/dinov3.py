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
from types import MethodType


class Dinov3Backbone(nn.Module):
    def __init__(self, config="dinov3_vitl16"):
        super().__init__()
        if config == "dinov3_vitl16_sat493m":
            config = "dinov3_vitl16"
        dinov3 = torch.hub.load("facebookresearch/dinov3", config, weights=URLS[config])
        self.model = dinov3
        # self.apply_attention_mask(attention_mask_ratio=0.99)
        
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
    
    def apply_attention_mask(self, attention_mask_ratio: float = 0.1):
        def my_compute_attention(self, qkv: torch.Tensor, attn_bias=None, rope=None) -> torch.Tensor:
            assert attn_bias is None
            B, N, _ = qkv.shape
            C = self.qkv.in_features

            qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
            q, k, v = torch.unbind(qkv, 2)
            q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
            if rope is not None:
                q, k = self.apply_rope(q, k, rope)
            if attention_mask_ratio > 0 and attention_mask_ratio < 1:
                num_keys = int(N * attention_mask_ratio)
                mask_indices = torch.randperm(N-5)[:num_keys] + 5  # add the 5 register tokens
                masked_k = k[:, :, mask_indices, :]
                masked_v = v[:, :, mask_indices, :]
                x = torch.nn.functional.scaled_dot_product_attention(q, masked_k, masked_v)
            else:
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = x.transpose(1, 2)
            return x.reshape([B, N, C])
        
        for blk in self.model.blocks:
            blk.attn.compute_attention = MethodType(my_compute_attention, blk.attn)