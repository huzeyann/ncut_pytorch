# Credits: https://github.com/tldr-group/HR-Dv2

# ==================== IMPORTS ====================
import torch
import torch.nn as nn
import numpy as np

from .patch import Patch

from types import MethodType
from typing import Literal, Union

AttentionOptions = Literal["q", "k", "v", "o", "none"]
DINONameOptions = Literal["dino_vitb8", "dino_vits8", "dino_vitb16", "dinov2_vitb14", "dinov2_vitb14_reg"]

# ==================== MODULE ====================


class LowResDINO(nn.Module):
    def __init__(
        self,
        dino_name: DINONameOptions = "dino_vitb8",
        dtype: Union[torch.dtype, int] = torch.float16,
        track_grad: bool = False,
        attention_mask_ratio: float = 0.1,
    ) -> None:
        super().__init__()

        self.dinov2: nn.Module
        if "dinov2" in dino_name:
            hub_path = "facebookresearch/dinov2"
            self.dinov2 = torch.hub.load(hub_path, dino_name, verbose=False)
        elif "dino" in dino_name:
            hub_path = "facebookresearch/dino:main"
            self.dinov2 = torch.hub.load(hub_path, dino_name, verbose=False)
        else:
            raise ValueError(f"Invalid model name: {dino_name}")

        self.dinov2.eval()
        
        if "dinov2" not in dino_name:
            self.dinov2.num_heads = 6  # type: ignore
            self.dinov2.num_register_tokens = 0  # type: ignore

        # If we want to save memory, change to float16
        if type(dtype) == int:
            dtype = torch.float16 if dtype == 16 else torch.float32

        self.dtype = dtype
        if dtype != torch.float32:
            self = self.to(dtype)
        self.track_grad = track_grad  # off by default to save memory

        self.attention_mask_ratio = attention_mask_ratio
        self.patch_last_block(self.dinov2, dino_name)

    def patch_last_block(self, dino_model: nn.Module, dino_name: str) -> None:
        """Patch the final block of the dino model to add attention return code.

        :param dino_model: DINO or DINOv2 model
        :type dino_model: nn.Module
        """
        final_block = dino_model.blocks[-1]  # type: ignore
        attn_block = final_block.attn  # type: ignore
        # hilariously this also works for dino i.e we can patch dino's attn block forward to
        # use the memeory efficienty attn like in dinov2
        attn_block.forward = MethodType(Patch._fix_attn_masking(self.attention_mask_ratio), attn_block)
        if "dinov2" in dino_name:
            final_block.forward = MethodType(Patch._fix_block_forward_dv2(), final_block)  # type: ignore
            dino_model.forward_feats_attn = MethodType(  # type: ignore
                Patch._add_new_forward_features_dv2(), dino_model
            )
        elif "dino" in dino_name:
            for i, blk in enumerate(dino_model.blocks):
                blk.forward = MethodType(Patch._fix_block_forward_dino(), blk)
                attn_block = blk.attn
                attn_block.forward = MethodType(Patch._fix_attn_masking(self.attention_mask_ratio), attn_block)
            final_block.forward = MethodType(Patch._fix_block_forward_dino(), final_block)  # type: ignore
            dino_model.forward_feats_attn = MethodType(  # type: ignore
                Patch._add_new_forward_features_dino(), dino_model
            )
        else:
            for i, blk in enumerate(dino_model.blocks):
                blk.forward = MethodType(Patch._fix_block_forward_dino(), blk)
                attn_block = blk.attn
                attn_block.forward = MethodType(Patch._fix_attn_masking(self.attention_mask_ratio), attn_block)
            final_block.forward = MethodType(Patch._fix_block_forward_dino(), final_block)  # type: ignore
            dino_model.forward_feats_attn = MethodType(  # type: ignore
                Patch._add_new_forward_features_vit(), dino_model
            )

    def _forward_batch(
        self,
        x: torch.Tensor,
        attn_choice: AttentionOptions = "none",
    ) -> torch.Tensor:
        """Feed input img $x through network and get low-res features.

        :param x: batched image tensor, (b, c, h, w)
        :param attn_choice: choice of attention, "none" or "q", "k", "v", "o"
        :return: low-res features, (b, c, h//k, w//k)
        """
        if self.dtype != torch.float32:  # cast (i.e to f16)
            x = x.type(self.dtype)

        out_dict = self.dinov2.forward_feats_attn(x, None, attn_choice)  # type: ignore
        if attn_choice != "none":
            feats, attn = out_dict["x_norm_patchtokens"], out_dict["x_patchattn"]
            features_batch = torch.concat((feats, attn), dim=-1)
        else:
            features_batch = out_dict["x_norm_patchtokens"]

        if self.dtype != torch.float32:  # cast (i.e to f16)
            features_batch = features_batch.type(self.dtype)
            
        b, l, c = features_batch.shape
        hw = np.sqrt(l).astype(int)
        features_batch = features_batch.reshape(b, hw, hw, c)
        features_batch = features_batch.permute(0, 3, 1, 2)
        return features_batch            
    
    def forward(
        self,
        x: torch.Tensor,
        attn_choice: AttentionOptions = "none",
        move_to_cpu: bool = True,
    ) -> torch.Tensor:
        """Feed input img $x through network and get low-res features.

        :param x: batched image tensor, (b, c, h, w)
        :param attn_choice: choice of attention, "none" or "q", "k", "v", "o"
        :return: low-res features, (b, c, h//k, w//k)
        :rtype: torch.Tensor
        """
        if self.track_grad:
            out = self._forward_batch(x, attn_choice)
        else:
            with torch.no_grad():
                out = self._forward_batch(x, attn_choice)
                if move_to_cpu:
                    out = out.cpu()
        return out

    


