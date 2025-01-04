# Author: Huzheng Yang
# %%
import logging
from typing import Literal, Optional, Tuple, Dict, Any
from einops import rearrange
import requests
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import numpy as np
import os
from functools import partial

MODEL_DICT = {}
LAYER_DICT = {}
SD_KEY_DICT = {}
RES_DICT = {}

class SAM2(nn.Module):

    def __init__(self, model_cfg='sam2_hiera_b+',):
        super().__init__()

        try:
            from sam2.build_sam import build_sam2
        except ImportError as e:
            s = f"""
            Import Error: {e}

            Please install segment_anything_2 from https://github.com/facebookresearch/segment-anything-2.git
            pip install git+https://github.com/huzeyann/segment-anything-2.git
            """
            raise ImportError(s)
        
        config_dict = {
            'sam2_hiera_l': ("sam2_hiera_large.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"),
            'sam2_hiera_b+': ("sam2_hiera_base_plus.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"),
            'sam2_hiera_s': ("sam2_hiera_small.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"),
            'sam2_hiera_t': ("sam2_hiera_tiny.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"),
        }
        filename, url = config_dict[model_cfg]
        if not os.path.exists(filename):
            print(f"Downloading {url}")
            r = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(r.content)
        sam2_checkpoint = filename
        
        device = 'cpu'
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

        image_encoder = sam2_model.image_encoder
        image_encoder.eval()
        
        from sam2.modeling.backbones.hieradet import do_pool
        from sam2.modeling.backbones.utils import window_partition, window_unpartition
        def new_forward(self, x: torch.Tensor) -> torch.Tensor:
            shortcut = x  # B, H, W, C
            x = self.norm1(x)

            # Skip connection
            if self.dim != self.dim_out:
                shortcut = do_pool(self.proj(x), self.pool)

            # Window partition
            window_size = self.window_size
            if window_size > 0:
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, window_size)

            # Window Attention + Q Pooling (if stage change)
            x = self.attn(x)
            if self.q_stride:
                # Shapes have changed due to Q pooling
                window_size = self.window_size // self.q_stride[0]
                H, W = shortcut.shape[1:3]

                pad_h = (window_size - H % window_size) % window_size
                pad_w = (window_size - W % window_size) % window_size
                pad_hw = (H + pad_h, W + pad_w)

            # Reverse window partition
            if self.window_size > 0:
                x = window_unpartition(x, window_size, pad_hw, (H, W))

            self.attn_output = x.clone()
            
            x = shortcut + self.drop_path(x)
            # MLP
            mlp_out = self.mlp(self.norm2(x))
            self.mlp_output = mlp_out.clone()
            x = x + self.drop_path(mlp_out)
            self.block_output = x.clone()
            return x        
        
        setattr(image_encoder.trunk.blocks[0].__class__, 'forward', new_forward)
        
        self.image_encoder = image_encoder
        
        
        
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.image_encoder(x)
        attn_outputs, mlp_outputs, block_outputs = [], [], []
        for block in self.image_encoder.trunk.blocks:
            attn_outputs.append(block.attn_output)
            mlp_outputs.append(block.mlp_output)
            block_outputs.append(block.block_output)
        return {
            'attn': attn_outputs,
            'mlp': mlp_outputs,
            'block': block_outputs
        }

MODEL_DICT["SAM2(sam2_hiera_t)"] = partial(SAM2, model_cfg='sam2_hiera_t')
LAYER_DICT["SAM2(sam2_hiera_t)"] = 12
RES_DICT["SAM2(sam2_hiera_t)"] = (1024, 1024)
MODEL_DICT["SAM2(sam2_hiera_s)"] = partial(SAM2, model_cfg='sam2_hiera_s')
LAYER_DICT["SAM2(sam2_hiera_s)"] = 16
RES_DICT["SAM2(sam2_hiera_s)"] = (1024, 1024)
MODEL_DICT["SAM2(sam2_hiera_b+)"] = partial(SAM2, model_cfg='sam2_hiera_b+')
LAYER_DICT["SAM2(sam2_hiera_b+)"] = 24
RES_DICT["SAM2(sam2_hiera_b+)"] = (1024, 1024)
MODEL_DICT["SAM2(sam2_hiera_l)"] = partial(SAM2, model_cfg='sam2_hiera_l')
LAYER_DICT["SAM2(sam2_hiera_l)"] = 48
RES_DICT["SAM2(sam2_hiera_l)"] = (1024, 1024)


class SAM(torch.nn.Module):
    def __init__(self, model_cfg='vit_b', **kwargs):
        super().__init__(**kwargs)
        try:
            from segment_anything import sam_model_registry, SamPredictor
            from segment_anything.modeling.sam import Sam
        except ImportError as e:
            s = f"""
            Import Error: {e}

            Please install segment_anything from https://github.com/facebookresearch/segment-anything.git
            pip install git+https://github.com/facebookresearch/segment-anything.git
            """
            raise ImportError(s)

        # https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
        # https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
        # https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
        config_dict = {
            'vit_h': ("sam_vit_h_4b8939.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"),
            'vit_l': ("sam_vit_l_0b3195.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"),
            'vit_b': ("sam_vit_b_01ec64.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"),
        }
        checkpoint = config_dict[model_cfg][0]
        if not os.path.exists(checkpoint):
            ckpt_url = config_dict[model_cfg][1]
            import requests

            r = requests.get(ckpt_url)
            with open(checkpoint, "wb") as f:
                f.write(r.content)

        sam: Sam = sam_model_registry[model_cfg](checkpoint=checkpoint)

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
        attn_outputs = torch.stack(attn_outputs)
        mlp_outputs = torch.stack(mlp_outputs)
        block_outputs = torch.stack(block_outputs)
        return {
            'attn': attn_outputs,
            'mlp': mlp_outputs,
            'block': block_outputs
        }

MODEL_DICT["SAM(sam_vit_b)"] = partial(SAM)
LAYER_DICT["SAM(sam_vit_b)"] = 12
RES_DICT["SAM(sam_vit_b)"] = (1024, 1024)
MODEL_DICT["SAM(sam_vit_l)"] = partial(SAM, model_cfg='vit_l')
LAYER_DICT["SAM(sam_vit_l)"] = 24
RES_DICT["SAM(sam_vit_l)"] = (1024, 1024)
MODEL_DICT["SAM(sam_vit_h)"] = partial(SAM, model_cfg='vit_h')
LAYER_DICT["SAM(sam_vit_h)"] = 32
RES_DICT["SAM(sam_vit_h)"] = (1024, 1024)



class MobileSAM(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        try:
            from mobile_sam import sam_model_registry
        except ImportError as e:
            s = f"""
            Import Error: {e}

            Please install mobile_sam from https://github.com/ChaoningZhang/MobileSAM.git
            pip install git+https://github.com/ChaoningZhang/MobileSAM.git@c12dd83
            """
            raise ImportError(s)

        url = "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt"
        model_type = "vit_t"
        sam_checkpoint = "mobile_sam.pt"
        if not os.path.exists(sam_checkpoint):
            import requests

            r = requests.get(url)
            with open(sam_checkpoint, "wb") as f:
                f.write(r.content)

        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

        def new_forward_fn(self, x):
            shortcut = x

            x = self.conv1(x)
            x = self.act1(x)

            x = self.conv2(x)
            x = self.act2(x)

            self.attn_output = rearrange(x.clone(), "b c h w -> b h w c")

            x = self.conv3(x)

            self.mlp_output = rearrange(x.clone(), "b c h w -> b h w c")

            x = self.drop_path(x)

            x += shortcut
            x = self.act3(x)

            self.block_output = rearrange(x.clone(), "b c h w -> b h w c")

            return x

        setattr(
            mobile_sam.image_encoder.layers[0].blocks[0].__class__,
            "forward",
            new_forward_fn,
        )

        def new_forward_fn2(self, x):
            H, W = self.input_resolution
            B, L, C = x.shape
            assert L == H * W, "input feature has wrong size"
            res_x = x
            if H == self.window_size and W == self.window_size:
                x = self.attn(x)
            else:
                x = x.view(B, H, W, C)
                pad_b = (self.window_size - H % self.window_size) % self.window_size
                pad_r = (self.window_size - W % self.window_size) % self.window_size
                padding = pad_b > 0 or pad_r > 0

                if padding:
                    x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

                pH, pW = H + pad_b, W + pad_r
                nH = pH // self.window_size
                nW = pW // self.window_size
                # window partition
                x = (
                    x.view(B, nH, self.window_size, nW, self.window_size, C)
                    .transpose(2, 3)
                    .reshape(B * nH * nW, self.window_size * self.window_size, C)
                )
                x = self.attn(x)
                # window reverse
                x = (
                    x.view(B, nH, nW, self.window_size, self.window_size, C)
                    .transpose(2, 3)
                    .reshape(B, pH, pW, C)
                )

                if padding:
                    x = x[:, :H, :W].contiguous()

                x = x.view(B, L, C)

            hw = np.sqrt(x.shape[1]).astype(int)
            self.attn_output = rearrange(x.clone(), "b (h w) c -> b h w c", h=hw)

            x = res_x + self.drop_path(x)

            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.local_conv(x)
            x = x.view(B, C, L).transpose(1, 2)

            mlp_output = self.mlp(x)
            self.mlp_output = rearrange(
                mlp_output.clone(), "b (h w) c -> b h w c", h=hw
            )

            x = x + self.drop_path(mlp_output)
            self.block_output = rearrange(x.clone(), "b (h w) c -> b h w c", h=hw)
            return x

        setattr(
            mobile_sam.image_encoder.layers[1].blocks[0].__class__,
            "forward",
            new_forward_fn2,
        )

        mobile_sam.eval()
        self.image_encoder = mobile_sam.image_encoder

    @torch.no_grad()
    def forward(self, x):
        with torch.no_grad():
            x = torch.nn.functional.interpolate(x, size=(1024, 1024), mode="bilinear")
        out = self.image_encoder(x)

        attn_outputs, mlp_outputs, block_outputs = [], [], []
        for i_layer in range(len(self.image_encoder.layers)):
            for i_block in range(len(self.image_encoder.layers[i_layer].blocks)):
                blk = self.image_encoder.layers[i_layer].blocks[i_block]
                attn_outputs.append(blk.attn_output)
                mlp_outputs.append(blk.mlp_output)
                block_outputs.append(blk.block_output)
        return {
            'attn': attn_outputs,
            'mlp': mlp_outputs,
            'block': block_outputs
        }

MODEL_DICT["MobileSAM(TinyViT)"] = partial(MobileSAM)
LAYER_DICT["MobileSAM(TinyViT)"] = 12
RES_DICT["MobileSAM(TinyViT)"] = (1024, 1024)


class DiNOv2(torch.nn.Module):
    def __init__(self, ver="dinov2_vitb14_reg", num_reg=5):
        super().__init__()
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", ver)
        self.dinov2.requires_grad_(False)
        self.dinov2.eval()
        self.num_reg = num_reg

        def new_block_forward(self, x: torch.Tensor) -> torch.Tensor:
            def attn_residual_func(x):
                return self.ls1(self.attn(self.norm1(x)))

            def ffn_residual_func(x):
                return self.ls2(self.mlp(self.norm2(x)))

            attn_output = attn_residual_func(x)

            hw = np.sqrt(attn_output.shape[1] - num_reg).astype(int)
            self.attn_output = rearrange(
                attn_output.clone()[:, num_reg:], "b (h w) c -> b h w c", h=hw
            )

            x = x + attn_output
            mlp_output = ffn_residual_func(x)
            self.mlp_output = rearrange(
                mlp_output.clone()[:, num_reg:], "b (h w) c -> b h w c", h=hw
            )
            x = x + mlp_output
            block_output = x
            self.block_output = rearrange(
                block_output.clone()[:, num_reg:], "b (h w) c -> b h w c", h=hw
            )
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
        return {
            'attn': attn_outputs,
            'mlp': mlp_outputs,
            'block': block_outputs
        }

MODEL_DICT["DiNOv2reg(dinov2_vits14_reg)"] = partial(DiNOv2, ver="dinov2_vits14_reg", num_reg=5)
LAYER_DICT["DiNOv2reg(dinov2_vits14_reg)"] = 12
RES_DICT["DiNOv2reg(dinov2_vits14_reg)"] = (672, 672)
MODEL_DICT["DiNOv2reg(dinov2_vitb14_reg)"] = partial(DiNOv2, ver="dinov2_vitb14_reg", num_reg=5)
LAYER_DICT["DiNOv2reg(dinov2_vitb14_reg)"] = 12
RES_DICT["DiNOv2reg(dinov2_vitb14_reg)"] = (672, 672)
MODEL_DICT["DiNOv2reg(dinov2_vitl14_reg)"] = partial(DiNOv2, ver="dinov2_vitl14_reg", num_reg=5)
LAYER_DICT["DiNOv2reg(dinov2_vitl14_reg)"] = 24
RES_DICT["DiNOv2reg(dinov2_vitl14_reg)"] = (672, 672)
MODEL_DICT["DiNOv2reg(dinov2_vitg14_reg)"] = partial(DiNOv2, ver="dinov2_vitg14_reg", num_reg=5)
LAYER_DICT["DiNOv2reg(dinov2_vitg14_reg)"] = 40
RES_DICT["DiNOv2reg(dinov2_vitg14_reg)"] = (672, 672)

MODEL_DICT["DiNOv2(dinov2_vits14)"] = partial(DiNOv2, ver="dinov2_vits14", num_reg=1)
LAYER_DICT["DiNOv2(dinov2_vits14)"] = 12
RES_DICT["DiNOv2(dinov2_vits14)"] = (672, 672)
MODEL_DICT["DiNOv2(dinov2_vitb14)"] = partial(DiNOv2, ver="dinov2_vitb14", num_reg=1)
LAYER_DICT["DiNOv2(dinov2_vitb14)"] = 12
RES_DICT["DiNOv2(dinov2_vitb14)"] = (672, 672)
MODEL_DICT["DiNOv2(dinov2_vitl14)"] = partial(DiNOv2, ver="dinov2_vitl14", num_reg=1)
LAYER_DICT["DiNOv2(dinov2_vitl14)"] = 24
RES_DICT["DiNOv2(dinov2_vitl14)"] = (672, 672)
MODEL_DICT["DiNOv2(dinov2_vitg14)"] = partial(DiNOv2, ver="dinov2_vitg14", num_reg=1)
LAYER_DICT["DiNOv2(dinov2_vitg14)"] = 40
RES_DICT["DiNOv2(dinov2_vitg14)"] = (672, 672)

class DVTDistillDiNOv2(DiNOv2):
    def __init__(self, ver="dinov2_vitb14", num_reg=1):
        super().__init__(ver=ver, num_reg=num_reg)
        url = "https://huggingface.co/jjiaweiyang/DVT/resolve/main/imgnet_distilled/vit_base_patch14_dinov2.lvd142m.pth"
        sd = torch.hub.load_state_dict_from_url(url, map_location='cpu')['model']
        # clean up the state dict, remove the prefix 'model.'
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("model."):
                new_sd[k[6:]] = v
            else:
                new_sd[k] = v
        sd = new_sd
        msg = self.dinov2.load_state_dict(sd, strict=False)
        logging.warning(msg)
        

MODEL_DICT["DiNOv2[DVT](dinov2_vitb14_dvt_imgnet_distill)"] = DVTDistillDiNOv2
LAYER_DICT["DiNOv2[DVT](dinov2_vitb14_dvt_imgnet_distill)"] = 12
RES_DICT["DiNOv2[DVT](dinov2_vitb14_dvt_imgnet_distill)"] = (672, 672)


class DiNO(nn.Module):
    def __init__(self, ver="dino_vitb8", save_qkv=False):
        super().__init__()
        model = torch.hub.load('facebookresearch/dino:main', ver)
        model = model.eval()
        self.save_qkv = save_qkv

        def remove_cls_and_reshape(x):
            x = x.clone()
            x = x[:, 1:]
            hw = np.sqrt(x.shape[1]).astype(int)
            x = rearrange(x, "b (h w) c -> b h w c", h=hw)
            return x
        
        def new_forward(self, x, return_attention=False):
            y, attn = self.attn(self.norm1(x))
            self.attn_output = remove_cls_and_reshape(y.clone())
            if return_attention:
                return attn
            x = x + self.drop_path(y)
            mlp_output = self.mlp(self.norm2(x))
            self.mlp_output = remove_cls_and_reshape(mlp_output.clone())
            x = x + self.drop_path(mlp_output)
            self.block_output = remove_cls_and_reshape(x.clone())
            return x

        setattr(model.blocks[0].__class__, "forward", new_forward)
        
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)
        
        if save_qkv:
            self.enable_save_qkv()
    
    def enable_save_qkv(self):
        def remove_cls_and_reshape_qkv(x):
            x = x.clone()
            # x [B, Heads, N, C]
            x = x[:, :, 1:]
            hw = np.sqrt(x.shape[2]).astype(int)
            x = rearrange(x, "b d (h w) c -> b h w d c", h=hw)
            return x

        def new_attn_forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            self.__q = remove_cls_and_reshape_qkv(q.clone())
            self.__k = remove_cls_and_reshape_qkv(k.clone())
            self.__v = remove_cls_and_reshape_qkv(v.clone())

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x, attn
        
        setattr(self.model.blocks[0].attn.__class__, "forward", new_attn_forward)
        self.save_qkv = True
        
    def forward(self, x):
        out = self.model(x)
        attn_outputs = [block.attn_output for block in self.model.blocks]
        mlp_outputs = [block.mlp_output for block in self.model.blocks]
        block_outputs = [block.block_output for block in self.model.blocks]
        if self.save_qkv:
            k = [block.attn.__k for block in self.model.blocks]
            q = [block.attn.__q for block in self.model.blocks]
            v = [block.attn.__v for block in self.model.blocks]
            return {
                'attn': attn_outputs,
                'mlp': mlp_outputs,
                'block': block_outputs,
                'k': k,
                'q': q,
                'v': v,
            }
        return {
            'attn': attn_outputs,
            'mlp': mlp_outputs,
            'block': block_outputs,
        }
            
MODEL_DICT["DiNO(dino_vits8_896)"] = partial(DiNO, ver="dino_vits8")
LAYER_DICT["DiNO(dino_vits8_896)"] = 12
RES_DICT["DiNO(dino_vits8_896)"] = (896, 896)
MODEL_DICT["DiNO(dino_vitb8_896)"] = partial(DiNO)
LAYER_DICT["DiNO(dino_vitb8_896)"] = 12
RES_DICT["DiNO(dino_vitb8_896)"] = (896, 896)

MODEL_DICT["DiNO(dino_vits8_672)"] = partial(DiNO, ver="dino_vits8")
LAYER_DICT["DiNO(dino_vits8_672)"] = 12
RES_DICT["DiNO(dino_vits8_672)"] = (672, 672)
MODEL_DICT["DiNO(dino_vitb8_672)"] = partial(DiNO)
LAYER_DICT["DiNO(dino_vitb8_672)"] = 12
RES_DICT["DiNO(dino_vitb8_672)"] = (672, 672)

MODEL_DICT["DiNO(dino_vits8_448)"] = partial(DiNO, ver="dino_vits8")
LAYER_DICT["DiNO(dino_vits8_448)"] = 12
RES_DICT["DiNO(dino_vits8_448)"] = (448, 448)
MODEL_DICT["DiNO(dino_vitb8_448)"] = partial(DiNO)
LAYER_DICT["DiNO(dino_vitb8_448)"] = 12
RES_DICT["DiNO(dino_vitb8_448)"] = (448, 448)
MODEL_DICT["DiNO(dino_vits16_448)"] = partial(DiNO, ver="dino_vits16")
LAYER_DICT["DiNO(dino_vits16_448)"] = 12
RES_DICT["DiNO(dino_vits16_448)"] = (448, 448)
MODEL_DICT["DiNO(dino_vitb16_448)"] = partial(DiNO, ver="dino_vitb16")
LAYER_DICT["DiNO(dino_vitb16_448)"] = 12
RES_DICT["DiNO(dino_vitb16_448)"] = (448, 448)


class FPNDiNO8plus16(nn.Module):
    def __init__(self, resolutions=[224, 448], interpolate='bilinear', reduce='mean'):
        super().__init__()
        model1 = load_model("DiNO(dino_vitb16_448)").eval()
        model2 = load_model("DiNO(dino_vitb8_448)").eval()
        self.model1 = model1
        self.model2 = model2
        self.resolutions = resolutions
        self.interpolate = interpolate
        self.reduce = reduce
    
    def forward(self, x):
        
        # iterate over different resolutions, extract features from each resolution
        feature_list1 = []
        feature_list2 = []
        for res in self.resolutions:
            _x = F.interpolate(x, size=(res, res), mode='bilinear')
            feature = self.model1(_x)
            feature_list1.append(feature)
            feature = self.model2(_x)
            feature_list2.append(feature)
        
        # combine features from different resolutions, resize to the biggest feature map, and average
        feature_keys = list(feature_list1[0].keys())
        n_layers = len(feature_list1[0][feature_keys[0]])
        combined_features = {key: [] for key in feature_keys}
        for key in feature_keys:
            for i_layer in range(n_layers):
                _features1 = [f[key][i_layer] for f in feature_list1]
                _features2 = [f[key][i_layer] for f in feature_list2]
                max_size = max([f.shape[-2] for f in _features2] + [f.shape[-2] for f in _features1])
                def resize_feat(feat):
                    feat = rearrange(feat, 'b h w c -> b c h w')
                    feat = F.interpolate(feat, size=(max_size, max_size), mode=self.interpolate)
                    feat = rearrange(feat, 'b c h w -> b h w c')
                    return feat
                _features1 = [resize_feat(feat) for feat in _features1]
                _features2 = [resize_feat(feat) for feat in _features2]
                _feature1 = torch.stack(_features1, dim=1).mean(dim=1)
                _feature2 = torch.stack(_features2, dim=1).mean(dim=1)
                if self.reduce == 'stack':
                    _feature = torch.cat([_feature1, _feature2], dim=-1)
                elif self.reduce == 'mean':
                    _feature = (_feature1 + _feature2) / 2
                else:
                    raise ValueError(f"Unsupported reduce method: {self.reduce}")
                combined_features[key].append(_feature)
        
        return combined_features  # return a dictionary of features, {'block': [layer1, layer2, ...], 'attn': ...}


MODEL_DICT["FPNDiNO8plus16[mean][FPN_448]"] = partial(FPNDiNO8plus16, resolutions=[224, 448])
LAYER_DICT["FPNDiNO8plus16[mean][FPN_448]"] = 12
RES_DICT["FPNDiNO8plus16[mean][FPN_448]"] = (448, 448)
MODEL_DICT["FPNDiNO8plus16[mean][FPN_672]"] = partial(FPNDiNO8plus16, resolutions=[224, 448, 672])
LAYER_DICT["FPNDiNO8plus16[mean][FPN_672]"] = 12
RES_DICT["FPNDiNO8plus16[mean][FPN_672]"] = (672, 672)
MODEL_DICT["FPNDiNO8plus16[stack][FPN_448]"] = partial(FPNDiNO8plus16, resolutions=[224, 448], reduce='stack')
LAYER_DICT["FPNDiNO8plus16[stack][FPN_448]"] = 12
RES_DICT["FPNDiNO8plus16[stack][FPN_448]"] = (448, 448)
MODEL_DICT["FPNDiNO8plus16[stack][FPN_672]"] = partial(FPNDiNO8plus16, resolutions=[224, 448, 672], reduce='stack')
LAYER_DICT["FPNDiNO8plus16[stack][FPN_672]"] = 12
RES_DICT["FPNDiNO8plus16[stack][FPN_672]"] = (672, 672)


class FPNDiNO8(nn.Module):
    def __init__(self, resolutions=[224, 448], interpolate='bilinear'):
        super().__init__()
        model1 = load_model("DiNO(dino_vitb8_448)").eval()
        self.model1 = model1
        self.resolutions = resolutions
        self.interpolate = interpolate
    
    def forward(self, x):
        
        # iterate over different resolutions, extract features from each resolution
        feature_list1 = []
        for res in self.resolutions:
            _x = F.interpolate(x, size=(res, res), mode='bilinear')
            feature = self.model1(_x)
            feature_list1.append(feature)

        
        # combine features from different resolutions, resize to the biggest feature map, and average
        feature_keys = list(feature_list1[0].keys())
        n_layers = len(feature_list1[0][feature_keys[0]])
        combined_features = {key: [] for key in feature_keys}
        for key in feature_keys:
            for i_layer in range(n_layers):
                _features1 = [f[key][i_layer] for f in feature_list1]
                max_size = max([f.shape[-2] for f in _features1])
                def resize_feat(feat):
                    feat = rearrange(feat, 'b h w c -> b c h w')
                    feat = F.interpolate(feat, size=(max_size, max_size), mode=self.interpolate)
                    feat = rearrange(feat, 'b c h w -> b h w c')
                    return feat
                _features1 = [resize_feat(feat) for feat in _features1]
                _feature1 = torch.stack(_features1, dim=1).mean(dim=1)
                _feature = _feature1
                combined_features[key].append(_feature)
        
        return combined_features  # return a dictionary of features, {'block': [layer1, layer2, ...], 'attn': ...}


MODEL_DICT["FPNDiNO8[FPN_448]"] = partial(FPNDiNO8, resolutions=[224, 448])
LAYER_DICT["FPNDiNO8[FPN_448]"] = 12
RES_DICT["FPNDiNO8[FPN_448]"] = (448, 448)
MODEL_DICT["FPNDiNO8[FPN_672]"] = partial(FPNDiNO8, resolutions=[224, 448, 672])
LAYER_DICT["FPNDiNO8[FPN_672]"] = 12
RES_DICT["FPNDiNO8[FPN_672]"] = (672, 672)


def resample_position_embeddings(embeddings, h, w):
    cls_embeddings = embeddings[0]
    patch_embeddings = embeddings[1:]  # [14*14, 768]
    hw = np.sqrt(patch_embeddings.shape[0]).astype(int)
    patch_embeddings = rearrange(patch_embeddings, "(h w) c -> c h w", h=hw)
    patch_embeddings = F.interpolate(patch_embeddings.unsqueeze(0), size=(h, w), mode="nearest").squeeze(0)
    patch_embeddings = rearrange(patch_embeddings, "c h w -> (h w) c")
    embeddings = torch.cat([cls_embeddings.unsqueeze(0), patch_embeddings], dim=0)
    return embeddings


def resample_clip_positional_embedding(model, h, w, patch_size):
    h, w = h // patch_size, w // patch_size
    positional_embedding = resample_position_embeddings(model.visual.positional_embedding, h, w)
    model.visual.positional_embedding = nn.Parameter(positional_embedding)
    return model


class OpenCLIPViT(nn.Module):
    def __init__(self, version='ViT-B-16', pretrained='laion2b_s34b_b88k', save_qkv=False):
        super().__init__()
        try:
            import open_clip
        except ImportError as e:
            s = f"""
            Import Error: {e}

            Please install open-clip-torch to use this model.
            pip install open-clip-torch==2.20.0
            """
            raise ImportError(s)
        
        self.save_qkv = save_qkv
        
        
        model, _, _ = open_clip.create_model_and_transforms(version, pretrained=pretrained)
        
        if version == 'ViT-B-16':
            # positional_embedding = resample_position_embeddings(model.visual.positional_embedding, 42, 42)
            self.patch_size = 16
        elif version == 'ViT-L-14':
            # positional_embedding = resample_position_embeddings(model.visual.positional_embedding, 48, 48)
            self.patch_size = 14
        elif version == 'ViT-H-14':
            # positional_embedding = resample_position_embeddings(model.visual.positional_embedding, 48, 48)
            self.patch_size = 14
        else:
            raise ValueError(f"Unsupported version: {version}")
        # model.visual.positional_embedding = nn.Parameter(positional_embedding)
        
        def new_forward(
                self,
                q_x: torch.Tensor,
                k_x: Optional[torch.Tensor] = None,
                v_x: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
        ):
            def remove_cls_and_reshape(x):
                x = x.clone()
                x = x[1:]
                hw = np.sqrt(x.shape[0]).astype(int)
                x = rearrange(x, "(h w) b c -> b h w c", h=hw)
                return x
            
            k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
            v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
                        
            attn_output = self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
            self.attn_output = remove_cls_and_reshape(attn_output.clone())
            x = q_x + self.ls_1(attn_output)
            mlp_output = self.mlp(self.ln_2(x))
            self.mlp_output = remove_cls_and_reshape(mlp_output.clone())
            x = x + self.ls_2(mlp_output)    
            self.block_output = remove_cls_and_reshape(x.clone())
            return x


        setattr(model.visual.transformer.resblocks[0].__class__, "forward", new_forward)
        
        self.model = model
        self.model.eval()
        
    def enable_save_qkv(self):
        def new_forward(
                self,
                q_x: torch.Tensor,
                k_x: Optional[torch.Tensor] = None,
                v_x: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
        ):
            def remove_cls_and_reshape(x):
                x = x.clone()
                x = x[1:]
                hw = np.sqrt(x.shape[0]).astype(int)
                x = rearrange(x, "(h w) b c -> b h w c", h=hw)
                return x
            
            k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
            v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
            
            # dirty patch for q k v, don't work for all cases
            qkv = q_x.clone()
            qkv_w = self.attn.in_proj_weight
            qkv_b = self.attn.in_proj_bias
            w_q, w_k, w_v = qkv_w.chunk(3)
            b_q, b_k, b_v = qkv_b.chunk(3)
            self.q_x = remove_cls_and_reshape(F.linear(qkv, w_q, b_q))
            self.k_x = remove_cls_and_reshape(F.linear(qkv, w_k, b_k))
            self.v_x = remove_cls_and_reshape(F.linear(qkv, w_v, b_v))
            num_heads = self.attn.num_heads
            self.q_x = rearrange(self.q_x, "b h w (d c) -> b h w d c", d=num_heads)
            self.k_x = rearrange(self.k_x, "b h w (d c) -> b h w d c", d=num_heads)
            self.v_x = rearrange(self.v_x, "b h w (d c) -> b h w d c", d=num_heads)
            
            attn_output = self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
            self.attn_output = remove_cls_and_reshape(attn_output.clone())
            x = q_x + self.ls_1(attn_output)
            mlp_output = self.mlp(self.ln_2(x))
            self.mlp_output = remove_cls_and_reshape(mlp_output.clone())
            x = x + self.ls_2(mlp_output)    
            self.block_output = remove_cls_and_reshape(x.clone())
            return x
        
        setattr(self.model.visual.transformer.resblocks[0].__class__, "forward", new_forward)
        self.save_qkv = True
            
    def forward(self, x):
        self.model = resample_clip_positional_embedding(self.model, x.shape[-2], x.shape[-1], self.patch_size)
        out = self.model(x)
        attn_outputs, mlp_outputs, block_outputs = [], [], []
        if self.save_qkv:
            k, q, v = [], [], []
            for block in self.model.visual.transformer.resblocks:
                attn_outputs.append(block.attn_output)
                mlp_outputs.append(block.mlp_output)
                block_outputs.append(block.block_output)
                k.append(block.k_x)
                q.append(block.q_x)
                v.append(block.v_x)
            return {
                'attn': attn_outputs,
                'mlp': mlp_outputs,
                'block': block_outputs,
                'k': k,
                'q': q,
                'v': v,
            }
        else:
            for block in self.model.visual.transformer.resblocks:
                attn_outputs.append(block.attn_output)
                mlp_outputs.append(block.mlp_output)
                block_outputs.append(block.block_output)
            return {
                'attn': attn_outputs,
                'mlp': mlp_outputs,
                'block': block_outputs
            }

MODEL_DICT["CLIP(ViT-B-16/openai)"] = partial(OpenCLIPViT, version='ViT-B-16', pretrained='openai')
LAYER_DICT["CLIP(ViT-B-16/openai)"] = 12
RES_DICT["CLIP(ViT-B-16/openai)"] = (672, 672)
MODEL_DICT["CLIP(ViT-L-14/openai)"] = partial(OpenCLIPViT, version='ViT-L-14', pretrained='openai')
LAYER_DICT["CLIP(ViT-L-14/openai)"] = 24
RES_DICT["CLIP(ViT-L-14/openai)"] = (672, 672)
MODEL_DICT["CLIP(ViT-H-14/openai)"] = partial(OpenCLIPViT, version='ViT-H-14', pretrained='openai')
LAYER_DICT["CLIP(ViT-H-14/openai)"] = 32
RES_DICT["CLIP(ViT-H-14/openai)"] = (672, 672)

MODEL_DICT["CLIP(ViT-B-16/laion2b_s34b_b88k)"] = partial(OpenCLIPViT, version='ViT-B-16', pretrained='laion2b_s34b_b88k')
LAYER_DICT["CLIP(ViT-B-16/laion2b_s34b_b88k)"] = 12
RES_DICT["CLIP(ViT-B-16/laion2b_s34b_b88k)"] = (672, 672)


class CLIPConvnext(nn.Module):
    def __init__(self, version='convnext_base_w_320', pretrained='laion_aesthetic_s13b_b82k'):
        super().__init__()
        try:
            import open_clip
        except ImportError as e:
            s = f"""
            Import Error: {e}

            Please install open-clip-torch to use this model.
            pip install open-clip-torch==2.20.0
            """
            raise ImportError(s)
    
        model, _, _ = open_clip.create_model_and_transforms(version, pretrained=pretrained)
        
        def new_forward(self, x):
            shortcut = x
            x = self.conv_dw(x)
            if self.use_conv_mlp:
                x = self.norm(x)
                x = self.mlp(x)
            else:
                x = x.permute(0, 2, 3, 1)
                x = self.norm(x)
                x = self.mlp(x)
                x = x.permute(0, 3, 1, 2)
            if self.gamma is not None:
                x = x.mul(self.gamma.reshape(1, -1, 1, 1))

            x = self.drop_path(x) + self.shortcut(shortcut)
            self.block_output = rearrange(x.clone(), "b c h w -> b h w c")
            return x

        setattr(model.visual.trunk.stages[0].blocks[0].__class__, "forward", new_forward)
        
        self.model = model
        self.model.eval()
    
    def forward(self, x):
        out = self.model(x)
        block_outputs = []
        for stage in self.model.visual.trunk.stages:
            for block in stage.blocks:
                block_outputs.append(block.block_output)
                shape = block.block_output.shape
                if len(shape) != 4 or shape[1] != shape[2]:
                    raise RuntimeError(f"Unexpected feature shape: {shape}, expected (B, H, W, C), please make sure to install `pip install open-clip-torch==2.20.0`")

        return {
            'block': block_outputs
        }
        
    
MODEL_DICT["CLIP(convnext_base_w_320/laion_aesthetic_s13b_b82k)"] = partial(CLIPConvnext, version='convnext_base_w_320', pretrained='laion_aesthetic_s13b_b82k')
LAYER_DICT["CLIP(convnext_base_w_320/laion_aesthetic_s13b_b82k)"] = 36
RES_DICT["CLIP(convnext_base_w_320/laion_aesthetic_s13b_b82k)"] = (960, 960)
MODEL_DICT["CLIP(convnext_large_d_320/laion2b_s29b_b131k_ft_soup)"] = partial(CLIPConvnext, version='convnext_large_d_320', pretrained='laion2b_s29b_b131k_ft_soup')
LAYER_DICT["CLIP(convnext_large_d_320/laion2b_s29b_b131k_ft_soup)"] = 36
RES_DICT["CLIP(convnext_large_d_320/laion2b_s29b_b131k_ft_soup)"] = (960, 960)
MODEL_DICT["CLIP(convnext_xxlarge/laion2b_s34b_b82k_augreg_soup)"] = partial(CLIPConvnext, version='convnext_xxlarge', pretrained='laion2b_s34b_b82k_augreg_soup')
LAYER_DICT["CLIP(convnext_xxlarge/laion2b_s34b_b82k_augreg_soup)"] = 40
RES_DICT["CLIP(convnext_xxlarge/laion2b_s34b_b82k_augreg_soup)"] = (960, 960)

class EVA02(nn.Module):
    
    def __init__(self, version='eva02_base_patch14_448/mim_in22k_ft_in1k', **kwargs):
        super().__init__(**kwargs)
        
        try:
            import timm
        except ImportError as e:
            s = f"""
            Import Error: {e}

            Please install timm to use this model.
            pip install timm==0.9.2
            """
            raise ImportError(s)
        
        model = timm.create_model(
            version,
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        model = model.eval()

        def new_forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        
            def remove_cls_and_reshape(x):
                x = x.clone()
                x = x[:, 1:]
                hw = np.sqrt(x.shape[1]).astype(int)
                x = rearrange(x, "b (h w) c -> b h w c", h=hw)
                return x
            
            if self.gamma_1 is None:
                attn_output = self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask)
                self.attn_output = remove_cls_and_reshape(attn_output.clone())
                x = x + self.drop_path1(attn_output)
                mlp_output = self.mlp(self.norm2(x))
                self.mlp_output = remove_cls_and_reshape(mlp_output.clone())
                x = x + self.drop_path2(mlp_output)
            else:
                attn_output = self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask)
                self.attn_output = remove_cls_and_reshape(attn_output.clone())
                x = x + self.drop_path1(self.gamma_1 * attn_output)
                mlp_output = self.mlp(self.norm2(x))
                self.mlp_output = remove_cls_and_reshape(mlp_output.clone())
                x = x + self.drop_path2(self.gamma_2 * mlp_output)
            self.block_output = remove_cls_and_reshape(x.clone())
            return x

        setattr(model.blocks[0].__class__, "forward", new_forward)
        
        self.model = model
        
    def forward(self, x):
        out = self.model(x)
        attn_outputs = [block.attn_output for block in self.model.blocks]
        mlp_outputs = [block.mlp_output for block in self.model.blocks]
        block_outputs = [block.block_output for block in self.model.blocks]
        shape = attn_outputs[0].shape
        if len(shape) != 4 or shape[1] != shape[2]:
            raise RuntimeError(f"Unexpected feature shape: {shape}, expected (B, H, W, C), please make sure to install `pip install timm==0.9.2`")
        return {
            'attn': attn_outputs,
            'mlp': mlp_outputs,
            'block': block_outputs
        }
        
MODEL_DICT["CLIP(eva02_base_patch14_448/mim_in22k_ft_in1k)"] = partial(EVA02, version='eva02_base_patch14_448/mim_in22k_ft_in1k')
LAYER_DICT["CLIP(eva02_base_patch14_448/mim_in22k_ft_in1k)"] = 12
RES_DICT["CLIP(eva02_base_patch14_448/mim_in22k_ft_in1k)"] = (448, 448)
MODEL_DICT["CLIP(eva02_large_patch14_448/mim_m38m_ft_in22k_in1k)"] = partial(EVA02, version='eva02_large_patch14_448.mim_m38m_ft_in22k_in1k')
LAYER_DICT["CLIP(eva02_large_patch14_448/mim_m38m_ft_in22k_in1k)"] = 24
RES_DICT["CLIP(eva02_large_patch14_448/mim_m38m_ft_in22k_in1k)"] = (448, 448)

class MAE(nn.Module):
    def __init__(self, size='base', pos_size=(42, 42), save_qkv=False, **kwargs):
        super().__init__(**kwargs)

        try:
            import timm
        except ImportError as e:
            s = f"""
            Import Error: {e}

            Please install timm to use this model.
            pip install timm==0.9.2
            """
            raise ImportError(s)
        
        self.save_qkv = save_qkv
        
        if size == 'base':
            self.mae_encoder = timm.models.vision_transformer.VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12)
            sd = torch.hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
            )
        if size == 'large':
            self.mae_encoder = timm.models.vision_transformer.VisionTransformer(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
            sd = torch.hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth"
            )
        if size == 'huge':
            self.mae_encoder = timm.models.vision_transformer.VisionTransformer(patch_size=14, embed_dim=1280, depth=32, num_heads=16)
            sd = torch.hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth"
            )
            
        checkpoint_model = sd["model"]
        state_dict = self.mae_encoder.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.mae_encoder.load_state_dict(checkpoint_model, strict=False)
        
        # resample the patch embeddings to 56x56, take 896x896 input
        pos_embed = self.mae_encoder.pos_embed[0]
        pos_embed = resample_position_embeddings(pos_embed, *pos_size)
        self.mae_encoder.pos_embed = nn.Parameter(pos_embed.unsqueeze(0))
        self.mae_encoder.img_size = (672, 672)
        self.mae_encoder.patch_embed.img_size = (672, 672)

        self.mae_encoder.requires_grad_(False)
        self.mae_encoder.eval()
                
        def forward(self, x):
            self.saved_attn_node = self.ls1(self.attn(self.norm1(x)))
            x = x + self.saved_attn_node.clone()
            self.saved_mlp_node = self.ls2(self.mlp(self.norm2(x)))
            x = x + self.saved_mlp_node.clone()
            self.saved_block_output = x.clone()
            return x
        
        setattr(self.mae_encoder.blocks[0].__class__, "forward", forward)
        
    def enable_save_qkv(self):
        def remove_cls_and_reshape_qkv(x):
            x = x.clone()
            # x [B, Heads, N, C]
            x = x[:, :, 1:]
            hw = np.sqrt(x.shape[2]).astype(int)
            x = rearrange(x, "b d (h w) c -> b h w d c", h=hw)
            return x

        def new_attn_forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            
            self.__q = remove_cls_and_reshape_qkv(q.clone())
            self.__k = remove_cls_and_reshape_qkv(k.clone())
            self.__v = remove_cls_and_reshape_qkv(v.clone())

            if self.fused_attn:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p,
                )
            else:
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        
        setattr(self.mae_encoder.blocks[0].attn.__class__, "forward", new_attn_forward)        
        self.save_qkv = True
                
    def forward(self, x):
        out = self.mae_encoder.forward(x)
        def remove_cls_and_reshape(x):
            x = x.clone()
            x = x[:, 1:]
            hw = np.sqrt(x.shape[1]).astype(int)
            x = rearrange(x, "b (h w) c -> b h w c", h=hw)
            return x
        
        if self.save_qkv:
            attn_outputs = [remove_cls_and_reshape(block.saved_attn_node) for block in self.mae_encoder.blocks]
            mlp_outputs = [remove_cls_and_reshape(block.saved_mlp_node) for block in self.mae_encoder.blocks]
            block_outputs = [remove_cls_and_reshape(block.saved_block_output) for block in self.mae_encoder.blocks]
            k = [block.attn.__k for block in self.mae_encoder.blocks]
            q = [block.attn.__q for block in self.mae_encoder.blocks]
            v = [block.attn.__v for block in self.mae_encoder.blocks]
            return {
                'attn': attn_outputs,
                'mlp': mlp_outputs,
                'block': block_outputs,
                'k': k,
                'q': q,
                'v': v,
            }
        else:
            attn_outputs = [remove_cls_and_reshape(block.saved_attn_node) for block in self.mae_encoder.blocks]
            mlp_outputs = [remove_cls_and_reshape(block.saved_mlp_node) for block in self.mae_encoder.blocks]
            block_outputs = [remove_cls_and_reshape(block.saved_block_output) for block in self.mae_encoder.blocks]
            return {
                'attn': attn_outputs,
                'mlp': mlp_outputs,
                'block': block_outputs
            }
        
        

MODEL_DICT["MAE(vit_base)"] = partial(MAE, size='base')
LAYER_DICT["MAE(vit_base)"] = 12
RES_DICT["MAE(vit_base)"] = (672, 672)
MODEL_DICT["MAE(vit_large)"] = partial(MAE, size='large')
LAYER_DICT["MAE(vit_large)"] = 24
RES_DICT["MAE(vit_large)"] = (672, 672)
MODEL_DICT["MAE(vit_huge)"] = partial(MAE, size='huge', pos_size=(48, 48))
LAYER_DICT["MAE(vit_huge)"] = 32
RES_DICT["MAE(vit_huge)"] = (672, 672)


class ViTatInit(nn.Module):
    def __init__(self, size='base', pos_size=(42, 42), load_patch_embed=False, load_first_block=False, **kwargs):
        super().__init__(**kwargs)

        try:
            import timm
        except ImportError as e:
            s = f"""
            Import Error: {e}

            Please install timm to use this model.
            pip install timm==0.9.2
            """
            raise ImportError(s)
                
        self.mae_encoder = timm.models.vision_transformer.VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12)
        if load_patch_embed or load_first_block:
            sd = torch.hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
            )
                
            checkpoint_model = sd["model"]
            filter_keys = []
            if load_patch_embed:
                filter_keys += ["patch_embed"]
            if load_first_block:
                filter_keys += ["blocks.0"]
            filtered_sd = {}
            for k, v in checkpoint_model.items():
                if any([f in k for f in filter_keys]):
                    filtered_sd[k] = v
            checkpoint_model = filtered_sd

            # load pre-trained model
            msg = self.mae_encoder.load_state_dict(checkpoint_model, strict=False)
        
        # # resample the patch embeddings to 56x56, take 896x896 input
        # pos_embed = self.mae_encoder.pos_embed[0]
        # pos_embed = resample_position_embeddings(pos_embed, *pos_size)
        # self.mae_encoder.pos_embed = nn.Parameter(pos_embed.unsqueeze(0))
        # self.mae_encoder.img_size = (672, 672)
        # self.mae_encoder.patch_embed.img_size = (672, 672)

        self.mae_encoder.requires_grad_(False)
        self.mae_encoder.eval()
                
        def forward(self, x):
            self.saved_attn_node = self.ls1(self.attn(self.norm1(x)))
            x = x + self.saved_attn_node.clone()
            self.saved_mlp_node = self.ls2(self.mlp(self.norm2(x)))
            x = x + self.saved_mlp_node.clone()
            self.saved_block_output = x.clone()
            return x
        
        setattr(self.mae_encoder.blocks[0].__class__, "forward", forward)
        
                
    def forward(self, x):
        out = self.mae_encoder.forward(x)
        def remove_cls_and_reshape(x):
            x = x.clone()
            x = x[:, 1:]
            hw = np.sqrt(x.shape[1]).astype(int)
            x = rearrange(x, "b (h w) c -> b h w c", h=hw)
            return x
        
        attn_outputs = [remove_cls_and_reshape(block.saved_attn_node) for block in self.mae_encoder.blocks]
        mlp_outputs = [remove_cls_and_reshape(block.saved_mlp_node) for block in self.mae_encoder.blocks]
        block_outputs = [remove_cls_and_reshape(block.saved_block_output) for block in self.mae_encoder.blocks]
        return {
            'attn': attn_outputs,
            'mlp': mlp_outputs,
            'block': block_outputs
        }
        
MODEL_DICT["Rand(vit)"] = ViTatInit
LAYER_DICT["Rand(vit)"] = 12
RES_DICT["Rand(vit)"] = (224, 224)
MODEL_DICT["Rand(vit_trained_patch_embed)"] = partial(ViTatInit, load_patch_embed=True)
LAYER_DICT["Rand(vit_trained_patch_embed)"] = 12
RES_DICT["Rand(vit_trained_patch_embed)"] = (224, 224)
MODEL_DICT["Rand(vit_trained_first_block)"] = partial(ViTatInit, load_first_block=True)
LAYER_DICT["Rand(vit_trained_first_block)"] = 12
RES_DICT["Rand(vit_trained_first_block)"] = (224, 224)
MODEL_DICT["Rand(vit_trained_patch_embed_first_block)"] = partial(ViTatInit, load_patch_embed=True, load_first_block=True)
LAYER_DICT["Rand(vit_trained_patch_embed_first_block)"] = 12
RES_DICT["Rand(vit_trained_patch_embed_first_block)"] = (224, 224)

class ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        try:
            import timm
        except ImportError as e:
            s = f"""
            Import Error: {e}

            Please install timm to use this model.
            pip install timm==0.9.2
            """
            raise ImportError(s)
            
        
        model = timm.create_model(
            'vit_base_patch16_224.augreg2_in21k_ft_in1k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        
        # resample the patch embeddings to 56x56, take 896x896 input
        pos_embed = model.pos_embed[0]
        pos_embed = resample_position_embeddings(pos_embed, 42, 42)
        model.pos_embed = nn.Parameter(pos_embed.unsqueeze(0))
        model.img_size = (672, 672)
        model.patch_embed.img_size = (672, 672)

        model.requires_grad_(False)
        model.eval()
        
        def forward(self, x):
            self.saved_attn_node = self.ls1(self.attn(self.norm1(x)))
            x = x + self.saved_attn_node.clone()
            self.saved_mlp_node = self.ls2(self.mlp(self.norm2(x)))
            x = x + self.saved_mlp_node.clone()
            self.saved_block_output = x.clone()
            return x
        
        setattr(model.blocks[0].__class__, "forward", forward)
        
        self.model = model
        
    def forward(self, x):
        out = self.model(x)
        def remove_cls_and_reshape(x):
            x = x.clone()
            x = x[:, 1:]
            hw = np.sqrt(x.shape[1]).astype(int)
            x = rearrange(x, "b (h w) c -> b h w c", h=hw)
            return x
        
        attn_outputs = [remove_cls_and_reshape(block.saved_attn_node) for block in self.model.blocks]
        mlp_outputs = [remove_cls_and_reshape(block.saved_mlp_node) for block in self.model.blocks]
        block_outputs = [remove_cls_and_reshape(block.saved_block_output) for block in self.model.blocks]
        shape = attn_outputs[0].shape
        if len(shape) != 4 or shape[1] != shape[2]:
            raise RuntimeError(f"Unexpected feature shape: {shape}, expected (B, H, W, C), please make sure to install `pip install timm==0.9.2`")
        return {
            'attn': attn_outputs,
            'mlp': mlp_outputs,
            'block': block_outputs
        }
            
MODEL_DICT["ImageNet(vit_base)"] = partial(ImageNet)
LAYER_DICT["ImageNet(vit_base)"] = 12
RES_DICT["ImageNet(vit_base)"] = (672, 672)


def expand_hw(tensor):
    hw = np.sqrt(tensor.shape[1]).astype(int)
    return rearrange(tensor, 'b (h w) c -> b h w c', h=hw, w=hw)

class StableDiffusion(nn.Module):
    def __init__(self, model_id="stabilityai/stable-diffusion-2", total_time_steps=50, timestep=10, 
                    positive_prompt="", negative_prompt=""):
        super().__init__()
        try:
            from diffusers import DDPMScheduler
            from diffusers import StableDiffusionPipeline
        except ImportError:
            raise ImportError("Please install diffusers to use this class. \n pip install diffusers==0.30.2")
        
        self.timestep = timestep
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        noise_scheduler = DDPMScheduler(num_train_timesteps=total_time_steps)
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, device="cpu")
        
        def new_attn_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        ) -> torch.Tensor:

            # Notice that normalization is always applied before the real computation in the following blocks.
            # 0. Self-Attention
            batch_size = hidden_states.shape[0]

            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.norm_type == "ada_norm_zero":
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif self.norm_type == "ada_norm_single":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            else:
                raise ValueError("Incorrect norm used")

            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            # 1. Prepare GLIGEN inputs
            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

            if self.norm_type == "ada_norm_zero":
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.norm_type == "ada_norm_single":
                attn_output = gate_msa * attn_output

            self.attn1_output = expand_hw(attn_output.clone())
            
            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            # 1.2 GLIGEN Control
            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

            # 3. Cross-Attention
            if self.attn2 is not None:
                if self.norm_type == "ada_norm":
                    norm_hidden_states = self.norm2(hidden_states, timestep)
                elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                    norm_hidden_states = self.norm2(hidden_states)
                elif self.norm_type == "ada_norm_single":
                    # For PixArt norm2 isn't applied here:
                    # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                    norm_hidden_states = hidden_states
                elif self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                else:
                    raise ValueError("Incorrect norm")

                if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                self.attn2_output = expand_hw(attn_output.clone())
                hidden_states = attn_output + hidden_states

            # 4. Feed-forward
            # i2vgen doesn't have this norm 🤷‍♂️
            if self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif not self.norm_type == "ada_norm_single":
                norm_hidden_states = self.norm3(hidden_states)

            if self.norm_type == "ada_norm_zero":
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            if self.norm_type == "ada_norm_single":
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

            
            ff_output = self.ff(norm_hidden_states)

            if self.norm_type == "ada_norm_zero":
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif self.norm_type == "ada_norm_single":
                ff_output = gate_mlp * ff_output
            
            self.ff_output = expand_hw(ff_output.clone())
            
            hidden_states = ff_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            self.block_output = expand_hw(hidden_states.clone())
            
            return hidden_states

        setattr(pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].__class__, "forward", new_attn_forward)

        def new_resnet_forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:

            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)
            
            from einops import rearrange
            self.conv1_output = hidden_states.clone()
            self.conv1_output = rearrange(self.conv1_output, 'b c h w -> b h w c')

            if self.time_emb_proj is not None:
                if not self.skip_time_act:
                    temb = self.nonlinearity(temb)
                temb = self.time_emb_proj(temb)[:, :, None, None]

            if self.time_embedding_norm == "default":
                if temb is not None:
                    hidden_states = hidden_states + temb
                hidden_states = self.norm2(hidden_states)
            elif self.time_embedding_norm == "scale_shift":
                if temb is None:
                    raise ValueError(
                        f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                    )
                time_scale, time_shift = torch.chunk(temb, 2, dim=1)
                hidden_states = self.norm2(hidden_states)
                hidden_states = hidden_states * (1 + time_scale) + time_shift
            else:
                hidden_states = self.norm2(hidden_states)

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            
            self.conv2_output = hidden_states.clone()
            self.conv2_output = rearrange(self.conv2_output, 'b c h w -> b h w c')
            
            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            self.block_output = output_tensor.clone()
            self.block_output = rearrange(self.block_output, 'b c h w -> b h w c')
            return output_tensor

        setattr(pipe.unet.down_blocks[0].resnets[0].__class__, "forward", new_resnet_forward)

        self.pipe = pipe
        self.noise_scheduler = noise_scheduler

    def forward(self, image, timestep=None, positive_prompt="", negative_prompt=""):
        timestep = self.timestep if timestep is None else timestep
        positive_prompt = self.positive_prompt if positive_prompt == "" else positive_prompt
        negative_prompt = self.negative_prompt if negative_prompt == "" else negative_prompt
        
        device = image.device
        self.pipe = self.pipe.to(device)
        
        noise = torch.randn(image.shape).to(device)
        image = self.noise_scheduler.add_noise(image, noise, torch.LongTensor([timestep]).to(device))
        
        prompt_embed, _ = self.pipe.encode_prompt(positive_prompt, device, image.shape[0], negative_prompt)
                
        latent = self.pipe.vae.encode(image).latent_dist.sample()
        
        out = self.pipe.unet(latent, self.noise_scheduler.timesteps[timestep], prompt_embed)
        
        
        def add_prefix(d, prefix):
            if d is None:
                return {}
            if len(d) == 0:
                return {}
            return {f"{prefix}_{k}": v for k, v in d.items()}
        
        
        def get_all_from_tsf_block(tsf_block):
            return {name: getattr(tsf_block, name) for name in ['attn1_output', 'attn2_output', 'ff_output', 'block_output']}
        
        def get_all_from_attentions_block(attentions_block):
            transformer_blocks = getattr(attentions_block, 'transformer_blocks')
            return_dict = {}
            for i in range(len(transformer_blocks)):
                return_dict.update(
                    add_prefix(get_all_from_tsf_block(transformer_blocks[i]), f"tsf_{i}"))
            return return_dict
                            
        
        def get_all_from_resnet_block(resnet_block):
            return {name: getattr(resnet_block, name) for name in ['conv1_output', 'conv2_output', 'block_output']}
        
        def get_all_from_block(block, block_name):
            return_dict = {}
            if block_name == 'attentions':
                block = getattr(block, 'attentions')
                for i in range(len(block)):
                    return_dict.update(
                        add_prefix(get_all_from_attentions_block(block[i]), f"{block_name}_{i}")
                        )
            if block_name == 'resnets':
                block = getattr(block, 'resnets')
                for i in range(len(block)):
                    return_dict.update(
                        add_prefix(get_all_from_resnet_block(block[i]), f"{block_name}_{i}")
                        )
            return return_dict
                
        def get_all_from_blocks(blocks, block_name):
            return_dict = {}
            for i_block, block in enumerate(blocks):
                if hasattr(block, 'attentions'):
                    return_dict.update(
                        add_prefix(get_all_from_block(block, "attentions"), f"{block_name}_{i_block}")
                    )
                if hasattr(block, 'resnets'):
                    return_dict.update(
                        add_prefix(get_all_from_block(block, "resnets"), f"{block_name}_{i_block}")
                    )
            return return_dict
        
        out_dict = {}

        if hasattr(self.pipe.unet, 'down_blocks'):
            out_dict.update(
                get_all_from_blocks(self.pipe.unet.down_blocks, "down")
            )
        if hasattr(self.pipe.unet, 'mid_block'):
            block = self.pipe.unet.mid_block
            if hasattr(block, 'attentions'):
                out_dict.update(
                    add_prefix(get_all_from_block(block, "attentions"), "mid")
                )
            if hasattr(block, 'resnets'):
                out_dict.update(
                    add_prefix(get_all_from_block(block, "resnets"), "mid")
                )
        if hasattr(self.pipe.unet, 'up_blocks'):
            out_dict.update(
                get_all_from_blocks(self.pipe.unet.up_blocks, "up")
            )
        
                        
        # remove '_output' from keys
        out_dict = {k.replace('_output', ''): v for k, v in out_dict.items()}
        # add the layer dimension to be consistent with other models
        out_dict = {k: [v] for k, v in out_dict.items()}
        shape = out_dict[list(out_dict.keys())[0]][0].shape
        if len(shape) != 4 or shape[1] != shape[2]:
            raise RuntimeError(f"Unexpected feature shape: {shape}, expected (B, H, W, C), please make sure to install `pip install diffusers==0.30.2`")
        return out_dict
    
MODEL_DICT["Diffusion(stabilityai/stable-diffusion-2)"] = partial(StableDiffusion, model_id="stabilityai/stable-diffusion-2")
LAYER_DICT["Diffusion(stabilityai/stable-diffusion-2)"] = 8
RES_DICT["Diffusion(stabilityai/stable-diffusion-2)"] = (512, 512)
SD_KEY_DICT["Diffusion(stabilityai/stable-diffusion-2)"] = ['down_0_attentions_0_tsf_0_attn1', 'down_0_attentions_0_tsf_0_attn2', 'down_0_attentions_0_tsf_0_ff', 'down_0_attentions_0_tsf_0_block', 'down_0_attentions_1_tsf_0_attn1', 'down_0_attentions_1_tsf_0_attn2', 'down_0_attentions_1_tsf_0_ff', 'down_0_attentions_1_tsf_0_block', 'down_0_resnets_0_conv1', 'down_0_resnets_0_conv2', 'down_0_resnets_0_block', 'down_0_resnets_1_conv1', 'down_0_resnets_1_conv2', 'down_0_resnets_1_block', 'down_1_attentions_0_tsf_0_attn1', 'down_1_attentions_0_tsf_0_attn2', 'down_1_attentions_0_tsf_0_ff', 'down_1_attentions_0_tsf_0_block', 'down_1_attentions_1_tsf_0_attn1', 'down_1_attentions_1_tsf_0_attn2', 'down_1_attentions_1_tsf_0_ff', 'down_1_attentions_1_tsf_0_block', 'down_1_resnets_0_conv1', 'down_1_resnets_0_conv2', 'down_1_resnets_0_block', 'down_1_resnets_1_conv1', 'down_1_resnets_1_conv2', 'down_1_resnets_1_block', 'down_2_attentions_0_tsf_0_attn1', 'down_2_attentions_0_tsf_0_attn2', 'down_2_attentions_0_tsf_0_ff', 'down_2_attentions_0_tsf_0_block', 'down_2_attentions_1_tsf_0_attn1', 'down_2_attentions_1_tsf_0_attn2', 'down_2_attentions_1_tsf_0_ff', 'down_2_attentions_1_tsf_0_block', 'down_2_resnets_0_conv1', 'down_2_resnets_0_conv2', 'down_2_resnets_0_block', 'down_2_resnets_1_conv1', 'down_2_resnets_1_conv2', 'down_2_resnets_1_block', 'down_3_resnets_0_conv1', 'down_3_resnets_0_conv2', 'down_3_resnets_0_block', 'down_3_resnets_1_conv1', 'down_3_resnets_1_conv2', 'down_3_resnets_1_block', 'mid_attentions_0_tsf_0_attn1', 'mid_attentions_0_tsf_0_attn2', 'mid_attentions_0_tsf_0_ff', 'mid_attentions_0_tsf_0_block', 'mid_resnets_0_conv1', 'mid_resnets_0_conv2', 'mid_resnets_0_block', 'mid_resnets_1_conv1', 'mid_resnets_1_conv2', 'mid_resnets_1_block', 'up_0_resnets_0_conv1', 'up_0_resnets_0_conv2', 'up_0_resnets_0_block', 'up_0_resnets_1_conv1', 'up_0_resnets_1_conv2', 'up_0_resnets_1_block', 'up_0_resnets_2_conv1', 'up_0_resnets_2_conv2', 'up_0_resnets_2_block', 'up_1_attentions_0_tsf_0_attn1', 'up_1_attentions_0_tsf_0_attn2', 'up_1_attentions_0_tsf_0_ff', 'up_1_attentions_0_tsf_0_block', 'up_1_attentions_1_tsf_0_attn1', 'up_1_attentions_1_tsf_0_attn2', 'up_1_attentions_1_tsf_0_ff', 'up_1_attentions_1_tsf_0_block', 'up_1_attentions_2_tsf_0_attn1', 'up_1_attentions_2_tsf_0_attn2', 'up_1_attentions_2_tsf_0_ff', 'up_1_attentions_2_tsf_0_block', 'up_1_resnets_0_conv1', 'up_1_resnets_0_conv2', 'up_1_resnets_0_block', 'up_1_resnets_1_conv1', 'up_1_resnets_1_conv2', 'up_1_resnets_1_block', 'up_1_resnets_2_conv1', 'up_1_resnets_2_conv2', 'up_1_resnets_2_block', 'up_2_attentions_0_tsf_0_attn1', 'up_2_attentions_0_tsf_0_attn2', 'up_2_attentions_0_tsf_0_ff', 'up_2_attentions_0_tsf_0_block', 'up_2_attentions_1_tsf_0_attn1', 'up_2_attentions_1_tsf_0_attn2', 'up_2_attentions_1_tsf_0_ff', 'up_2_attentions_1_tsf_0_block', 'up_2_attentions_2_tsf_0_attn1', 'up_2_attentions_2_tsf_0_attn2', 'up_2_attentions_2_tsf_0_ff', 'up_2_attentions_2_tsf_0_block', 'up_2_resnets_0_conv1', 'up_2_resnets_0_conv2', 'up_2_resnets_0_block', 'up_2_resnets_1_conv1', 'up_2_resnets_1_conv2', 'up_2_resnets_1_block', 'up_2_resnets_2_conv1', 'up_2_resnets_2_conv2', 'up_2_resnets_2_block', 'up_3_attentions_0_tsf_0_attn1', 'up_3_attentions_0_tsf_0_attn2', 'up_3_attentions_0_tsf_0_ff', 'up_3_attentions_0_tsf_0_block', 'up_3_attentions_1_tsf_0_attn1', 'up_3_attentions_1_tsf_0_attn2', 'up_3_attentions_1_tsf_0_ff', 'up_3_attentions_1_tsf_0_block', 'up_3_attentions_2_tsf_0_attn1', 'up_3_attentions_2_tsf_0_attn2', 'up_3_attentions_2_tsf_0_ff', 'up_3_attentions_2_tsf_0_block', 'up_3_resnets_0_conv1', 'up_3_resnets_0_conv2', 'up_3_resnets_0_block', 'up_3_resnets_1_conv1', 'up_3_resnets_1_conv2', 'up_3_resnets_1_block', 'up_3_resnets_2_conv1', 'up_3_resnets_2_conv2', 'up_3_resnets_2_block']

MODEL_DICT["Diffusion(CompVis/stable-diffusion-v1-4)"] = partial(StableDiffusion, model_id="CompVis/stable-diffusion-v1-4")
LAYER_DICT["Diffusion(CompVis/stable-diffusion-v1-4)"] = 8
RES_DICT["Diffusion(CompVis/stable-diffusion-v1-4)"] = (512, 512)
SD_KEY_DICT["Diffusion(CompVis/stable-diffusion-v1-4)"] = ['down_0_attentions_0_tsf_0_attn1', 'down_0_attentions_0_tsf_0_attn2', 'down_0_attentions_0_tsf_0_ff', 'down_0_attentions_0_tsf_0_block', 'down_0_attentions_1_tsf_0_attn1', 'down_0_attentions_1_tsf_0_attn2', 'down_0_attentions_1_tsf_0_ff', 'down_0_attentions_1_tsf_0_block', 'down_0_resnets_0_conv1', 'down_0_resnets_0_conv2', 'down_0_resnets_0_block', 'down_0_resnets_1_conv1', 'down_0_resnets_1_conv2', 'down_0_resnets_1_block', 'down_1_attentions_0_tsf_0_attn1', 'down_1_attentions_0_tsf_0_attn2', 'down_1_attentions_0_tsf_0_ff', 'down_1_attentions_0_tsf_0_block', 'down_1_attentions_1_tsf_0_attn1', 'down_1_attentions_1_tsf_0_attn2', 'down_1_attentions_1_tsf_0_ff', 'down_1_attentions_1_tsf_0_block', 'down_1_resnets_0_conv1', 'down_1_resnets_0_conv2', 'down_1_resnets_0_block', 'down_1_resnets_1_conv1', 'down_1_resnets_1_conv2', 'down_1_resnets_1_block', 'down_2_attentions_0_tsf_0_attn1', 'down_2_attentions_0_tsf_0_attn2', 'down_2_attentions_0_tsf_0_ff', 'down_2_attentions_0_tsf_0_block', 'down_2_attentions_1_tsf_0_attn1', 'down_2_attentions_1_tsf_0_attn2', 'down_2_attentions_1_tsf_0_ff', 'down_2_attentions_1_tsf_0_block', 'down_2_resnets_0_conv1', 'down_2_resnets_0_conv2', 'down_2_resnets_0_block', 'down_2_resnets_1_conv1', 'down_2_resnets_1_conv2', 'down_2_resnets_1_block', 'down_3_resnets_0_conv1', 'down_3_resnets_0_conv2', 'down_3_resnets_0_block', 'down_3_resnets_1_conv1', 'down_3_resnets_1_conv2', 'down_3_resnets_1_block', 'mid_attentions_0_tsf_0_attn1', 'mid_attentions_0_tsf_0_attn2', 'mid_attentions_0_tsf_0_ff', 'mid_attentions_0_tsf_0_block', 'mid_resnets_0_conv1', 'mid_resnets_0_conv2', 'mid_resnets_0_block', 'mid_resnets_1_conv1', 'mid_resnets_1_conv2', 'mid_resnets_1_block', 'up_0_resnets_0_conv1', 'up_0_resnets_0_conv2', 'up_0_resnets_0_block', 'up_0_resnets_1_conv1', 'up_0_resnets_1_conv2', 'up_0_resnets_1_block', 'up_0_resnets_2_conv1', 'up_0_resnets_2_conv2', 'up_0_resnets_2_block', 'up_1_attentions_0_tsf_0_attn1', 'up_1_attentions_0_tsf_0_attn2', 'up_1_attentions_0_tsf_0_ff', 'up_1_attentions_0_tsf_0_block', 'up_1_attentions_1_tsf_0_attn1', 'up_1_attentions_1_tsf_0_attn2', 'up_1_attentions_1_tsf_0_ff', 'up_1_attentions_1_tsf_0_block', 'up_1_attentions_2_tsf_0_attn1', 'up_1_attentions_2_tsf_0_attn2', 'up_1_attentions_2_tsf_0_ff', 'up_1_attentions_2_tsf_0_block', 'up_1_resnets_0_conv1', 'up_1_resnets_0_conv2', 'up_1_resnets_0_block', 'up_1_resnets_1_conv1', 'up_1_resnets_1_conv2', 'up_1_resnets_1_block', 'up_1_resnets_2_conv1', 'up_1_resnets_2_conv2', 'up_1_resnets_2_block', 'up_2_attentions_0_tsf_0_attn1', 'up_2_attentions_0_tsf_0_attn2', 'up_2_attentions_0_tsf_0_ff', 'up_2_attentions_0_tsf_0_block', 'up_2_attentions_1_tsf_0_attn1', 'up_2_attentions_1_tsf_0_attn2', 'up_2_attentions_1_tsf_0_ff', 'up_2_attentions_1_tsf_0_block', 'up_2_attentions_2_tsf_0_attn1', 'up_2_attentions_2_tsf_0_attn2', 'up_2_attentions_2_tsf_0_ff', 'up_2_attentions_2_tsf_0_block', 'up_2_resnets_0_conv1', 'up_2_resnets_0_conv2', 'up_2_resnets_0_block', 'up_2_resnets_1_conv1', 'up_2_resnets_1_conv2', 'up_2_resnets_1_block', 'up_2_resnets_2_conv1', 'up_2_resnets_2_conv2', 'up_2_resnets_2_block', 'up_3_attentions_0_tsf_0_attn1', 'up_3_attentions_0_tsf_0_attn2', 'up_3_attentions_0_tsf_0_ff', 'up_3_attentions_0_tsf_0_block', 'up_3_attentions_1_tsf_0_attn1', 'up_3_attentions_1_tsf_0_attn2', 'up_3_attentions_1_tsf_0_ff', 'up_3_attentions_1_tsf_0_block', 'up_3_attentions_2_tsf_0_attn1', 'up_3_attentions_2_tsf_0_attn2', 'up_3_attentions_2_tsf_0_ff', 'up_3_attentions_2_tsf_0_block', 'up_3_resnets_0_conv1', 'up_3_resnets_0_conv2', 'up_3_resnets_0_block', 'up_3_resnets_1_conv1', 'up_3_resnets_1_conv2', 'up_3_resnets_1_block', 'up_3_resnets_2_conv1', 'up_3_resnets_2_conv2', 'up_3_resnets_2_block']

class StableDiffusion3(nn.Module):
    def __init__(self, total_time_steps=50, timestep=10, return_flat_dict=True):
        super().__init__()
        try:
            from diffusers import StableDiffusion3Pipeline
        except ImportError:
            raise ImportError("Please install the diffusers package by running `pip install diffusers==0.30.2`")
        
        access_token = os.getenv("HF_ACCESS_TOKEN")
        if access_token is None:
            raise ValueError("HF_ACCESS_TOKEN environment variable must be set")
        
        
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", token=access_token)
        
        from diffusers.models.attention import _chunked_feed_forward
        def new_forward(
            self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor
        ):
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

            if self.context_pre_only:
                norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
            else:
                norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                    encoder_hidden_states, emb=temb
                )

            # Attention.
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
            )

            # Process attention outputs for the `hidden_states`.
            attn_output = gate_msa.unsqueeze(1) * attn_output
            
            self.attn_output = expand_hw(attn_output.clone())
            
            hidden_states = hidden_states + attn_output

            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
            else:
                ff_output = self.ff(norm_hidden_states)
            ff_output = gate_mlp.unsqueeze(1) * ff_output
            
            self.mlp_output = expand_hw(ff_output.clone())

            hidden_states = hidden_states + ff_output

            # Process attention outputs for the `encoder_hidden_states`.
            if self.context_pre_only:
                encoder_hidden_states = None
            else:
                context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
                encoder_hidden_states = encoder_hidden_states + context_attn_output

                norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
                norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
                if self._chunk_size is not None:
                    # "feed_forward_chunk_size" can be used to save memory
                    context_ff_output = _chunked_feed_forward(
                        self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                    )
                else:
                    context_ff_output = self.ff_context(norm_encoder_hidden_states)
                encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

            self.block_output = expand_hw(hidden_states.clone())
            
            return encoder_hidden_states, hidden_states

        setattr(pipe.transformer.transformer_blocks[0].__class__, 'forward', new_forward)     
        
        from diffusers import DDPMScheduler
        
        self.timestep = timestep
        noise_scheduler = DDPMScheduler(num_train_timesteps=total_time_steps)
        
        self.noise_scheduler = noise_scheduler
        self.timestep = timestep
        self.vae = pipe.vae
        self.transformer = pipe.transformer
        
        self.return_flat_dict = return_flat_dict
        
    def forward(self, image, timestep=None, return_flat_dict=False):
        timestep = self.timestep if timestep is None else timestep
        device = image.device
        self.vae = self.vae.to(device)
        self.transformer = self.transformer.to(device)
        
        bsz = image.shape[0]
        encoder_hidden_states = torch.zeros(bsz, 1, 4096).to(device)
        pooled_projections = torch.zeros(bsz, 2048).to(device)
        
        noise = torch.randn(image.shape).to(device)
        image = self.noise_scheduler.add_noise(image, noise, torch.LongTensor([timestep]).to(device))
        
        latent = self.vae.encode(image).latent_dist.sample()
        
        out = self.transformer(latent, 
                            timestep=self.noise_scheduler.timesteps[timestep].unsqueeze(0).to(device), 
                            pooled_projections=pooled_projections, encoder_hidden_states=encoder_hidden_states)
        
        attn_outputs = [block.attn_output for block in self.transformer.transformer_blocks]
        mlp_outputs = [block.mlp_output for block in self.transformer.transformer_blocks]
        block_outputs = [block.block_output for block in self.transformer.transformer_blocks]
        out_dict = {
            'attn': attn_outputs,
            'mlp': mlp_outputs,
            'block': block_outputs
        }
        shape = out_dict['attn'][0].shape
        if len(shape) != 4 or shape[1] != shape[2]:
            raise RuntimeError(f"Unexpected feature shape: {shape}, expected (B, H, W, C), please make sure to install `pip install diffusers==0.30.2`")
        
        if self.return_flat_dict or return_flat_dict:
#            out_dict = {f'{k}_{i}': [v] for k, values in out_dict.items() for i, v in enumerate(values)}
            out_dict = {}
            for i_layer in range(len(attn_outputs)):
                out_dict[f'attn_{i_layer}'] = [attn_outputs[i_layer]]
                out_dict[f'mlp_{i_layer}'] = [mlp_outputs[i_layer]]
                out_dict[f'block_{i_layer}'] = [block_outputs[i_layer]]
        
        return out_dict

MODEL_DICT["Diffusion(stabilityai/stable-diffusion-3-medium-diffusers)"] = partial(StableDiffusion3, total_time_steps=50, timestep=10)
LAYER_DICT["Diffusion(stabilityai/stable-diffusion-3-medium-diffusers)"] = 24
RES_DICT["Diffusion(stabilityai/stable-diffusion-3-medium-diffusers)"] = (1024, 1024)
SD_KEY_DICT["Diffusion(stabilityai/stable-diffusion-3-medium-diffusers)"] = ['attn_0', 'mlp_0', 'block_0', 'attn_1', 'mlp_1', 'block_1', 'attn_2', 'mlp_2', 'block_2', 'attn_3', 'mlp_3', 'block_3', 'attn_4', 'mlp_4', 'block_4', 'attn_5', 'mlp_5', 'block_5', 'attn_6', 'mlp_6', 'block_6', 'attn_7', 'mlp_7', 'block_7', 'attn_8', 'mlp_8', 'block_8', 'attn_9', 'mlp_9', 'block_9', 'attn_10', 'mlp_10', 'block_10', 'attn_11', 'mlp_11', 'block_11', 'attn_12', 'mlp_12', 'block_12', 'attn_13', 'mlp_13', 'block_13', 'attn_14', 'mlp_14', 'block_14', 'attn_15', 'mlp_15', 'block_15', 'attn_16', 'mlp_16', 'block_16', 'attn_17', 'mlp_17', 'block_17', 'attn_18', 'mlp_18', 'block_18', 'attn_19', 'mlp_19', 'block_19', 'attn_20', 'mlp_20', 'block_20', 'attn_21', 'mlp_21', 'block_21', 'attn_22', 'mlp_22', 'block_22', 'attn_23', 'mlp_23', 'block_23']




class LISA(nn.Module):

    def __init__(self, prompt_str='where is the object?'):
        super().__init__()
        try :
            import lisa
        except ImportError:
            raise ImportError("Please install lisa from \n `pip install git+https://github.com/huzeyann/LISA.git`")
            
        from transformers import AutoTokenizer
        from lisa.model.LISA import LISAForCausalLM

        version = "xinlai/LISA-7B-v1"
        model_max_length = 512
        tokenizer = AutoTokenizer.from_pretrained(
            version,
            cache_dir=None,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.unk_token
        seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

        vision_tower = "openai/clip-vit-large-patch14"
        model = LISAForCausalLM.from_pretrained(
            version, low_cpu_mem_usage=True, vision_tower=vision_tower, seg_token_idx=seg_token_idx,
            torch_dtype=torch.bfloat16,
        )

        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        model.get_model().initialize_vision_modules(model.get_model().config)
        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16)

        model = model.to(dtype=torch.bfloat16)

        def expand_hw(tensor):
            hw = np.sqrt(tensor.shape[-2]).astype(int)
            return rearrange(tensor, "b (h w) c -> b h w c", h=hw)

        def new_forward(
            self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
        ) -> Tuple[Tensor, Tensor]:
            self.input_keys = expand_hw(keys.clone())
            # print("forward", queries.shape, keys.shape, query_pe.shape, key_pe.shape)
            # Self attention block
            if self.skip_first_layer_pe:
                queries = self.self_attn(q=queries, k=queries, v=queries)
            else:
                q = queries + query_pe
                attn_out = self.self_attn(q=q, k=q, v=queries)
                queries = queries + attn_out
            queries = self.norm1(queries)

            # Cross attention block, tokens attending to image embedding
            q = queries + query_pe
            k = keys + key_pe
            attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
            queries = queries + attn_out
            queries = self.norm2(queries)

            # MLP block
            mlp_out = self.mlp(queries)
            queries = queries + mlp_out
            queries = self.norm3(queries)

            # Cross attention block, image embedding attending to tokens
            q = queries + query_pe
            k = keys + key_pe
            attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
            
            self.attn_output = expand_hw(attn_out.clone())
            
            keys = keys + attn_out
            keys = self.norm4(keys)
            
            self.block_output = expand_hw(keys.clone())
            # print("forward, block_output", queries.shape, keys.shape)


            return queries, keys

        setattr(model.model.visual_model.mask_decoder.transformer.layers[0].__class__, "forward", new_forward)
        setattr(model.model.visual_model.mask_decoder.transformer.layers[0].__class__, "__call__", new_forward)

        import math
        def new_final_forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            # Input projections
            q = self.q_proj(q)
            k = self.k_proj(k)
            v = self.v_proj(v)

            # Separate into heads
            q = self._separate_heads(q, self.num_heads)
            k = self._separate_heads(k, self.num_heads)
            v = self._separate_heads(v, self.num_heads)

            # Attention
            _, _, _, c_per_head = q.shape
            attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
            attn = attn / math.sqrt(c_per_head)
            attn = torch.softmax(attn, dim=-1)

            # Get output
            out = attn @ v
            out = self._recombine_heads(out)
            out = self.out_proj(out)
            
            self.attn_output = out.clone()
            # print("final_forward", q.shape, k.shape, v.shape, out.shape)

            return out

        setattr(model.model.visual_model.mask_decoder.transformer.final_attn_token_to_image.__class__, "forward", new_final_forward)
        setattr(model.model.visual_model.mask_decoder.transformer.final_attn_token_to_image.__class__, "__call__", new_final_forward)
        
        self.model = model
        self.tokenizer = tokenizer
        self.vision_tower = vision_tower
        self.prompt_str = prompt_str
        
    def forward(self, images, input_str=None):
        
        from lisa.model.llava import conversation as conversation_lib
        from lisa.model.llava.mm_utils import tokenizer_image_token
        from lisa.utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

        input_str = input_str if input_str is not None else self.prompt_str

        # Model Inference
        conv = conversation_lib.conv_templates['llava_v1'].copy()
        conv.messages = []

        prompt = input_str
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        use_mm_start_end = True
        if use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()


        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(images.device)
        # print("input_ids", input_ids.shape, input_ids)

        # resize to 224
        image_clip = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        image_clip = image_clip.bfloat16()

        image = images.bfloat16()
        
        
        resize_list = [(1024, 1024)]
        original_size_list = [(1024, 1024)]
        
        outs = {}
        for i_image in range(image.shape[0]):
            output_ids, pred_masks = self.model.evaluate(
                image_clip[i_image].unsqueeze(0),
                image[i_image].unsqueeze(0),
                input_ids,
                resize_list,
                original_size_list,
                max_new_tokens=512,
                tokenizer=self.tokenizer,
            )
            output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

            text_output = self.tokenizer.decode(output_ids, skip_special_tokens=False)
            text_output = text_output.replace("\n", "").replace("  ", " ")
            text_output = text_output.split("ASSISTANT: ")[-1]
            
            num_layers = len(self.model.model.visual_model.mask_decoder.transformer.layers)
            for i_layer in range(num_layers):
                layer = self.model.model.visual_model.mask_decoder.transformer.layers[i_layer]
                if i_image == 0:
                    outs[f"dec_{i_layer}_input"] = []
                    outs[f"dec_{i_layer}_attn"] = []
                    outs[f"dec_{i_layer}_block"] = []
                outs[f"dec_{i_layer}_input"].append(layer.input_keys.clone())
                outs[f"dec_{i_layer}_attn"].append(layer.attn_output.clone())
                outs[f"dec_{i_layer}_block"].append(layer.block_output.clone())
                # print(f"Layer {i_layer} done")
                # print(f"shape: {layer.attn_output.shape}")
        outs = {k: [torch.cat(v, 0)] for k, v in outs.items()}
        return outs

MODEL_DICT["LISA(xinlai/LISA-7B-v1)"] = partial(LISA, prompt_str='where is the object?')
LAYER_DICT["LISA(xinlai/LISA-7B-v1)"] = 2
RES_DICT["LISA(xinlai/LISA-7B-v1)"] = (1024, 1024)


def download_all_models():
    for model_name in MODEL_DICT:
        print(f"Downloading {model_name}")
        try:
            model = MODEL_DICT[model_name]()
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")
            continue
        
def get_demo_model_names():
    # for my huggingface demo
    return ['SAM2(sam2_hiera_t)', 'SAM2(sam2_hiera_s)', 'SAM2(sam2_hiera_b+)', 'SAM2(sam2_hiera_l)', 'SAM(sam_vit_b)', 'MobileSAM(TinyViT)', 'DiNOv2reg(dinov2_vitb14_reg)', 'DiNOv2(dinov2_vitb14)', 'DiNO(dino_vitb8)', 'CLIP(ViT-B-16/openai)', 'CLIP(ViT-B-16/laion2b_s34b_b88k)', 'CLIP(eva02_base_patch14_448/mim_in22k_ft_in1k)', 'CLIP(convnext_base_w_320/laion_aesthetic_s13b_b82k)', 'MAE(vit_base)', 'ImageNet(vit_base)']
    # list_of_models = []
    # for model_name in MODEL_DICT:
    #     if LAYER_DICT[model_name] != "not sure":
    #         list_of_models.append(model_name)
    # return list_of_models

MODEL_NAMES = list(MODEL_DICT.keys())
        
def list_models():
    return MODEL_NAMES

def load_model(model_name, quite=False):
    if model_name not in MODEL_DICT:
        raise ValueError(f"Model `{model_name}` not found. Please choose from: {MODEL_NAMES}")
    model = MODEL_DICT[model_name]()
    resolution = RES_DICT[model_name]
    if not quite:
        print(f"Loaded {model_name}, please use input resolution: {resolution}")
    return model


@torch.no_grad()
def extract_features(images: torch.Tensor, model: nn.Module, 
                    node_type: Literal["attn", "mlp", 'block'] = 'block', layer=-1, 
                    batch_size=8, use_cuda=True, device='cuda:0'):
    use_cuda = torch.cuda.is_available() and use_cuda
    
    if use_cuda:
        model = model.to(device)
        
    chunked_idxs = torch.split(torch.arange(images.shape[0]), batch_size)
    
    outputs = []
    for idxs in chunked_idxs:
        inp = images[idxs]
        if use_cuda:
            inp = inp.to(device)
        out = model(inp)  
        # out: {'attn': [B, H, W, C], 'mlp': [B, H, W, C], 'block': [B, H, W, C]}
        try:
            out = out[node_type]
        except KeyError:
            raise ValueError(f"Node type {node_type} not found in model.")
        out = out[layer]
        # normalize before save
        out = F.normalize(out, dim=-1)
        outputs.append(out.cpu().float())
    outputs = torch.cat(outputs, dim=0)

    return outputs

if __name__ == "__main__":
    model = DVTDistillDiNOv2()
    inp = torch.randn(2, 3, 224, 224)
    out = model(inp)
    print(out.keys())
    print(out['attn'][0].shape)

