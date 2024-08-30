# Author: Huzheng Yang
# %%
from typing import Literal, Optional, Tuple
from einops import rearrange
import requests
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os
from functools import partial

MODEL_DICT = {}
LAYER_DICT = {}
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
            pip install git+https://github.com/facebookresearch/segment-anything-2.git
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
            pip install git+https://github.com/huzeyann/segment-anything-2.git
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

class DiNO(nn.Module):
    def __init__(self, ver="dino_vitb8"):
        super().__init__()
        model = torch.hub.load('facebookresearch/dino:main', ver)
        model = model.eval()

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
        
    def forward(self, x):
        out = self.model(x)
        attn_outputs = [block.attn_output for block in self.model.blocks]
        mlp_outputs = [block.mlp_output for block in self.model.blocks]
        block_outputs = [block.block_output for block in self.model.blocks]
        return {
            'attn': attn_outputs,
            'mlp': mlp_outputs,
            'block': block_outputs
        }
            
MODEL_DICT["DiNO(dino_vits8[hi-res])"] = partial(DiNO, ver="dino_vits8")
LAYER_DICT["DiNO(dino_vits8[hi-res])"] = 12
RES_DICT["DiNO(dino_vits8[hi-res])"] = (896, 896)
MODEL_DICT["DiNO(dino_vitb8[hi-res])"] = partial(DiNO)
LAYER_DICT["DiNO(dino_vitb8[hi-res])"] = 12
RES_DICT["DiNO(dino_vitb8[hi-res])"] = (896, 896)

MODEL_DICT["DiNO(dino_vits8[mid-res])"] = partial(DiNO, ver="dino_vits8")
LAYER_DICT["DiNO(dino_vits8[mid-res])"] = 12
RES_DICT["DiNO(dino_vits8[mid-res])"] = (672, 672)
MODEL_DICT["DiNO(dino_vitb8[mid-res])"] = partial(DiNO)
LAYER_DICT["DiNO(dino_vitb8[mid-res])"] = 12
RES_DICT["DiNO(dino_vitb8[mid-res])"] = (672, 672)

MODEL_DICT["DiNO(dino_vits8)"] = partial(DiNO, ver="dino_vits8")
LAYER_DICT["DiNO(dino_vits8)"] = 12
RES_DICT["DiNO(dino_vits8)"] = (448, 448)
MODEL_DICT["DiNO(dino_vitb8)"] = partial(DiNO)
LAYER_DICT["DiNO(dino_vitb8)"] = 12
RES_DICT["DiNO(dino_vitb8)"] = (448, 448)
MODEL_DICT["DiNO(dino_vits16)"] = partial(DiNO, ver="dino_vits16")
LAYER_DICT["DiNO(dino_vits16)"] = 12
RES_DICT["DiNO(dino_vits16)"] = (448, 448)
MODEL_DICT["DiNO(dino_vitb16)"] = partial(DiNO, ver="dino_vitb16")
LAYER_DICT["DiNO(dino_vitb16)"] = 12
RES_DICT["DiNO(dino_vitb16)"] = (448, 448)


def resample_position_embeddings(embeddings, h, w):
    cls_embeddings = embeddings[0]
    patch_embeddings = embeddings[1:]  # [14*14, 768]
    hw = np.sqrt(patch_embeddings.shape[0]).astype(int)
    patch_embeddings = rearrange(patch_embeddings, "(h w) c -> c h w", h=hw)
    patch_embeddings = F.interpolate(patch_embeddings.unsqueeze(0), size=(h, w), mode="nearest").squeeze(0)
    patch_embeddings = rearrange(patch_embeddings, "c h w -> (h w) c")
    embeddings = torch.cat([cls_embeddings.unsqueeze(0), patch_embeddings], dim=0)
    return embeddings


class OpenCLIPViT(nn.Module):
    def __init__(self, version='ViT-B-16', pretrained='laion2b_s34b_b88k'):
        super().__init__()
        try:
            import open_clip
        except ImportError as e:
            s = f"""
            Import Error: {e}

            Please install open-clip-torch to use this model.
            pip install open-clip-torch
            """
            raise ImportError(s)
        
        model, _, _ = open_clip.create_model_and_transforms(version, pretrained=pretrained)
        
        positional_embedding = resample_position_embeddings(model.visual.positional_embedding, 42, 42)
        model.visual.positional_embedding = nn.Parameter(positional_embedding)
        
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
    
    def forward(self, x):
        out = self.model(x)
        attn_outputs, mlp_outputs, block_outputs = [], [], []
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
            pip install open-clip-torch
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
            pip install timm
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
    def __init__(self, size='base', **kwargs):
        super().__init__(**kwargs)

        try:
            import timm
        except ImportError as e:
            s = f"""
            Import Error: {e}

            Please install timm to use this model.
            pip install timm
            """
            raise ImportError(s)
        
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
        pos_embed = resample_position_embeddings(pos_embed, 42, 42)
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
        
        

MODEL_DICT["MAE(vit_base)"] = partial(MAE, size='base')
LAYER_DICT["MAE(vit_base)"] = 12
RES_DICT["MAE(vit_base)"] = (672, 672)
MODEL_DICT["MAE(vit_large)"] = partial(MAE, size='large')
LAYER_DICT["MAE(vit_large)"] = 24
RES_DICT["MAE(vit_large)"] = (672, 672)
MODEL_DICT["MAE(vit_huge)"] = partial(MAE, size='huge')
LAYER_DICT["MAE(vit_huge)"] = 32
RES_DICT["MAE(vit_huge)"] = (672, 672)


class ImageNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        try:
            import timm
        except ImportError as e:
            s = f"""
            Import Error: {e}

            Please install timm to use this model.
            pip install timm
            """
            raise ImportError(s)
            return
        
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
        return {
            'attn': attn_outputs,
            'mlp': mlp_outputs,
            'block': block_outputs
        }
            
MODEL_DICT["ImageNet(vit_base)"] = partial(ImageNet)
LAYER_DICT["ImageNet(vit_base)"] = 12
RES_DICT["ImageNet(vit_base)"] = (672, 672)

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


def load_model(model_name):
    model = MODEL_DICT[model_name]()
    resolution = RES_DICT[model_name]
    print(f"Loaded {model_name}, please use input resolution: {resolution}")
    return model
        
def list_models():
    return list(MODEL_DICT.keys())


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

    
if __name__ == '__main__':
    # inp = torch.rand(1, 3, 1024, 1024)
    # model = MAE()
    # out = model(inp)
    # print(out[0][0].shape, out[0][1].shape, out[0][2].shape)
    print(list_models())
    for model_name in list_models():
        if LAYER_DICT[model_name] == "not sure":
            model = MODEL_DICT[model_name]().cuda()
            print(f"Model: {model_name}")
            inp = torch.rand(1, 3, *RES_DICT[model_name]).cuda()
            out = model(inp)
            print(len(out['block']))

# %%
