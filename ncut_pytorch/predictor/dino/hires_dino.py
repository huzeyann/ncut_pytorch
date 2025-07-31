# Credits: https://github.com/tldr-group/HR-Dv2

from functools import partial
from types import MethodType
from typing import List, Tuple, Literal, Union

# ==================== IMPORTS ====================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .patch import Patch
from .transform import combine_transforms_pairwise
from .transform import iden_partial, get_shift_transforms, get_flip_transforms

Interpolation = Literal[
    "nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"
]
AttentionOptions = Literal["q", "k", "v", "o", "none"]
DINONameOptions = Literal["dino_vitb8", "dino_vits8", "dino_vitb16", "dinov2_vitb14", "dinov2_vitb14_reg"]

# ==================== MODULE ====================


class HighResDINO(nn.Module):
    def __init__(
        self,
        dino_name: DINONameOptions,
        stride: int = 5,
        dtype: Union[torch.dtype, int] = torch.float16,
        track_grad: bool = False,
        attention_mask_ratio: float = 0.1,
        chunk_size: int = 4,
        feature_resolution: int = 1024,
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

        # Get params of Dv2 model and store references to original settings & methods
        feat, patch = self.get_model_params(dino_name)
        self.original_patch_size: int = patch
        self.original_stride = _pair(patch)
        # may need to deepcopy this instead of just referencing
        # self.original_pos_enc = self.dinov2.interpolate_pos_encoding
        self.feat_dim: int = feat
        self.n_heads: int = 6
        self.n_register_tokens = 4
        self.chunk_size = chunk_size
        self.feature_resolution = feature_resolution
        
        # mask out some of the attention Keys (K) to save compute
        self.attention_mask_ratio = attention_mask_ratio

        self.stride = _pair(stride)
        # we need to set the stride to the original once before we set it to desired stride
        # i don't know why
        self.set_model_stride(self.dinov2, patch)
        self.set_model_stride(self.dinov2, stride)

        self.transforms: List[partial] = []
        self.inverse_transforms: List[partial] = []
        self.interpolation_mode: Interpolation = "nearest-exact"

        # If we want to save memory, change to float16
        if type(dtype) == int:
            dtype = torch.float16 if dtype == 16 else torch.float32

        self.dtype = dtype
        if dtype != torch.float32:
            self = self.to(dtype)
        self.track_grad = track_grad  # off by default to save memory

        self.patch_last_block(self.dinov2, dino_name)

    def get_model_params(self, dino_name: str) -> Tuple[int, int]:
        """Match a name like dinov2_vits14 / dinov2_vitg16_lc etc. to feature dim and patch size.

        :param dino_name: string of dino model name on torch hub
        :type dino_name: str
        :return: tuple of original patch size and hidden feature dimension
        :rtype: Tuple[int, int]
        """
        split_name = dino_name.split("_")
        model = split_name[1]
        arch, patch_size = model[3], int(model[4:])
        feat_dim_lookup = {"s": 384, "b": 768, "l": 1024, "g": 1536}
        feat_dim: int = feat_dim_lookup[arch]
        return feat_dim, patch_size

    def set_model_stride(
        self, dino_model: nn.Module, stride_l: int, verbose: bool = False
    ) -> None:
        """Create new positional encoding interpolation method for $dino_model with
        supplied $stride, and set the stride of the patch embedding projection conv2D
        to $stride.

        :param dino_model: Dv2 model
        :type dino_model: DinoVisionTransformer
        :param new_stride: desired stride, usually stride < original_stride for higher res
        :type new_stride: int
        :return: None
        :rtype: None
        """

        new_stride_pair = torch.nn.modules.utils._pair(stride_l)
        # if new_stride_pair == self.stride:
        #    return  # early return as nothing to be done
        self.stride = new_stride_pair
        dino_model.patch_embed.proj.stride = new_stride_pair  # type: ignore
        if verbose:
            print(f"Setting stride to ({stride_l},{stride_l})")

        # if new_stride_pair == self.original_stride:
        # if resetting to original, return original method
        #    dino_model.interpolate_pos_encoding = self.original_pos_enc  # type: ignore
        # else:
        dino_model.interpolate_pos_encoding = MethodType(  # type: ignore
            Patch._fix_pos_enc(self.original_patch_size, new_stride_pair),
            dino_model,
        )  # typed ignored as they can't type check reassigned methods (generally is poor practice)

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

    def get_n_patches(self, img_h: int, img_w: int) -> Tuple[int, int]:
        stride_l = self.stride[0]
        n_patch_h: int = 1 + (img_h - self.original_patch_size) // stride_l
        n_patch_w: int = 1 + (img_w - self.original_patch_size) // stride_l
        return (n_patch_h, n_patch_w)

    def set_transforms(
        self, transforms: List[partial], inv_transforms: List[partial]
    ) -> None:
        assert len(transforms) == len(
            inv_transforms
        ), "Each transform must have an inverse!"
        self.transforms = transforms
        self.inverse_transforms = inv_transforms

    @torch.no_grad()
    def get_transformed_input_batch(
        self, x: torch.Tensor, transforms: List[partial]
    ) -> torch.Tensor:
        """Loop through a list of (invertible) transforms, apply them to input $x, store
        in a list then batch and return.

        :param x: input unbatched standardized image tensor
        :type x: torch.Tensor
        :param transforms: list of partial functions representing transformations on image
        :type transforms: List[partial]
        :return: batch of transformed images
        :rtype: torch.Tensor
        """
        img_list: List[torch.Tensor] = []
        if len(transforms) == 0:  # if we want to test VE by itself
            # print("Warning: no transforms supplied, using identity transform")
            self.transforms.append(iden_partial)
            self.inverse_transforms.append(iden_partial)

        for transform in transforms:
            transformed_img: torch.Tensor = transform(x)
            img_list.append(transformed_img)

        if len(img_list[0].shape) == 3:  # if c, h, w then stack list
            img_batch = torch.stack(img_list)
        else:  # if b, c, h, w then cat
            img_batch = torch.cat(img_list)

        if self.dtype != torch.float32:
            img_batch = img_batch.to(self.dtype)
        return img_batch

    @torch.no_grad()
    def invert_transforms(
        self, feature_batch: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """For each flat Dv2 features of our transformed imgs in $feature_batch, loop through,
        make them spatial again by reshaping, permuting and resizing, then perform the
        corresponding inverse transform and add to our summand variable. Finally we divide by
        N_imgs to create average.

        # TODO: parameterise this with the inverse transform s.t can just feed single batch (of
        say the attn map) and the iden partial transform and get upsampled

        :param feature_batch: batch of N_transform features from Dv2 with shape (n_patches, n_features)
        :type feature_batch: torch.Tensor
        :param x: original input img of size (channels, img_h, img_w), useful for resizing later
        :type x: torch.Tensor
        :return: feature image of size (n_features, img_h, img_w)
        :rtype: torch.Tensor
        """
        _, img_h, img_w = x.shape
        c = feature_batch.shape[-1]
        stride_l = self.stride[0]
        n_patch_w: int = 1 + (img_w - self.original_patch_size) // stride_l
        n_patch_h: int = 1 + (img_h - self.original_patch_size) // stride_l

        # Summand variable here to be memory efficient
        out_feature_img: torch.Tensor = torch.zeros(
            1,
            c,
            self.feature_resolution,
            self.feature_resolution,
            device=x.device,
            dtype=self.dtype,
            requires_grad=self.track_grad,
        )

        for i, inv_transform in enumerate(self.inverse_transforms):
            feat_patch_flat = feature_batch[i]
            # interp expects batched spatial tensors so reshape and unsqueeze
            feat_patch = feat_patch_flat.view((n_patch_h, n_patch_w, c))

            permuted = feat_patch.permute((2, 0, 1)).unsqueeze(0)

            full_size = F.interpolate(
                permuted,
                (img_h, img_w),
                mode=self.interpolation_mode,
            )
            inverted: torch.Tensor = inv_transform(full_size)
            
            # resize the inverted feature map to the output resolution
            out = F.interpolate(
                inverted, 
                (self.feature_resolution, self.feature_resolution), 
                mode=self.interpolation_mode
            )
            
            out_feature_img += out

        n_imgs: int = feature_batch.shape[0]
        mean = out_feature_img / n_imgs
        mean = mean.squeeze(0)
        return mean

    def _forward_one_image(
        self,
        x: torch.Tensor,
        attn_choice: AttentionOptions = "none",
    ) -> torch.Tensor:
        """Feed input img $x through network and get high-res features.

        :param x: unbatched image tensor, (c, h, w)
        :param attn_choice: choice of attention, "none" or "q", "k", "v", "o"
        :return: upsampled features, (c, h, w)
        """
        if self.dtype != torch.float32:  # cast (i.e to f16)
            x = x.type(self.dtype)

        img_batch = self.get_transformed_input_batch(x, self.transforms)
        N_imgs = img_batch.shape[0]
        
        all_features = []
        for i in range(0, N_imgs, self.chunk_size):
            _img_batch = img_batch[i:i+self.chunk_size]
            out_dict = self.dinov2.forward_feats_attn(_img_batch, None, attn_choice)  # type: ignore
            if attn_choice != "none":
                feats, attn = out_dict["x_norm_patchtokens"], out_dict["x_patchattn"]
                features_batch = torch.concat((feats, attn), dim=-1)
            else:
                features_batch = out_dict["x_norm_patchtokens"]

            if self.dtype != torch.float32:  # cast (i.e to f16)
                features_batch = features_batch.type(self.dtype)
            
            all_features.append(features_batch)

        features_batch = torch.cat(all_features, dim=0)
        upsampled_features = self.invert_transforms(features_batch, x)
        return upsampled_features
    
    def _forward_auto_chunk(self, *args, **kwargs) -> torch.Tensor:
        try:
            return self._forward_one_image(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError("CUDA out of memory, please try to reduce the chunk size.\n    hires_dino_model.chunk_size = 1")
        
    def forward(
        self,
        x: torch.Tensor,
        attn_choice: AttentionOptions = "none",
        move_to_cpu: bool = True,
    ) -> torch.Tensor:
        """Feed input img $x through network and get low and high res features.

        :param x: batched image tensor, (b, c, h, w)
        :param attn_choice: choice of attention, "none" or "q", "k", "v", "o"
        :return: upsampled features, (b, c, h, w)
        :rtype: torch.Tensor
        """
        upsampled_features = []
        for i in range(x.shape[0]):
            if self.track_grad:
                out = self._forward_auto_chunk(x[i], attn_choice)
            else:
                with torch.no_grad():
                    out = self._forward_auto_chunk(x[i], attn_choice)
                    if move_to_cpu:
                        out = out.cpu()
            upsampled_features.append(out)
        upsampled_features = torch.stack(upsampled_features, dim=0)
        return upsampled_features

    

def hires_dino(dino_name: DINONameOptions = "dino_vitb8",
               stride: int = 6,
               shift_dists: List[int] = [1, 2, 3],
               flip_transforms: bool = True,
               attention_mask_ratio: float = 0.1,
               dtype: Union[torch.dtype, int] = torch.float16,
               track_grad: bool = False,
               chunk_size: int = 4,
               feature_resolution: int = 512
               ) -> HighResDINO:
    """
    Args:
        dino_name: name of the DINO model to use
        stride: stride size of the tokenization, smaller is better but slower
        shift_dists: pixel shifts for multiple image transformations, more shifts means more crispy features
        flip_transforms: whether to use flip transforms, remove positional features
        attention_mask_ratio: ratio of attention keys to mask out
        dtype: data type of the model
        track_grad: whether to track gradients
        chunk_size: number of images to process in one batch, in case of OOM
        feature_resolution: resolution of the output features
    """

    model = HighResDINO(dino_name, stride, dtype, track_grad, attention_mask_ratio, chunk_size, feature_resolution)

    fwd_shift, inv_shift = get_shift_transforms(shift_dists)
    if flip_transforms:  # add flip transforms
        fwd_flip, inv_flip = get_flip_transforms()
        fwd, inv = combine_transforms_pairwise(fwd_shift, fwd_flip, inv_shift, inv_flip)
    else:
        fwd, inv = fwd_shift, inv_shift
    model.set_transforms(fwd, inv)

    model = model.eval()

    return model


