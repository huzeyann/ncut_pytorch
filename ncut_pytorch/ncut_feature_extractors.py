from einops import rearrange
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch import nn


def image_dinov2_feature(images, resolution=(448, 448), layer=11):
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
    # extract DINOv2 last layer features from the image
    class DiNOv2Feature(torch.nn.Module):
        def __init__(self, ver="dinov2_vitb14_reg", layer=11):
            super().__init__()
            self.dinov2 = torch.hub.load("facebookresearch/dinov2", ver)
            self.dinov2.requires_grad_(False)
            self.dinov2.eval()
            self.dinov2 = self.dinov2.cuda()
            self.layer = layer

        def forward(self, x):
            out = self.dinov2.get_intermediate_layers(x, reshape=True, n=np.arange(12))[self.layer]
            return out

    feat_extractor = DiNOv2Feature(layer=layer)

    feats = []
    for i, image in enumerate(images):
        torch_image = transform(image)
        feat = feat_extractor(torch_image.unsqueeze(0).cuda()).cpu()
        feat = feat.squeeze(0).permute(1, 2, 0)
        feats.append(feat)
    feats = torch.stack(feats).squeeze(0)
    return feats


    
class SAM(torch.nn.Module):
    def __init__(self, n=7, checkpoint="/data/sam_model/sam_vit_b_01ec64.pth", **kwargs):
        super().__init__(**kwargs)
        self.n = n
        from segment_anything import sam_model_registry, SamPredictor
        from segment_anything.modeling.sam import Sam
        sam: Sam = sam_model_registry["vit_b"](
            checkpoint=checkpoint
        )
        
        def new_forward(self, x: torch.Tensor, n) -> torch.Tensor:
            x = self.patch_embed(x)
            if self.pos_embed is not None:
                x = x + self.pos_embed
            # pool = torch.nn.AvgPool3d(4)
            ret_dict, cls_dict = {}, {}
            for i, blk in enumerate(self.blocks):
                if i not in n:
                    break
                x = blk(x)
                x_save = x.clone()
                # x_save.unsqueeze(1)
                # x_save = pool(x_save)
                # ret_dict[f'layer{i}'] = x_save.squeeze(1)
                x_save = x_save.permute(0, 3, 1, 2)
                # x_save = torch.nn.functional.interpolate(x_save, size=(8, 8), mode="bilinear")
                # x_save = x_save.squeeze(0)
                ret_dict[f"{i}"] = x_save
                cls_dict[f"{i}"] = x_save.mean(dim=(2, 3))

            # x = self.neck(x.permute(0, 3, 1, 2))

            return ret_dict, cls_dict

        setattr(sam.image_encoder.__class__, "forward", new_forward)
        
        self.image_encoder = sam.image_encoder
        self.image_encoder.requires_grad_(True)
        self.image_encoder.train()
        self.image_encoder = self.image_encoder.cuda()
        
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # up sample to 1024
        with torch.no_grad():
            x = torch.nn.functional.interpolate(x, size=(1024, 1024), mode="bilinear")
        n = np.arange(np.max([self.n])+1)
        ret_dict, cls_dict = self.image_encoder(x, n)
        return ret_dict[str(self.n)]


def image_sam_feature(images, resolution=(1024, 1024), layer=11, checkpoint="/data/sam_model/sam_vit_b_01ec64.pth"):
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

    feat_extractor = SAM(n=layer, checkpoint=checkpoint)

    feats = []
    for i, image in enumerate(images):
        torch_image = transform(image)
        feat = feat_extractor(torch_image.unsqueeze(0).cuda()).cpu()
        feat = feat.squeeze(0).permute(1, 2, 0)
        feats.append(feat)
    feats = torch.stack(feats).squeeze(0)
    return feats


class VideoMAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        try:
            from transformers import VideoMAEForVideoClassification
        except ImportError:
            raise ImportError(
                "Please install the transformers library: pip install transformers"
            )

        self.model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )
        self.model.requires_grad_(False)
        self.model.eval()

    def forward(self, x):
        assert x.dim() == 5
        assert x.shape[1:] == (16, 3, 224, 224)  # frame, color channel, height, width

        outputs = self.model(x, output_hidden_states=True, return_dict=True)
        last_layer = outputs.hidden_states[-1]
        return last_layer


from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np


def transform_images(frames, size=(224, 224)):
    resized = []
    length = len(frames)
    for i in range(length):
        frame = frames[i]
        # image = Image.fromarray((frame * 255).astype(np.uint8))
        image = Image.fromarray(frame)
        image = image.resize(size, Image.ANTIALIAS)
        image = np.array(image) / 255.0
        resized.append(np.array(image))
    frames = np.stack(resized, axis=0)
    frames = frames.transpose(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    frames = torch.tensor(frames, dtype=torch.float32)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    return frames


def read_video(video_path: str) -> torch.Tensor:
    try:
        from decord import VideoReader
    except ImportError:
        raise ImportError("Please install the decord library: pip install decord")

    vr = VideoReader(video_path)
    print(f"Total frames: {len(vr)}")
    # frames = vr.get_batch(range(len(vr))).asnumpy()
    lenth = len(vr)
    lenth = 1600 if lenth > 1600 else lenth
    frames = vr.get_batch(np.arange(lenth)).asnumpy()
    # if less than 1600 frames, repeat the last frame
    if lenth < 1600:
        last_frame = frames[-1]
        for i in range(1600 - lenth):
            frames = np.append(frames, last_frame.reshape(1, *last_frame.shape), axis=0)
    # frames = np.array(frames)
    frames = transform_images(frames)
    return frames


def video_mae_feature(video_path):
    frames = read_video(video_path)
    videomae = VideoMAE()
    videomae = videomae.cuda()
    frames = frames.cuda()
    frames = rearrange(frames, "(b t) c h w -> b t c h w", affinity_focal_gamma=16)
    feats = videomae(frames)
    return feats


from typing import List
import torch
import os


class Llama3:
    def __init__(
        self,
        ckpt_dir="/data/Meta-Llama-3-8B",
        tokenizer_path="/data/Meta-Llama-3-8B/tokenizer.model",
        max_batch_size=4,
        max_seq_len=128,
    ):
        from llama import Llama

        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "11451"
        torch.distributed.init_process_group("nccl")

        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )

    @torch.no_grad()
    def get_feats(self, prompts: List[str]):

        list_attns, list_h = [], []
        prompt_tokens, prompt_texts = [], []
        for prompt in prompts:
            out = self.generator.text_completion(prompts=[prompt], max_gen_len=1)
            _attns = self.generator.saved_attns
            _h = self.generator.saved_h
            _prompt_tokens = self.generator.prompt_tokens
            _prompt_texts = self.generator.prompt_texts
            for i, prompt in enumerate(_prompt_tokens):
                list_attns.append(_attns[i][:, : len(prompt)])
                list_h.append(_h[i][: len(prompt)])
                prompt_tokens.append(prompt)
                prompt_texts.append(_prompt_texts[i])
        return list_attns, list_h, prompt_tokens, prompt_texts


def llama3_feature(text_string):
    assert isinstance(text_string, str), "Input must be a string."
    
    text_string = text_string.replace("\n\n", "\n")
    lines = text_string.strip().split("\n")

    llama3 = Llama3()
    list_attns, list_h, prompt_tokens, prompt_texts = llama3.get_feats(lines)
    features = torch.cat(list_h, dim=0)
    return features, prompt_tokens, prompt_texts
