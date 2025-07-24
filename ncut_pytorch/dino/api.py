from typing import Tuple
from torchvision import transforms
from .hires_dino import hires_dino
from .hires_dino import HighResDINO
from .transform import get_input_transform



def hires_dino_256() -> Tuple[HighResDINO, transforms.Compose]:
    model = hires_dino(dino_name="dino_vitb8", 
                    stride=6, 
                    shift_dists=[1, 2, 3],
                    flip_transforms=True,
                    chunk_size=6,
                    feature_resolution=256)
    transform = get_input_transform(resize=256)
    return model, transform
    

def hires_dino_512() -> Tuple[HighResDINO, transforms.Compose]:
    model = hires_dino(dino_name="dino_vitb8", 
                    stride=6, 
                    shift_dists=[1, 2, 3],
                    flip_transforms=True,
                    chunk_size=4,
                    feature_resolution=512)
    transform = get_input_transform(resize=512)
    return model, transform


def hires_dino_1024() -> Tuple[HighResDINO, transforms.Compose]:
    model = hires_dino(dino_name="dino_vitb8", 
                    stride=6, 
                    shift_dists=[1, 2, 3],
                    flip_transforms=True,
                    chunk_size=1,
                    feature_resolution=1024)
    transform = get_input_transform(resize=1024)
    return model, transform


def hires_dinov2() -> Tuple[HighResDINO, transforms.Compose]:
    model = hires_dino(dino_name="dinov2_vitb14_reg", 
                    stride=6, 
                    shift_dists=[1, 2, 3],
                    flip_transforms=True,
                    chunk_size=1,
                    feature_resolution=1008)
    transform = get_input_transform(resize=1008)
    return model, transform