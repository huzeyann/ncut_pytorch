import torch
from typing import Tuple
from ncut_pytorch.predictor.vision_predictor import NcutVisionPredictor
from ncut_pytorch.predictor.dino.transform import get_input_transform

AVAILABLE_MODELS = ["dinov3_l", "dinov3_s", "dinov3_b", "dinov3_vits14", "dinov2_s", "dinov2_b", "dinov2_reg4_s", "dinov3_s_plus", "dino_b", "clip_b", "siglip2_b", "vit_b", "radio_b", "radio_l", "radio_h"]

class NcutJafarPredictor(NcutVisionPredictor):
    def __init__(self,
                 model_name: str = "dinov3_l",
                 input_resolution: Tuple[int, int] = (512, 512),
                 output_resolution: Tuple[int, int] = (512, 512),
                 batch_size: int = 1):
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not found in AVAILABLE_MODELS")
        model = torch.hub.load("huzeyann/JAFAR", model_name, output_resolution=output_resolution, force_reload=True)
        transform = get_input_transform(resize=input_resolution)  
        super().__init__(model, transform, batch_size)
