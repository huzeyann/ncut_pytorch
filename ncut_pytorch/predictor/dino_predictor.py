import torch

from ncut_pytorch.predictor.vision_predictor import NcutVisionPredictor

from ncut_pytorch.predictor.dino import hires_dino_256, hires_dino_512, hires_dino_1024
from ncut_pytorch.predictor.dino import LowResDINO
from ncut_pytorch.predictor.dino.transform import get_input_transform

SUPER_RESOLUTION_MODELS = {
    256: hires_dino_256,
    512: hires_dino_512,
    1024: hires_dino_1024,
}

class NcutDinoPredictorSR(NcutVisionPredictor):
    def __init__(self,
                 input_size: int = 512,
                 dtype: torch.dtype = torch.float16,
                 batch_size: int = 1):
        model, transform = SUPER_RESOLUTION_MODELS[input_size](dtype=dtype)
        super().__init__(model, transform, batch_size)


class NcutDinoPredictorFeatUp(NcutVisionPredictor):
    def __init__(self,
                 input_size: int = 512,
                 dtype: torch.dtype = torch.float16,
                 batch_size: int = 8):
        model = torch.hub.load("huzeyann/FeatUp", 'dino', use_norm=False)
        transform = get_input_transform(resize=input_size)
        super().__init__(model, transform, batch_size)


class NcutDinoPredictor(NcutVisionPredictor):
    def __init__(self,
                 input_size: int = 512,
                 dtype: torch.dtype = torch.float16,
                 batch_size: int = 8):
        model = LowResDINO(dtype=dtype)
        transform = get_input_transform(resize=input_size)
        super().__init__(model, transform, batch_size)




# class NcutDinoPredictor(NcutVisionPredictor):
#     def __init__(self,
#                  input_size: int = 512,
#                  dtype: torch.dtype = torch.float16,
#                  super_resolution: bool = True,
#                  batch_size: int = 32):
#         super_resolution_model, transform = SUPER_RESOLUTION_MODELS[input_size](dtype=dtype)
#         low_resolution_model = LowResDINO(dtype=dtype)
#         model = super_resolution_model if super_resolution else low_resolution_model
#         super().__init__(model, transform, batch_size)

#TODO: add FeatUp dino