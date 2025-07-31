import torch

from ncut_pytorch.predictor.vision_predictor import NcutVisionPredictor

from ncut_pytorch.predictor.dino import hires_dino_256, hires_dino_512, hires_dino_1024
from ncut_pytorch.predictor.dino import LowResDINO

MODEL_REGISTRY = {
    256: hires_dino_256,
    512: hires_dino_512,
    1024: hires_dino_1024,
}

class NcutDinoPredictor(NcutVisionPredictor):
    def __init__(self,
                 input_size: int = 512,
                 dtype: torch.dtype = torch.float16,
                 run_faster: bool = False,
                 batch_size: int = 32):
        model, transform = MODEL_REGISTRY[input_size](dtype=dtype)
        super().__init__(model, transform, batch_size)
        self.dtype = dtype
        if run_faster:
            self.run_faster()

    def run_faster(self):
        model = LowResDINO(dtype=self.dtype)
        device = next(self.model.parameters()).device
        model = model.to(device)
        self.model = model
