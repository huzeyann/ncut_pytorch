# Tutorial 6 - Custom Model

Whenever you want to integrate your own model into Ncut package, simply define the `__init__` and `forward` function as shown below. Then you can use all functions in NCut.

``` py

import torch
from PIL import Image
from matplotlib import pyplot as plt
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import Sam

from ncut_pytorch.predictor.dino.transform import get_input_transform
from ncut_pytorch.predictor.vision_predictor import NcutVisionPredictor

URL_DICT = {
    'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

class SAM(torch.nn.Module):
    def __init__(self, model_cfg='vit_l', **kwargs):
        super().__init__(**kwargs)

        statedict = torch.hub.load_state_dict_from_url(URL_DICT[model_cfg], map_location='cpu')
        sam: Sam = sam_model_registry[model_cfg]()
        sam.load_state_dict(statedict)
        sam = sam.eval()
        self.sam = sam

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = torch.nn.functional.interpolate(x, size=(1024, 1024), mode="bilinear")
        out = self.sam.image_encoder(x)
        out = torch.nn.functional.normalize(out, dim=-1)
        return out  # (B, C, H, W)

if __name__ == "__main__":
    transform = get_input_transform(resize=1024)
    model = SAM()
    ncut_sam = NcutVisionPredictor(model, transform, batch_size=1)
    ncut_sam = ncut_sam.to('cuda')

    images = [Image.open(f"images/pose/single_{i:04d}.jpg") for i in range(20)]

    ncut_sam.set_images(images)
    
    image = ncut_sam.summary(n_segments=[10, 25, 50, 100])
    display(image)

```

