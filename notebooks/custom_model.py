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
    def __init__(self, model_cfg='vit_b', **kwargs):
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
        return out

if __name__ == "__main__":
    transform = get_input_transform(resize=1024)
    model = SAM()
    ncut_sam = NcutVisionPredictor(model, transform, batch_size=1)
    ncut_sam = ncut_sam.to('mps')

    image = Image.open("images/view_0.jpg")
    ncut_sam.set_images([image])
    segments = ncut_sam.generate(n_cluster=32)
    color = ncut_sam.color_discrete(segments, draw_border=True)
    plt.imshow(color[0])
    plt.show()

