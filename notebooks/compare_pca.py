# %%
from ncut_pytorch.predictor import NcutDinoPredictor, NcutDinoPredictorSR, NcutDinov3Predictor, NcutDinoPredictorFeatUp
from ncut_pytorch.predictor import NcutVisionPredictor
from PIL import Image

from ncut_pytorch.predictor.dino.api import hires_dino_256, hires_dino_512, hires_dino_1024, hires_dinov2
from ncut_pytorch.predictor.dino.transform import get_input_transform
# ncut_sam = NcutDinov3Predictor(model_cfg="dinov3_vits16")
# ncut_sam = NcutDinoPredictorFeatUp()

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
        return out

sam = SAM()
transform = get_input_transform(resize=1024)
sam_predictor = NcutVisionPredictor(sam, transform, 1)
sam_predictor = sam_predictor.to('cuda')


dinov2reg, transform = hires_dinov2()
dinov2reg.chunk_size = 1
dinov2reg.feature_resolution = 448
transform = get_input_transform(resize=(448, 448))
dinov2reg_predictor = NcutVisionPredictor(dinov2reg, transform, 1)
dinov2reg_predictor = dinov2reg_predictor.to('cuda')

from ncut_pytorch.predictor.dino.api import hires_dino
model = hires_dino(dino_name="dinov2_vitb14",
                    stride=6,
                    shift_dists=[1, 2, 3],
                    flip_transforms=True,
                    chunk_size=1,
                    dtype=torch.float16,
                    feature_resolution=1008)
transform = get_input_transform(resize=1008)
dinov2_predictor = NcutVisionPredictor(model, transform, 1)
dinov2_predictor = dinov2_predictor.to('cuda')

dinov1_predictor = NcutDinoPredictorSR()

dinov3h_predictor = NcutDinov3Predictor(model_cfg="dinov3_vith16plus")
dinov3l_predictor = NcutDinov3Predictor(model_cfg="dinov3_vitl16")
dinov3b_predictor = NcutDinov3Predictor(model_cfg="dinov3_vitb16")
dinov3s_predictor = NcutDinov3Predictor(model_cfg="dinov3_vits16")

dinov3sat_predictor = NcutDinov3Predictor(model_cfg="dinov3_vitl16_sat493m")

NAME_TO_PREDICTOR = {
    "dinov3": dinov3l_predictor,
    "dinov2reg": dinov2reg_predictor,
    "dinov2": dinov2_predictor,
    "dinov1": dinov1_predictor,
    "sam": sam_predictor,
    "dinov3h": dinov3h_predictor,
    "dinov3l": dinov3l_predictor,
    "dinov3b": dinov3b_predictor,
    "dinov3s": dinov3s_predictor,
    "dinov3sat": dinov3sat_predictor,
}


images = [Image.open("images/view_0.jpg"), Image.open("images/view_1.jpg"), Image.open("images/view_2.jpg")
            , Image.open("images/view_3.jpg"), Image.open("images/view_ego.jpg"), Image.open("images/image2.jpg")]
#%%
# images = [Image.open("images/view_0.jpg"), Image.open("images/view_2.jpg"), Image.open("images/view_ego.jpg")]
images = [Image.open(f"images/ducks/single_{i:04d}.jpg") for i in range(20)]
# %%
compare_models = ["dinov3h", "dinov3l", "dinov3b", "dinov3s"]
compare_models = ["dinov3sat", "dinov3h", "dinov3l", "dinov3b", "dinov3s"]
compare_models = ["dinov3", "dinov2reg", "dinov2", "dinov1", "sam"]
compare_models = ["dinov3", "dinov1"]
n_col = len(compare_models) + 1
n_row = 3
fig_size_h = 3.5 * n_row
fig_size_w = 3.5 * n_col

fig, axs = plt.subplots(n_row, n_col, figsize=(fig_size_w, fig_size_h))

for i, image in enumerate(images):
    if i > 2:
        break
    axs[i, 0].imshow(image)
    # axs[i, 0].set_title(f"Image {i}")

use_pca = False

for i, model in enumerate(compare_models):
    predictor = NAME_TO_PREDICTOR[model]
    predictor = predictor.to('cuda')
    # transform = get_input_transform(resize=2048)
    # predictor.transform = transform
    # predictor.set_images(images)
    # predictor.refresh_color_palette()
    if use_pca:
        _features = predictor.predictor._features.float()
        from ncut_pytorch.utils.math import pca_lowrank
        embeds = pca_lowrank(_features, q=3)
        from ncut_pytorch.color.coloring import rgb_from_3d_rgb_cube
        color = rgb_from_3d_rgb_cube(embeds)
        predictor.predictor._color_palette = color
    segments = predictor.generate(n_segment=50 if model != "sam" else 20)
    color = predictor.color_discrete(segments)
    # color = predictor.color_continues()
    for i_img, img in enumerate(color):
        if i_img > 2:
            break
        axs[i_img, i+1].imshow(img)
    axs[0, i+1].set_title(f"{model}", fontsize=20)
    predictor = predictor.to('cpu')

for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()
# %%
