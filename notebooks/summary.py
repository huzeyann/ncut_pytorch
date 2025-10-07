# %%
from ncut_pytorch.predictor import NcutDinoPredictor, NcutDinoPredictorSR, NcutDinov3Predictor, NcutDinoPredictorFeatUp
from ncut_pytorch.predictor import NcutVisionPredictor
from PIL import Image

from ncut_pytorch.predictor.dino.api import hires_dino_256, hires_dino_512, hires_dino_1024, hires_dinov2
from ncut_pytorch.predictor.dino.transform import get_input_transform
ncut_sam = NcutDinov3Predictor(model_cfg="dinov3_vitl16")
ncut_sam = ncut_sam.to('cuda')
ncut_sam.predictor.color_method = 'tsne'


images = [Image.open("images/view_0.jpg"), Image.open("images/view_1.jpg"), Image.open("images/view_2.jpg")
            , Image.open("images/view_3.jpg"), Image.open("images/view_ego.jpg"), Image.open("images/image2.jpg")]

ncut_sam.set_images(images)
# %%
image = ncut_sam.summary(n_segments=[10, 25, 50, 100], draw_border=True)
# %%
image
# %%
