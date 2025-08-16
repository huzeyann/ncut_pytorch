# %%
from ncut_pytorch.predictor import NcutDinoPredictor, NcutDinoPredictorSR
from PIL import Image

ncut_sam = NcutDinoPredictor(512)
ncut_sam = ncut_sam.to('cuda')


images = [Image.open("images/view_0.jpg"), Image.open("images/view_1.jpg"), Image.open("images/view_2.jpg")
            , Image.open("images/view_3.jpg"), Image.open("images/view_ego.jpg"), Image.open("images/image2.jpg")]
ncut_sam.set_images(images)
# %%
image = ncut_sam.summary(draw_border=False)
# %%
image
# %%
